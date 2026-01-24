from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncIterator
import os
import redis
import json
import asyncio

# LangChain imports
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.callbacks.base import AsyncCallbackHandler
from pinecone import Pinecone

app = FastAPI(title="Advanced RAG Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Configuration ====================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# ==================== LangChain Setup ====================
llm = Ollama(
    base_url=OLLAMA_HOST,
    model=OLLAMA_MODEL,
    temperature=0.7
)

llm_no_stream = Ollama(
    base_url=OLLAMA_HOST,
    model=OLLAMA_MODEL,
    temperature=0.3
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

# ==================== Streaming Callback ====================
class StreamingCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        self.tokens = []
        self.finish = False

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

    async def on_llm_end(self, response, **kwargs) -> None:
        self.finish = True

# ==================== Models ====================
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    retrieval_strategy: str = "similarity"
    top_k: int = 5
    stream: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str
    strategy_used: str

# ==================== Retrieval Strategies ====================
def get_retriever(strategy: str, k: int = 5):
    if strategy == "similarity":
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    elif strategy == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": k * 3,
                "lambda_mult": 0.5
            }
        )

    elif strategy == "similarity_score_threshold":
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": 0.7
            }
        )

    elif strategy == "multi_query":
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm_no_stream,
            include_original=True
        )

    elif strategy == "compression":
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": k * 2})
        compressor = LLMChainExtractor.from_llm(llm_no_stream)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

    else:
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

# ==================== Session Management ====================
def get_memory(session_id: str) -> ConversationBufferMemory:
    history_key = f"chat_history:{session_id}"
    history = redis_client.lrange(history_key, 0, -1)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    for msg in history:
        msg_data = json.loads(msg)
        if msg_data["role"] == "user":
            memory.chat_memory.add_user_message(msg_data["content"])
        else:
            memory.chat_memory.add_ai_message(msg_data["content"])

    return memory

def save_to_history(session_id: str, role: str, content: str):
    history_key = f"chat_history:{session_id}"
    redis_client.rpush(
        history_key,
        json.dumps({"role": role, "content": content})
    )
    redis_client.expire(history_key, 3600)

# ==================== Prompt Template ====================
CHAT_PROMPT = PromptTemplate(
    template="""You are a helpful AI assistant. Use the following context to answer the question accurately.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Instructions:
- Answer based on the context provided
- If the answer isn't in the context, say "I don't have enough information"
- Be concise and direct

Answer:""",
    input_variables=["context", "chat_history", "question"]
)

# ==================== Endpoints ====================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "rag-service",
        "features": ["streaming", "multi_query", "mmr", "compression"],
        "ollama_model": OLLAMA_MODEL,
        "pinecone_index": PINECONE_INDEX
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        import uuid
        session_id = request.session_id or str(uuid.uuid4())

        retriever = get_retriever(request.retrieval_strategy, request.top_k)
        memory = get_memory(session_id)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": CHAT_PROMPT},
            verbose=True
        )

        if request.stream:
            return StreamingResponse(
                stream_response(qa_chain, request.query, session_id, request.retrieval_strategy),
                media_type="text/event-stream"
            )
        else:
            result = qa_chain({"question": request.query})

            save_to_history(session_id, "user", request.query)
            save_to_history(session_id, "assistant", result["answer"])

            sources = [
                {
                    "content": doc.page_content[:500],
                    "metadata": doc.metadata,
                }
                for doc in result.get("source_documents", [])
            ]

            return ChatResponse(
                answer=result["answer"],
                sources=sources,
                session_id=session_id,
                strategy_used=request.retrieval_strategy
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_response(qa_chain, query: str, session_id: str, strategy: str) -> AsyncIterator[str]:
    try:
        callback = StreamingCallbackHandler()

        result = await asyncio.to_thread(
            qa_chain,
            {"question": query},
            callbacks=[callback]
        )

        full_answer = ""
        for token in callback.tokens:
            full_answer += token
            yield f"data: {json.dumps({'token': token, 'type': 'token'})}\n\n"
            await asyncio.sleep(0.01)

        save_to_history(session_id, "user", query)
        save_to_history(session_id, "assistant", full_answer)

        sources = [
            {
                "content": doc.page_content[:500],
                "metadata": doc.metadata,
            }
            for doc in result.get("source_documents", [])
        ]

        yield f"data: {json.dumps({'sources': sources, 'type': 'sources', 'strategy': strategy})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e), 'type': 'error'})}\n\n"

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    redis_client.delete(f"chat_history:{session_id}")
    return {"status": "success", "message": "Session cleared"}

@app.get("/retrieval-strategies")
async def list_strategies():
    return {
        "strategies": {
            "similarity": {
                "description": "Standard cosine similarity search",
                "best_for": "Straightforward questions",
                "speed": "fast"
            },
            "mmr": {
                "description": "Maximal Marginal Relevance",
                "best_for": "Diverse perspectives",
                "speed": "medium"
            },
            "multi_query": {
                "description": "Multiple query variations",
                "best_for": "Complex questions",
                "speed": "slow"
            },
            "compression": {
                "description": "LLM-compressed results",
                "best_for": "Long documents",
                "speed": "slow"
            }
        }
    }

@app.post("/test-strategies")
async def test_strategies(query: str, top_k: int = 5):
    strategies = ["similarity", "mmr", "multi_query", "compression"]
    results = {}

    for strategy in strategies:
        try:
            retriever = get_retriever(strategy, top_k)

            if strategy == "multi_query":
                docs = await asyncio.to_thread(retriever.get_relevant_documents, query)
            else:
                docs = retriever.get_relevant_documents(query)

            results[strategy] = {
                "num_docs": len(docs),
                "docs": [
                    {
                        "content": doc.page_content[:200],
                        "metadata": doc.metadata,
                    }
                    for doc in docs[:3]
                ]
            }
        except Exception as e:
            results[strategy] = {"error": str(e)}

    return {
        "query": query,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
