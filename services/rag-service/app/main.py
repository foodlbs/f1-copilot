from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncIterator
import os
import redis
import json
import asyncio
import hashlib
from datetime import datetime

# LangChain imports
from langchain_community.llms import Ollama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import Document
from pinecone import Pinecone
from ddgs import DDGS

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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "ollama"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

# ==================== LangChain Setup ====================
def get_llm(temperature: float = 0.7, streaming: bool = True):
    """Get LLM based on configured provider."""
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=temperature,
            streaming=streaming
        )
    else:
        return Ollama(
            base_url=OLLAMA_HOST,
            model=OLLAMA_MODEL,
            temperature=temperature
        )

llm = get_llm(temperature=0.7, streaming=True)
llm_no_stream = get_llm(temperature=0.3, streaming=False)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY
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
    """Callback handler for streaming tokens from LLM."""

    def __init__(self):
        self.tokens = []
        self.finish = False

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:  # Only append non-empty tokens
            self.tokens.append(token)

    async def on_llm_end(self, response, **kwargs) -> None:
        self.finish = True

    async def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        """Handle chat model start - required for ChatOpenAI compatibility."""
        pass

    async def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        """Handle LLM start."""
        pass

    async def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        """Handle chain start."""
        pass

    async def on_chain_end(self, outputs, **kwargs) -> None:
        """Handle chain end."""
        pass

    async def on_retriever_start(self, serialized, query, **kwargs) -> None:
        """Handle retriever start."""
        pass

    async def on_retriever_end(self, documents, **kwargs) -> None:
        """Handle retriever end."""
        pass

# ==================== Models ====================
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    retrieval_strategy: str = "similarity"
    top_k: int = 5
    stream: bool = False
    enable_web_fallback: bool = True  # Enable web search if vector DB lacks info

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str
    strategy_used: str
    web_search_used: bool = False
    cached_for_future: bool = False

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

# ==================== Web Search Fallback ====================
def check_insufficient_info(answer: str) -> bool:
    """Check if the RAG response indicates insufficient information."""
    insufficient_phrases = [
        "i don't have enough information",
        "i don't have sufficient information",
        "no information available",
        "cannot find relevant",
        "not in the context",
        "no relevant data",
        "unable to find",
        "i couldn't find",
        "no data available"
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in insufficient_phrases)

def perform_web_search(query: str, max_results: int = 5) -> List[dict]:
    """Perform web search using DuckDuckGo."""
    try:
        ddgs = DDGS()
        results = list(ddgs.text(
            f"F1 Formula 1 {query}",  # Add F1 context to search
            max_results=max_results
        ))
        print(f"Web search returned {len(results)} results for: {query}")
        return results
    except Exception as e:
        print(f"Web search error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return []

def generate_doc_id(content: str) -> str:
    """Generate a unique document ID from content."""
    return hashlib.md5(content.encode()).hexdigest()

async def ingest_search_results(query: str, search_results: List[dict]) -> List[Document]:
    """Ingest web search results into Pinecone for future retrieval."""
    documents = []
    vectors_to_upsert = []

    for result in search_results:
        content = f"{result.get('title', '')}\n\n{result.get('body', '')}"
        doc_id = generate_doc_id(content)

        # Check if already exists in Redis cache (avoid duplicate ingestion)
        cache_key = f"web_doc:{doc_id}"
        if redis_client.exists(cache_key):
            continue

        metadata = {
            "source": "web_search",
            "url": result.get("href", ""),
            "title": result.get("title", ""),
            "query": query,
            "ingested_at": datetime.now().isoformat(),
            "text": content  # Required for PineconeVectorStore
        }

        # Create embedding
        embedding = embeddings.embed_query(content)

        vectors_to_upsert.append({
            "id": doc_id,
            "values": embedding,
            "metadata": metadata
        })

        documents.append(Document(page_content=content, metadata=metadata))

        # Mark as ingested in Redis (TTL: 7 days)
        redis_client.setex(cache_key, 604800, "1")

    # Batch upsert to Pinecone
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
        print(f"Ingested {len(vectors_to_upsert)} web search results into Pinecone")

    return documents

async def answer_with_web_context(query: str, web_docs: List[Document], memory) -> str:
    """Generate answer using web search results as context."""
    context = "\n\n".join([doc.page_content for doc in web_docs])

    web_prompt = PromptTemplate(
        template="""You are a helpful F1 racing assistant. Use the following web search results to answer the question.

Web Search Results:
{context}

Chat History:
{chat_history}

Question: {question}

Instructions:
- Answer based on the web search results provided
- Mention this information comes from web search
- Be concise and accurate
- Focus on F1/Formula 1 related information

Answer:""",
        input_variables=["context", "chat_history", "question"]
    )

    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    chat_history_str = "\n".join([f"{m.type}: {m.content}" for m in chat_history]) if chat_history else "None"

    prompt_text = web_prompt.format(
        context=context,
        chat_history=chat_history_str,
        question=query
    )

    response = llm_no_stream.invoke(prompt_text)

    # Handle both string and AIMessage responses
    if hasattr(response, 'content'):
        return response.content
    return str(response)

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
        "features": ["streaming", "multi_query", "mmr", "compression", "web_search_fallback"],
        "llm_provider": LLM_PROVIDER,
        "llm_model": OPENAI_MODEL if LLM_PROVIDER == "openai" else OLLAMA_MODEL,
        "pinecone_index": PINECONE_INDEX,
        "web_search_enabled": WEB_SEARCH_ENABLED
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
                stream_response(qa_chain, request.query, session_id, request.retrieval_strategy, request.enable_web_fallback),
                media_type="text/event-stream"
            )
        else:
            result = qa_chain({"question": request.query})
            answer = result["answer"]
            web_search_used = False
            cached_for_future = False
            sources = []

            # Check if we need web search fallback
            if (WEB_SEARCH_ENABLED and
                request.enable_web_fallback and
                check_insufficient_info(answer)):

                print(f"Insufficient info detected, performing web search for: {request.query}")
                search_results = perform_web_search(request.query, WEB_SEARCH_MAX_RESULTS)

                if search_results:
                    # Ingest results into Pinecone for future queries
                    web_docs = await ingest_search_results(request.query, search_results)
                    cached_for_future = len(web_docs) > 0

                    # Generate new answer with web context
                    answer = await answer_with_web_context(request.query, web_docs, memory)
                    web_search_used = True

                    sources = [
                        {
                            "content": doc.page_content[:500],
                            "metadata": doc.metadata,
                        }
                        for doc in web_docs
                    ]
            else:
                sources = [
                    {
                        "content": doc.page_content[:500],
                        "metadata": doc.metadata,
                    }
                    for doc in result.get("source_documents", [])
                ]

            save_to_history(session_id, "user", request.query)
            save_to_history(session_id, "assistant", answer)

            return ChatResponse(
                answer=answer,
                sources=sources,
                session_id=session_id,
                strategy_used=request.retrieval_strategy,
                web_search_used=web_search_used,
                cached_for_future=cached_for_future
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_response(qa_chain, query: str, session_id: str, strategy: str, enable_web_fallback: bool = True) -> AsyncIterator[str]:
    try:
        callback = StreamingCallbackHandler()

        result = await asyncio.to_thread(
            qa_chain,
            {"question": query},
            callbacks=[callback]
        )

        full_answer = ""
        for token in callback.tokens:
            if token:  # Skip empty tokens
                full_answer += token
                yield f"data: {json.dumps({'token': token, 'type': 'token'})}\n\n"
                await asyncio.sleep(0.01)

        web_search_used = False
        cached_for_future = False
        sources = []

        # Check if we need web search fallback
        if (WEB_SEARCH_ENABLED and
            enable_web_fallback and
            check_insufficient_info(full_answer)):

            yield f"data: {json.dumps({'type': 'web_search_start', 'message': 'Searching the web for more information...'})}\n\n"

            search_results = perform_web_search(query, WEB_SEARCH_MAX_RESULTS)

            if search_results:
                memory = get_memory(session_id)
                web_docs = await ingest_search_results(query, search_results)
                cached_for_future = len(web_docs) > 0

                # Generate new answer with web context
                web_answer = await answer_with_web_context(query, web_docs, memory)
                web_search_used = True

                # Stream the web-based answer
                yield f"data: {json.dumps({'type': 'web_answer_start'})}\n\n"
                for char in web_answer:
                    yield f"data: {json.dumps({'token': char, 'type': 'token'})}\n\n"
                    await asyncio.sleep(0.005)

                full_answer = web_answer
                sources = [
                    {
                        "content": doc.page_content[:500],
                        "metadata": doc.metadata,
                    }
                    for doc in web_docs
                ]
        else:
            sources = [
                {
                    "content": doc.page_content[:500],
                    "metadata": doc.metadata,
                }
                for doc in result.get("source_documents", [])
            ]

        save_to_history(session_id, "user", query)
        save_to_history(session_id, "assistant", full_answer)

        yield f"data: {json.dumps({'sources': sources, 'type': 'sources', 'strategy': strategy, 'web_search_used': web_search_used, 'cached_for_future': cached_for_future})}\n\n"
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
