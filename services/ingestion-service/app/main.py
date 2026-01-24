from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone

app = FastAPI(title="Ingestion Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Configuration ====================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# ==================== LangChain Setup ====================
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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# ==================== Models ====================
class DocumentInput(BaseModel):
    content: str
    metadata: Optional[dict] = {}

class IngestRequest(BaseModel):
    documents: List[DocumentInput]

class IngestResponse(BaseModel):
    status: str
    documents_added: int
    chunks_created: int

# ==================== Endpoints ====================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ingestion-service",
        "pinecone_index": PINECONE_INDEX
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    try:
        documents = [
            Document(
                page_content=doc.content,
                metadata=doc.metadata
            )
            for doc in request.documents
        ]

        splits = text_splitter.split_documents(documents)
        vectorstore.add_documents(splits)

        return IngestResponse(
            status="success",
            documents_added=len(documents),
            chunks_created=len(splits)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["filename"] = file.filename

        splits = text_splitter.split_documents(documents)
        vectorstore.add_documents(splits)

        os.unlink(tmp_file_path)

        return IngestResponse(
            status="success",
            documents_added=len(documents),
            chunks_created=len(splits)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/text-file")
async def ingest_text_file(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        loader = TextLoader(tmp_file_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["filename"] = file.filename

        splits = text_splitter.split_documents(documents)
        vectorstore.add_documents(splits)

        os.unlink(tmp_file_path)

        return IngestResponse(
            status="success",
            documents_added=len(documents),
            chunks_created=len(splits)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    try:
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness,
            "namespaces": stats.namespaces
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
