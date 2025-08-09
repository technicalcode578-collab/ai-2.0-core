import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import APIRouter
from pydantic import BaseModel

# ------------------------
# Setup
# ------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "vector_db"
DB_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_or_create_collection("ai_core_vectors")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------
# Data models
# ------------------------
class AddTextsRequest(BaseModel):
    texts: list[str]
    ids: list[str] | None = None
    metadatas: list[dict] | None = None

class QueryTextsRequest(BaseModel):
    queries: list[str]
    n_results: int = 5

# ------------------------
# Core functions
# ------------------------
def add_texts(texts, ids=None, metadatas=None):
    embeddings = embedder.encode(texts).tolist()
    if ids is None:
        ids = [f"id_{i}" for i in range(len(texts))]
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

def query_texts(query_texts, n_results=5):
    query_embeddings = embedder.encode(query_texts).tolist()
    return collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results
    )

# ------------------------
# FastAPI router
# ------------------------
router = APIRouter(prefix="/embeddings", tags=["Embeddings"])

@router.post("/add")
async def add_endpoint(req: AddTextsRequest):
    add_texts(req.texts, req.ids, req.metadatas)
    return {"status": "success", "count": len(req.texts)}

@router.post("/query")
async def query_endpoint(req: QueryTextsRequest):
    results = query_texts(req.queries, req.n_results)
    return {"status": "success", "results": results}

@router.get("/test")
async def test_embeddings():
    return {"status": "ok", "message": "Embeddings endpoint working"}
