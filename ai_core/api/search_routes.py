from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import chromadb

from ai_core.database import models, session
from ai_core.api import schemas
from ai_core.core import search_engine
from ai_core.models.clip_embedder import SimpleClipEmbedder

# --- API Setup & Initialization ---
router = APIRouter()

# Initialize our AI and DB connections once when the server starts
VECTOR_DB_PATH = "./data/vector_db"
VECTOR_DB_COLLECTION = "song_thought_vectors"
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
vector_collection = chroma_client.get_or_create_collection(name=VECTOR_DB_COLLECTION)
embedder = SimpleClipEmbedder(device="cpu") # Use CPU for API server

@router.get("/search/semantic", response_model=List[schemas.Song], tags=["Search"])
def semantic_search_endpoint(
    q: str, 
    limit: int = 5,
    db: Session = Depends(session.get_db_session)
):
    """
    Performs semantic search for songs based on a natural language query.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty.")
    
    results = search_engine.semantic_search(
        query_text=q,
        db=db,
        vector_collection=vector_collection,
        embedder=embedder,
        limit=limit
    )
    
    if not results:
        raise HTTPException(status_code=404, detail="No matching songs found for your query.")
        
    return results
