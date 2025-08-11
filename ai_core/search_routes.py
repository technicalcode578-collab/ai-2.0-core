from fastapi import APIRouter, HTTPException
import chromadb
import torch
import os

# --- This is the key fix ---
# We now import our custom class directly and create an instance of it.
from ai_core.utils.clap_embedder import CLAPEmbedder

# --- Configuration ---
VECTOR_DB_PATH = "/app/data/vector_db"
CHROMA_COLLECTION = "acytel_music_v2_clip"

# --- Initialize Router, Database, and Embedder ---
router = APIRouter()
embedder = CLAPEmbedder(device="cpu") # Initialize our custom embedder on the CPU

try:
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION)
    print("✅ Search API: Successfully connected to ChromaDB collection.")
except Exception as e:
    print(f"❌ Search API: Failed to connect to ChromaDB. Will not be available. Error: {e}")
    collection = None

# --- API Endpoint Definition ---
@router.post("/search/text")
def search_by_text_description(query_text: str, top_k: int = 5):
    """
    Finds the most similar tracks to a given text description.
    """
    if collection is None:
        raise HTTPException(status_code=503, detail="Database connection is not available.")

    try:
        # 1. Convert the text query into an AI embedding using our class method
        query_vector = embedder.get_text_embedding(query_text)
        
        # 2. Query the database
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        # 3. Format and return the results
        similar_tracks = []
        if results and results['ids'] and results['ids'][0]:
            for i, (item_id, distance, metadata) in enumerate(zip(results['ids'][0], results['distances'][0], results['metadatas'][0])):
                similar_tracks.append({
                    "rank": i + 1,
                    "id": item_id,
                    "distance": f"{distance:.4f}",
                    "metadata": metadata
                })
        
        return {
            "query_text": query_text,
            "results_found": len(similar_tracks),
            "similar_tracks": similar_tracks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")