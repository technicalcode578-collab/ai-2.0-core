import numpy as np
from sqlalchemy.orm import Session
from typing import List
import chromadb

from ai_core.database import models
from ai_core.models.clip_embedder import SimpleClipEmbedder

def semantic_search(
    query_text: str, 
    db: Session, 
    vector_collection: chromadb.Collection,
    embedder: SimpleClipEmbedder,
    limit: int = 5
) -> List[models.Song]:
    """
    Performs semantic search on the music library based on a text query.

    Args:
        query_text: The user's natural language search query.
        db: The SQLAlchemy database session.
        vector_collection: The ChromaDB collection of song vectors.
        embedder: The AI model embedder instance.
        limit: The number of results to return.

    Returns:
        A list of the most relevant Song objects.
    """
    print(f"Performing semantic search for: '{query_text}'...")

    # Step 1: Convert the user's text query into an AI embedding vector.
    # Note: The embedder needs a new get_text_embedding method.
    query_vector = embedder.get_text_embedding(query_text)
    if query_vector is None:
        print("Could not generate a vector for the query text.")
        return []

    # Step 2: Query the vector database to find the most similar song vectors.
    results = vector_collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=limit,
    )

    if not results or not results['ids'][0]:
        print("No similar songs found in the vector database.")
        return []

    # The vector DB returns the IDs of the songs in our SQL database.
    recommended_song_ids = [int(song_id) for song_id in results['ids'][0]]
    print(f"Found top {len(recommended_song_ids)} matching song IDs: {recommended_song_ids}")

    # Step 3: Fetch the full song details from our SQL database for the top matches.
    recommendations = db.query(models.Song).filter(models.Song.id.in_(recommended_song_ids)).all()
    
    # Sort the final list to match the recommendation order from the vector search
    recommendations.sort(key=lambda song: recommended_song_ids.index(song.id))
    
    return recommendations
