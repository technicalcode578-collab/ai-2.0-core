import numpy as np
from sqlalchemy.orm import Session
from typing import Optional

from ai_core.database import models

def calculate_user_fingerprint(user_id: int, db: Session) -> Optional[np.ndarray]:
    """
    Calculates a user's music taste fingerprint.

    This is done by averaging the embedding vectors of all songs the user has
    fully listened to.

    Args:
        user_id: The ID of the user.
        db: The SQLAlchemy database session.

    Returns:
        A NumPy array representing the user's taste fingerprint, or None if
        the user has no listening history.
    """
    print(f"Calculating taste fingerprint for user_id: {user_id}...")

    # Step 1: Query the UserEvent table for all "positive" listening events.
    positive_events = (
        db.query(models.UserEvent.song_id)
        .filter(
            models.UserEvent.user_id == user_id,
            models.UserEvent.event_type == "SONG_PLAYED_FULL"
        )
        .distinct()
        .all()
    )

    if not positive_events:
        print("No positive listening events found for this user.")
        return None

    # Extract the song IDs from the query result
    song_ids = [event.song_id for event in positive_events]
    print(f"Found {len(song_ids)} unique songs in user's history.")

    # Step 2: Fetch the corresponding clip_embedding vectors from the Song table.
    song_embeddings_query = (
        db.query(models.Song.clip_embedding)
        .filter(models.Song.id.in_(song_ids), models.Song.clip_embedding.isnot(None))
        .all()
    )
    
    if not song_embeddings_query:
        print("Could not find embeddings for the songs in the user's history.")
        return None

    # Step 3: Convert the raw binary embeddings back into NumPy arrays.
    # We assume a 768-dimensional float32 vector from the CLIP ViT-L-14 model.
    embedding_vectors = [
        np.frombuffer(embedding[0], dtype=np.float32) 
        for embedding in song_embeddings_query
    ]

    # Step 4: Use numpy.mean to compute the average vector (the fingerprint).
    fingerprint_vector = np.mean(embedding_vectors, axis=0)
    
    print(f"Successfully calculated fingerprint vector with shape: {fingerprint_vector.shape}")
    return fingerprint_vector

