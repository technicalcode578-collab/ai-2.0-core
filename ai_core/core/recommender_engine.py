import numpy as np
from sqlalchemy.orm import Session
from typing import List
from ai_core.database import models
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(user_id: int, db: Session, limit: int = 10) -> List[models.Song]:
    print(f"Generating {limit} recommendations for user_id: {user_id}...")
    user_fingerprint_obj = db.query(models.UserFingerprint).filter(models.UserFingerprint.user_id == user_id).first()
    if not user_fingerprint_obj or not user_fingerprint_obj.fingerprint_vector:
        return []

    user_fingerprint = np.frombuffer(user_fingerprint_obj.fingerprint_vector, dtype=np.float32).reshape(1, -1)
    
    listened_song_ids = [
        event.song_id for event in 
        db.query(models.UserEvent.song_id).filter(models.UserEvent.user_id == user_id).distinct().all()
    ]

    candidate_songs = (
        db.query(models.Song.id, models.Song.clip_embedding)
        .filter(models.Song.id.notin_(listened_song_ids), models.Song.clip_embedding.isnot(None))
        .all()
    )

    if not candidate_songs:
        return []
        
    candidate_ids = [song.id for song in candidate_songs]
    candidate_embeddings = np.array([np.frombuffer(song.clip_embedding, dtype=np.float32) for song in candidate_songs])
    similarity_scores = cosine_similarity(user_fingerprint, candidate_embeddings)[0]
    top_indices = np.argsort(similarity_scores)[::-1][:limit]
    recommended_song_ids = [candidate_ids[i] for i in top_indices]
    
    recommendations = db.query(models.Song).filter(models.Song.id.in_(recommended_song_ids)).all()
    recommendations.sort(key=lambda song: recommended_song_ids.index(song.id))
    return recommendations
