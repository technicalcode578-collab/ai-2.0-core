from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import datetime
from typing import List

# Import our custom project modules
from ai_core.database import models, session
from ai_core.core import fingerprint_engine, recommender_engine
from ai_core.api import schemas

router = APIRouter()

@router.post("/users/{user_id}/generate-fingerprint", tags=["Personalization"])
def generate_fingerprint(
    user_id: int, 
    db: Session = Depends(session.get_db_session)
):
    """
    Calculates and saves a user's music taste fingerprint.
    """
    fingerprint_vector = fingerprint_engine.calculate_user_fingerprint(user_id, db)
    
    if fingerprint_vector is None:
        raise HTTPException(status_code=404, detail="No listening history found for user, cannot generate fingerprint.")

    db_fingerprint = db.query(models.UserFingerprint).filter(models.UserFingerprint.user_id == user_id).first()

    if db_fingerprint:
        db_fingerprint.fingerprint_vector = fingerprint_vector.tobytes()
        db_fingerprint.last_updated = datetime.datetime.utcnow()
    else:
        db_fingerprint = models.UserFingerprint(
            user_id=user_id,
            fingerprint_vector=fingerprint_vector.tobytes()
        )
        db.add(db_fingerprint)
    
    db.commit()
    
    return {"status": "success", "user_id": user_id, "message": "User fingerprint has been successfully generated/updated."}


@router.get("/users/{user_id}/recommendations", response_model=List[schemas.Song], tags=["Personalization"])
def get_user_recommendations(
    user_id: int, 
    limit: int = 10,
    db: Session = Depends(session.get_db_session)
):
    """
    Generates and returns a list of personalized song recommendations
    based on the user's calculated taste fingerprint.
    """
    recommendations = recommender_engine.get_recommendations(
        user_id=user_id, 
        db=db, 
        limit=limit
    )
    if not recommendations:
        raise HTTPException(
            status_code=404, 
            detail="Could not generate recommendations. User may need more listening history or fingerprint is not yet generated."
        )
    return recommendations
