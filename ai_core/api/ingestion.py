from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ai_core.database import models, session
from ai_core.api import schemas

router = APIRouter()

@router.post("/ingest-event", response_model=schemas.UserEvent)
def ingest_user_event(
    event: schemas.UserEventCreate, 
    db: Session = Depends(session.get_db_session)
):
    try:
        db_event = models.UserEvent(**event.dict())
        db.add(db_event)
        db.commit()
        db.refresh(db_event)
        return db_event
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
