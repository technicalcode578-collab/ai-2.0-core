from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional
import uuid
from . import history_db

router = APIRouter(prefix="/history", tags=["history"])

class HistoryEvent(BaseModel):
    query: Optional[str] = None
    results_count: Optional[int] = None

@router.post("/add")
async def add_history_event(
    event_type: str,
    user_id: str,
    event: HistoryEvent,
    session_id: Optional[str] = None
):
    sid = history_db.add_event(event_type, user_id, event.dict(), session_id)
    return {"status": "success", "session_id": sid}

@router.get("/")
async def get_history(limit: int = Query(10), session_id: Optional[str] = None):
    return history_db.get_history(limit=limit, session_id=session_id)
