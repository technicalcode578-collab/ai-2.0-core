from fastapi import APIRouter, Query
from typing import Optional
from ai_core import history_db

router = APIRouter()

@router.get("/history")
def read_history(limit: int = Query(10, description="Number of events to return")):
    """
    Return the most recent history events.
    """
    return history_db.get_history(limit)

@router.post("/history/add")
def create_event(
    event_type: str,
    user_id: str,
    session_id: Optional[str] = None,
    payload: Optional[dict] = None
):
    """
    Add a new history event to the database.
    """
    history_db.add_event(event_type, user_id, session_id, payload)
    return {"status": "success", "message": "Event added"}

@router.delete("/history/clear")
def clear_all_history():
    """
    Clear all history events from the database.
    """
    history_db.clear_history()
    return {"status": "success", "message": "All history cleared"}
@router.get("/history/search")
def search_history_route(
    event_type: Optional[str] = None,
    user_id: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 10
):
    """
    Search history by event_type, user_id, or keyword in payload.
    """
    return history_db.search_history(event_type, user_id, keyword, limit)
