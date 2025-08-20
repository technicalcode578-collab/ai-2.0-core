import datetime
import datetime
from typing import Optional, List
from pydantic import BaseModel

class UserEventCreate(BaseModel):
    user_id: int
    song_id: int
    event_type: str

class UserEvent(UserEventCreate):
    id: int
    timestamp: datetime.datetime

    class Config:
        from_attributes = True

# Pydantic model for returning Song data from the API
class Song(BaseModel):
    id: int
    title: str
    artist: str
    bpm: Optional[float] = None
    key: Optional[str] = None

    class Config:
        from_attributes = True
