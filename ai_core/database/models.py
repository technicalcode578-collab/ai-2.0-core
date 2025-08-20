import datetime
from sqlalchemy import (
    Column, 
    Integer, 
    String, 
    Float, 
    DateTime, 
    LargeBinary, 
    ForeignKey,
    Text
)
from sqlalchemy.orm import declarative_base
from .session import engine

# Create a base class for our models to inherit from
Base = declarative_base()

class Song(Base):
    __tablename__ = "songs"
    
    id = Column(Integer, primary_key=True, index=True)
    filepath = Column(String, unique=True, nullable=False)
    title = Column(String)
    artist = Column(String)
    clip_embedding = Column(LargeBinary)
    bpm = Column(Float)
    key = Column(String)
    lyrics = Column(Text, nullable=True)
    lyric_summary = Column(String, nullable=True)

class UserEvent(Base):
    __tablename__ = "user_events"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    song_id = Column(Integer, ForeignKey("songs.id"), nullable=False)
    event_type = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class UserFingerprint(Base):
    __tablename__ = "user_fingerprints"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, unique=True, nullable=False)
    fingerprint_vector = Column(LargeBinary)
    last_updated = Column(DateTime, default=datetime.datetime.utcnow)

def create_db_and_tables():
    """
    Binds to the engine and creates all defined tables in the database.
    This function should be called once when the application starts.
    """
    print("Initializing database: creating tables if they do not exist...")
    Base.metadata.create_all(bind=engine)
    print("Database tables are ready.")
