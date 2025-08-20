from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from decouple import config

# Use python-decouple to fetch the database URL from the .env file
DATABASE_URL = config("DATABASE_URL")

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} # Required for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_session():
    """
    Dependency function to get a database session for API requests.
    Ensures the session is always closed after the request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
