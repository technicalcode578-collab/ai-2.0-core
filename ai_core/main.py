from fastapi import FastAPI
from .database import models
from .api import ingestion, search_routes # Add search_routes

models.create_db_and_tables()

app = FastAPI(
    title="Acytel Music AI - Phase 2",
    description="The core intelligence engine for music personalization.",
    version="2.0.0"
)

app.include_router(ingestion.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(search_routes.router, prefix="/api/v1", tags=["Search"]) # Include the new router

@app.get("/")
def read_root():
    return {"status": "Acytel Music AI is running."}
