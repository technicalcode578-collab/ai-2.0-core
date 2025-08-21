from fastapi import FastAPI
from .database import models
# We now import all of our API router modules
from .api import ingestion, search_routes, personalization, alchemy

# This creates the database tables on startup
models.create_db_and_tables()

app = FastAPI(
    title="Acytel Music AI",
    description="The core intelligence engine for music personalization.",
    version="2.0.0"
)

# Include all of our API routers
app.include_router(ingestion.router, prefix="/api/v1")
app.include_router(search_routes.router, prefix="/api/v1")
app.include_router(personalization.router, prefix="/api/v1")
app.include_router(alchemy.router, prefix="/api/v1") # Add the new alchemy router

@app.get("/")
def read_root():
    return {"status": "Acytel Music AI is running."}