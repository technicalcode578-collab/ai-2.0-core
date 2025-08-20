from fastapi import FastAPI
from .database import models
# The leading dots correctly import from within the 'ai_core' package
from .api import ingestion, search_routes, personalization

# This creates the database tables on startup
models.create_db_and_tables()

app = FastAPI(
    title="Acytel Music AI",
    description="The core intelligence engine for music personalization.",
    version="2.0.0"
)

# These lines make all your APIs live with a /api/v1 prefix
app.include_router(ingestion.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(search_routes.router, prefix="/api/v1", tags=["Search"])
app.include_router(personalization.router, prefix="/api/v1", tags=["Personalization"])

@app.get("/")
def read_root():
    return {"status": "Acytel Music AI is running."}