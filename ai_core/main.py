from fastapi import FastAPI
from ai_core import embeddings, history_routes
from ai_core.history_db import init_db
from ai_core import search_routes

app = FastAPI(title="AI 2.0 Core API")

# Include routers
app.include_router(embeddings.router)
app.include_router(history_routes.router)
app.include_router(search_routes.router)

# Startup event to initialize database
@app.on_event("startup")
def startup_event():
    init_db()  # Synchronous init

@app.get("/")
async def root():
    return {"status": "AI Core is running 🚀"}
