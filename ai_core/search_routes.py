# ai_core/search_routes.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import List
from pathlib import Path
import numpy as np
import uvicorn

from ai_core.utils.clap_embedder import CLAPEmbedder

# Import your chroma collection (reuse)
from ai_core import embeddings as embeddings_mod
collection = getattr(embeddings_mod, "collection", None)
if collection is None:
    raise RuntimeError("Chroma collection not found in ai_core.embeddings")

router = APIRouter()

embedder = CLAPEmbedder()

@router.post("/search/audio")
async def search_by_audio(file: UploadFile = File(...), n_results: int = Query(5, ge=1, le=50)):
    # save uploaded file temporarily
    tmp_dir = Path("/tmp/ai_core_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / file.filename
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        qvec = embedder.embed_audio_path(str(tmp_path)).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding error: {e}")

    res = collection.query(query_embeddings=[qvec], n_results=n_results)
    return {"query_file": file.filename, "results": res}

@router.get("/search/text")
def search_by_text(q: str = Query(...), n_results: int = Query(5, ge=1, le=50)):
    # use CLAP text embedding
    tvec = embedder.embed_texts([q])[0].tolist()
    res = collection.query(query_embeddings=[tvec], n_results=n_results)
    return {"query": q, "results": res}
