from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pathlib import Path

# Import our custom project modules
from ai_core.database import models, session
from ai_core.core import deconstruction_engine

router = APIRouter()

@router.post("/alchemy/deconstruct/{song_id}", tags=["Alchemy Engine"])
def deconstruct_song_endpoint(
    song_id: int, 
    db: Session = Depends(session.get_db_session)
):
    """
    Deconstructs a song into its component stems (vocals, drums, etc.)
    and returns the file paths to the new audio files.
    """
    # Step 1: Find the song in our database
    song = db.query(models.Song).filter(models.Song.id == song_id).first()
    if not song:
        raise HTTPException(status_code=404, detail=f"Song with ID {song_id} not found.")

    # --- THIS IS THE FIX ---
    # The database gives us the Gitpod path, e.g., /workspace/ai-2-0-core/data/audio/song.mp3
    # We need to translate it to the path inside the container, e.g., /app/data/audio/song.mp3
    gitpod_base_path = "/workspace/ai-2-0-core"
    container_base_path = "/app"
    
    if song.filepath.startswith(gitpod_base_path):
        container_filepath = song.filepath.replace(gitpod_base_path, container_base_path, 1)
    else:
        container_filepath = song.filepath # Assume it's already a relative or correct path
    # -------------------------

    # Step 2: Define the output directory for the stems
    output_dir = Path("./data/stems") / str(song_id)
    
    # Step 3: Call our deconstruction engine with the CORRECT path
    try:
        stem_paths = deconstruction_engine.deconstruct_song(
            input_filepath=container_filepath, # Use the translated path
            output_directory=str(output_dir)
        )
        return {
            "status": "success",
            "song_id": song_id,
            "title": song.title,
            "stems": stem_paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to deconstruct song: {str(e)}")