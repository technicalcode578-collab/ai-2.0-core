import os
import sys
import json
from pathlib import Path
import warnings
import re
from tqdm import tqdm
import torch
import numpy as np

# --- Environment Setup ---
# This ensures the script can find our other project modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
warnings.filterwarnings("ignore")

# --- Import All Project Components ---
from ai_core.database.session import SessionLocal
from ai_core.database import models
from ai_core.models.clip_embedder import SimpleClipEmbedder
from ai_core.core import lyric_fetcher, metadata_enricher
import librosa
import chromadb

# --- Configuration ---
AUDIO_DIR = project_root / "data" / "audio"
METADATA_PATH = project_root / "data" / "metadata.json"
VECTOR_DB_PATH = project_root / "data" / "vector_db"
VECTOR_DB_COLLECTION = "song_thought_vectors"

def run_genesis_engine():
    """
    The master script for Phase 1. Ingests, enriches, analyzes, and
    creates the fused 'thought vector' for every song in the library.
    """
    print("--- 🚀 Launching The Genesis Engine ---")
    
    db = SessionLocal()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize our AI models and databases
    clip_embedder = SimpleClipEmbedder(device=device)
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    vector_collection = chroma_client.get_or_create_collection(name=VECTOR_DB_COLLECTION)

    try:
        # Load the user's original metadata file
        with open(METADATA_PATH, 'r') as f:
            metadata_list = json.load(f)
        
        audio_filenames = {f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')}
        print(f"Found {len(metadata_list)} metadata entries and {len(audio_filenames)} audio files.")

        # --- Main Processing Loop ---
        for metadata in tqdm(metadata_list, desc="Processing Library"):
            title = metadata.get('title')
            artist = metadata.get('artist')
            if not title or not artist:
                continue

            # --- The Scribe Process ---
            # 1. Find the matching audio file
            best_match = next((fname for fname in audio_filenames if artist.lower() in fname.lower() and title.lower() in fname.lower()), None)
            if not best_match:
                print(f"Skipping '{title}': No matching audio file found.")
                continue
            
            filepath_str = str(AUDIO_DIR / best_match)

            # 2. Check if song already exists in our factual DB
            song = db.query(models.Song).filter(models.Song.filepath == filepath_str).first()
            if not song:
                # 3. If not, enrich it with external data
                print(f"\nNew song found: '{title}'. Enriching metadata...")
                enriched_info = metadata_enricher.enrich_metadata(artist=artist, title=title)
                lyrics_text = lyric_fetcher.get_lyrics(artist=artist, title=title)

                # 4. Create the new Song record for SQLite
                song = models.Song(
                    filepath=filepath_str,
                    title=title,
                    artist=artist,
                    lyrics=lyrics_text,
                    # We can add more enriched fields here later (e.g., year)
                )
                db.add(song)
                db.commit()
                db.refresh(song)
            
            # --- The Synapse Process ---
            # 1. Check if the vector embedding already exists
            existing_vector = vector_collection.get(ids=[str(song.id)])
            if existing_vector['ids']:
                continue # Skip if already processed

            # 2. If not, generate the audio embedding
            print(f"Generating audio embedding for '{title}'...")
            audio_embedding = clip_embedder.get_audio_embedding_from_file(filepath_str)

            if audio_embedding is not None:
                # For now, the audio embedding is our "Thought Vector"
                # In the future, we will fuse this with lyric and art vectors
                thought_vector = audio_embedding

                # 3. Store the "Thought Vector" in ChromaDB, linked by the song's SQL ID
                vector_collection.add(
                    ids=[str(song.id)],
                    embeddings=[thought_vector.tolist()]
                )
        
        song_count = db.query(models.Song).count()
        vector_count = vector_collection.count()
        print(f"\n--- ✅ Genesis Engine Complete ---")
        print(f"Factual Database (SQLite): {song_count} songs.")
        print(f"Vector Database (ChromaDB): {vector_count} thought vectors.")

    finally:
        db.close()

if __name__ == "__main__":
    run_genesis_engine()
