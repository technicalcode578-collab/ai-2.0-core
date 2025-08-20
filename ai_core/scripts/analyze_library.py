import os
import sys
import json
from pathlib import Path
import warnings
import re

# --- Environment Setup ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
warnings.filterwarnings("ignore")

from ai_core.database.session import SessionLocal
from ai_core.database import models
from ai_core.models.clip_embedder import SimpleClipEmbedder
from ai_core.core import lyric_fetcher
from transformers import T5ForConditionalGeneration, T5Tokenizer
import librosa
from tqdm import tqdm
import torch

# --- LLM Brain Definition ---
class LLMCoPilot:
    def __init__(self, device="cuda"):
        print(f"Loading Google FLAN-T5 model onto device '{device}'...")
        self.device = device
        model_name = "google/flan-t5-base"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print("LLM Co-Pilot loaded successfully.")

    def summarize_lyrics(self, lyrics: str, title: str, artist: str) -> str:
        try:
            prompt = f"""Analyze the following song lyrics and provide a brief, one-sentence summary of their theme and mood.

Title: "{title}"
Artist: {artist}
Lyrics:
{lyrics[:1500]}

One-sentence summary of the theme and mood:"""

            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"Error generating lyric summary: {e}")
            return None

# --- Main Analysis Pipeline ---
def analyze_and_enrich_library():
    print("--- 🚀 Starting Full Library Analysis (Audio + Lyrics) ---")
    db = SessionLocal()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    clip_embedder = SimpleClipEmbedder(device=device)
    llm_copilot = LLMCoPilot(device=device)

    try:
        # --- Load Data and Match Files ---
        METADATA_PATH = project_root / "data" / "metadata.json"
        AUDIO_DIR = project_root / "data" / "audio"
        with open(METADATA_PATH, 'r') as f:
            metadata_list = json.load(f)
        audio_filenames = {f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')}
        
        print("\n--- Matching metadata to audio files... ---")
        tasks = []
        def clean_text(text): return re.sub(r'[^a-z0-9]', '', text.lower())

        for metadata in metadata_list:
            title_clean = clean_text(metadata.get('title', ''))
            artist_clean = clean_text(re.split(r',|&|ft\.', metadata.get('artist', ''))[0].strip())
            
            best_match = next((fname for fname in audio_filenames if artist_clean in clean_text(os.path.splitext(fname)[0]) and title_clean in clean_text(os.path.splitext(fname)[0])), None)
            
            if best_match:
                tasks.append({"filepath": str(AUDIO_DIR / best_match), "metadata": metadata})
            else:
                print(f"WARNING: Could not find a matching audio file for '{metadata.get('title')}' by '{metadata.get('artist')}'.")
        print(f"Successfully matched {len(tasks)} songs for analysis.")

        # --- Main Analysis Loop ---
        for task in tqdm(tasks, desc="Analyzing Library"):
            filepath_str = task["filepath"]
            song_meta = task["metadata"]
            
            song = db.query(models.Song).filter(models.Song.filepath == filepath_str).first()
            if not song:
                song = models.Song(filepath=filepath_str, title=song_meta.get('title'), artist=song_meta.get('artist'))
                db.add(song)
                db.commit()
                db.refresh(song)
            
            if not song.bpm or not song.clip_embedding:
                y, sr = librosa.load(filepath_str)
                song.bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
                embedding = clip_embedder.get_audio_embedding_from_file(filepath_str)
                song.clip_embedding = embedding.tobytes() if embedding is not None else None
            
            if not song.lyrics:
                lyrics_text = lyric_fetcher.get_lyrics(artist=song.artist, title=song.title)
                if lyrics_text:
                    song.lyrics = lyrics_text
                    song.lyric_summary = llm_copilot.summarize_lyrics(lyrics=lyrics_text, title=song.title, artist=song.artist)
            
            db.commit()

        song_count = db.query(models.Song).count()
        print(f"\n✅ Success. Library analysis complete. The 'songs' table now contains {song_count} entries.")

    finally:
        db.close()

if __name__ == "__main__":
    analyze_and_enrich_library()