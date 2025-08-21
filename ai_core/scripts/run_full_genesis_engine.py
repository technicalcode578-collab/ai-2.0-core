import os
import json
import sys
import torch
from PIL import Image
import numpy as np
import warnings
import re
import subprocess

def execute_command(command):
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"ERROR executing command: {command}")
        print(process.stderr)
        raise RuntimeError("A critical command failed. Aborting.")
    print(process.stdout)

print("--- 🚀 STEP 1: Installing a simple, self-contained environment... ---")
execute_command('pip install -q sentence-transformers chromadb "librosa>=0.9.2" tqdm transformers sentencepiece accelerate')
execute_command('apt-get update -qq && apt-get install -y -qq ffmpeg')

from sentence_transformers import SentenceTransformer
import librosa
from tqdm import tqdm
import chromadb
from transformers import T5ForConditionalGeneration, T5Tokenizer

warnings.filterwarnings("ignore")

class SimpleClipEmbedder:
    def __init__(self, device="cuda"):
        print(f"Loading public CLIP model onto device '{device}'...")
        self.device = device
        self.model = SentenceTransformer('clip-ViT-L-14', device=self.device)
        print("CLIP model loaded successfully.")

    def get_audio_embedding_from_file(self, file_path: str) -> np.ndarray:
        try:
            y, sr = librosa.load(file_path, sr=22050, mono=True)
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            normalized_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.min()) / (log_mel_spectrogram.max() - log_mel_spectrogram.min())
            spectrogram_rgb = np.stack([normalized_spectrogram]*3, axis=-1)
            image = Image.fromarray((spectrogram_rgb * 255).astype(np.uint8))
            embedding = self.model.encode(image, batch_size=1, convert_to_numpy=True, show_progress_bar=False)
            return embedding
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None

class LLMCoPilot:
    def __init__(self, device="cuda"):
        model_name = "google/flan-t5-base"
        print(f"Loading FLAN-T5 model ('{model_name}') onto device '{device}'...")
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print("LLM Co-Pilot loaded successfully.")

    def summarize_lyrics(self, lyrics: str, title: str, artist: str) -> str:
        try:
            prompt = f"summarize: {lyrics[:2000]}"
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"Error generating lyric summary: {e}")
            return None

def run_pipeline():
    print("\n--- 🚀 Starting Full Library Analysis (Audio + Lyrics) ---")
    project_root = Path('/kaggle/working/ai-2-0-core') if os.path.exists('/kaggle/working') else Path('.')
    db = SessionLocal()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    clip_embedder = SimpleClipEmbedder(device=device)
    llm_copilot = LLMCoPilot(device=device)

    try:
        METADATA_PATH = project_root / "data" / "metadata.json"
        AUDIO_DIR = project_root / "data" / "audio"
        with open(METADATA_PATH, 'r') as f:
            metadata_list = json.load(f)
        audio_filenames = {f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')}
        
        tasks = []
        def clean_text(text): return re.sub(r'[^a-z0-9]', '', text.lower())
        for metadata in metadata_list:
            title_clean = clean_text(metadata.get('title', ''))
            artist_clean = clean_text(re.split(r',|&|ft\.', metadata.get('artist', ''))[0].strip())
            best_match = next((fname for fname in audio_filenames if artist_clean in clean_text(os.path.splitext(fname)[0]) and title_clean in clean_text(os.path.splitext(fname)[0])), None)
            if best_match:
                tasks.append({"filepath": str(AUDIO_DIR / best_match), "metadata": metadata})
        
        for task in tqdm(tasks, desc="Analyzing Library"):
            filepath_str, song_meta = task["filepath"], task["metadata"]
            song = db.query(models.Song).filter(models.Song.filepath == filepath_str).first()
            if not song:
                song = models.Song(filepath=filepath_str, title=song_meta.get('title'), artist=song_meta.get('artist'))
                db.add(song); db.commit(); db.refresh(song)
            
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

        print(f"\n✅ Success. The 'songs' table now contains {db.query(models.Song).count()} entries.")
    finally:
        db.close()

if __name__ == "__main__":
    # This script is designed to be run in a fresh, powerful environment like Kaggle.
    # It assumes the repository has been cloned into the root working directory.
    # We must also import the database models from our project structure.
    from pathlib import Path
    sys.path.append(str(Path('.').resolve()))
    from ai_core.database import models, session
    from ai_core.core import lyric_fetcher
    SessionLocal = session.SessionLocal
    run_pipeline()
