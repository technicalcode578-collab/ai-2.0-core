# ai_core/utils/audio_ingest.py
import os, json
from pathlib import Path

AUDIO_DIR = Path(__file__).parent.parent / "data" / "audio"
INDEX_FILE = Path(__file__).parent.parent / "data" / "audio_index.json"

def ingest_audio():
    files = [f for f in AUDIO_DIR.glob("*.wav")] + [f for f in AUDIO_DIR.glob("*.mp3")]
    index = []
    for f in files:
        index.append({
            "file_name": f.name,
            "path": str(f),
            "metadata": {
                "title": f.stem,
                "format": f.suffix.replace('.', ''),
                "size_bytes": f.stat().st_size
            }
        })
    with open(INDEX_FILE, "w") as out:
        json.dump(index, out, indent=2)
    print(f"Ingested {len(index)} audio files into {INDEX_FILE}")

if __name__ == "__main__":
    ingest_audio()
