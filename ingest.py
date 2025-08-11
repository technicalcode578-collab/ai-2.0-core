import os
import json
import sys
import torch
from tqdm import tqdm
import chromadb
import re
import warnings
import numpy as np
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore")

# --- We will install and import libraries inside the main function ---

class RobustClipEmbedder:
    def __init__(self, device="cuda"):
        # We will initialize the model after installation
        self.device = device
        self.model = None

    def load_model(self):
        """Loads the model after dependencies are installed."""
        from sentence_transformers import SentenceTransformer
        print(f"Loading public CLIP model onto device '{self.device}'...")
        self.model = SentenceTransformer('clip-ViT-L-14', device=self.device)
        print("CLIP model loaded successfully.")

    def get_audio_embedding_from_file(self, file_path: str) -> np.ndarray:
        """
        Computes an embedding using a robust pydub + librosa pipeline.
        """
        if self.model is None:
            print("Model is not loaded.")
            return None
            
        from pydub import AudioSegment
        import librosa

        try:
            # Step A: Load the audio file robustly using pydub
            audio_segment = AudioSegment.from_file(file_path)
            
            # Step B: Convert to a standardized format (mono, 22050Hz)
            audio_segment = audio_segment.set_channels(1).set_frame_rate(22050)
            
            # Step C: Get raw audio data as a normalized numpy array
            samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            samples /= (2**15) # Normalize from 16-bit integer to float
            
            # Step D: Now use librosa on the clean data to create the spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=samples, sr=audio_segment.frame_rate)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            normalized_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.min()) / (log_mel_spectrogram.max() - log_mel_spectrogram.min())
            spectrogram_rgb = np.stack([normalized_spectrogram]*3, axis=-1)
            image = Image.fromarray((spectrogram_rgb * 255).astype(np.uint8))
            
            # Step E: Get the embedding
            embedding = self.model.encode(image, batch_size=1, convert_to_numpy=True, show_progress_bar=False)
            return embedding
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None

def run_pipeline():
    """Main function to orchestrate the entire ingestion pipeline."""
    # --- Installation ---
    print("--- Installing environment with robust audio library... ---")
    os.system('pip install -q sentence-transformers chromadb "librosa>=0.9.2" tqdm pydub')
    os.system('apt-get update -qq && apt-get install -y -qq ffmpeg')
    
    # --- Execution ---
    print("\n--- Starting Ingestion Pipeline ---")
    try:
        project_root = '/app'
        os.chdir(project_root)

        VECTOR_DB_PATH = os.path.join(project_root, "data/vector_db")
        METADATA_PATH = os.path.join(project_root, "data/metadata.json")
        AUDIO_DIR = os.path.join(project_root, "data/audio")
        CHROMA_COLLECTION = "acytel_music_v2_clip"

        with open(METADATA_PATH, 'r') as f:
            metadata_list = json.load(f)
        audio_filenames = os.listdir(AUDIO_DIR)
        
        print("\n--- Matching metadata to audio files... ---")
        tasks = []
        def clean_text(text): return re.sub(r'[^a-z0-9]', '', text.lower())
        for metadata in metadata_list:
            title_clean, artist_clean = clean_text(metadata.get('title','')), clean_text(re.split(r',|&|ft\.', metadata.get('artist',''))[0].strip())
            best_match = next((fname for fname in audio_filenames if artist_clean in clean_text(os.path.splitext(fname)[0]) and title_clean in clean_text(os.path.splitext(fname)[0])), None)
            if best_match: tasks.append({"path": os.path.join(AUDIO_DIR, best_match), "metadata": metadata})
            else: print(f"WARNING: Could not find a matching audio file for '{metadata.get('title')}' by '{metadata.get('artist')}'.")
        print(f"Successfully matched {len(tasks)} songs.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nUsing device: {device}")
        
        embedder = RobustClipEmbedder(device=device)
        embedder.load_model() # Load the model now that libraries are installed
        
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
        print(f"Vector DB Collection '{CHROMA_COLLECTION}' is ready.")

        ids_to_upsert, embeddings_to_upsert, metadatas_to_upsert = [], [], []
        for i, task in enumerate(tqdm(tasks, desc="Embedding your songs")):
            embedding = embedder.get_audio_embedding_from_file(task["path"])
            if embedding is not None:
                clean_metadata = {k: ", ".join(map(str,v)) if isinstance(v, list) else v for k, v in task["metadata"].items()}
                ids_to_upsert.append(f"track_{i:04d}")
                embeddings_to_upsert.append(embedding.tolist())
                metadatas_to_upsert.append(clean_metadata)

        if ids_to_upsert:
            print(f"\nUpserting {len(ids_to_upsert)} new embeddings into ChromaDB...")
            collection.upsert(ids=ids_to_upsert, embeddings=embeddings_to_upsert, metadatas=metadatas_to_upsert)
            print("Upsert complete.")

        print(f"\n\n--- ✅ INGESTION COMPLETE. Database built. Contains {collection.count()} songs. ✅ ---")

    except Exception as e:
        print(f"\n--- An error occurred during the pipeline ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")

if __name__ == "__main__":
    run_pipeline()