import torch
from PIL import Image
import numpy as np
import warnings
from sentence_transformers import SentenceTransformer
import librosa

# Suppress warnings for a clean output
warnings.filterwarnings("ignore")

class CLAPEmbedder:
    """
    The final, stable embedder using a public CLIP model from sentence-transformers.
    It works by treating audio spectrograms as images.
    """
    def __init__(self, device="cpu"):
        print(f"Loading public CLIP model onto device '{device}'...")
        self.device = device
        # Load a standard, public, non-gated CLIP model
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
            # Get the embedding using the model's powerful image encoder
            embedding = self.model.encode(image, batch_size=1, convert_to_numpy=True, show_progress_bar=False)
            return embedding
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Encodes a text string into an embedding vector."""
        try:
            # The same SentenceTransformer model can encode text directly
            embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return embedding
        except Exception as e:
            print(f"Error encoding text '{text}': {e}")
            return None