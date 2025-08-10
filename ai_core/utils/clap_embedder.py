import torch
import numpy as np
from transformers import ClapModel, ClapProcessor
import librosa
import warnings

# Suppress warnings for a cleaner log
warnings.filterwarnings("ignore")

class CLAPEmbedder:
    """
    A modern wrapper class for the Microsoft CLAP model using the Transformers library.
    """
    def __init__(self, model_name="microsoft/clap-htsat-unfused", device="cpu"):
        """
        Initializes the CLAP model from Hugging Face.
        Note: We default to 'cpu' as the Codespace may not have a GPU.
        The Docker container will run this on the CPU.
        """
        print(f"Loading Microsoft CLAP model '{model_name}' onto device '{device}'...")
        self.device = device
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        print("Microsoft CLAP model loaded successfully.")

    def get_audio_embedding_from_file(self, file_path: str) -> np.ndarray:
        """
        Computes the audio embedding for a single audio file.
        """
        try:
            # Load and resample audio to the required 48000Hz
            audio_array, sr = librosa.load(file_path, sr=48000, mono=True)
            
            # Process the audio array
            inputs = self.processor(audios=audio_array, sampling_rate=48000, return_tensors="pt").to(self.device)
            
            # Get the embedding
            with torch.no_grad():
                embedding = self.model.get_audio_features(**inputs)
            
            # Return as a CPU numpy array
            return embedding.cpu().numpy()[0]
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None