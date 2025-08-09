# ai_core/utils/clap_embedder.py
import torch
import torchaudio
import laion_clap
import numpy as np

class CLAPEmbedder:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CLAPEmbedder] device={self.device}")
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()  # downloads if needed
        self.model = self.model.to(self.device)

    def _load_audio(self, path, target_sr=48000):
        audio, sr = torchaudio.load(path)  # returns Tensor [channels, samples]
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
        # convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        return audio

    def embed_audio_path(self, path):
        audio = self._load_audio(path)  # Tensor [1, N]
        # model expects batch dimension
        # laion_clap API: get_audio_embedding_from_data(tensor, use_tensor=True)
        emb = self.model.get_audio_embedding_from_data(audio.to(self.device), use_tensor=True)
        emb_np = emb.detach().cpu().numpy().squeeze()
        # ensure 1D numpy vector
        return emb_np

    def embed_texts(self, texts: list[str]):
        emb = self.model.get_text_embedding(texts, use_tensor=True)
        return emb.detach().cpu().numpy()
