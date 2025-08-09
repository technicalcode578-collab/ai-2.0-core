import torch
import torchaudio
import laion_clap

class CLAPEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CLAP] Using device: {self.device}")
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()  # Downloads pretrained weights if not cached
        self.model = self.model.to(self.device)

    def embed_audio(self, file_path):
        """Extract an audio embedding."""
        audio, sr = torchaudio.load(file_path)
        if sr != 48000:
            audio = torchaudio.functional.resample(audio, sr, 48000)
        audio = audio.mean(dim=0, keepdim=True)  # mono
        emb = self.model.get_audio_embedding_from_data(audio, use_tensor=True)
        return emb.detach().cpu().numpy()

    def embed_text(self, text_list):
        """Extract embeddings for a list of text strings."""
        emb = self.model.get_text_embedding(text_list, use_tensor=True)
        return emb.detach().cpu().numpy()

if __name__ == "__main__":
    embedder = CLAPEmbedder()
    audio_emb = embedder.embed_audio("sample.wav")
    text_emb = embedder.embed_text(["lofi chill beats", "happy upbeat pop"])
    print("Audio embedding shape:", audio_emb.shape)
    print("Text embedding shape:", text_emb.shape)
