import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from pathlib import Path
from typing import Dict

def deconstruct_song(
    input_filepath: str, 
    output_directory: str
) -> Dict[str, str]:
    """
    Separates a song into its core stems (vocals, drums, bass, other) using Demucs.

    Args:
        input_filepath: The path to the source audio file.
        output_directory: The directory where the separated stem files will be saved.

    Returns:
        A dictionary mapping stem names to their output file paths.
    """
    print(f"Starting deconstruction for: {input_filepath}")
    
    # Ensure the output directory exists
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Load the pre-trained Demucs model
    model = get_model(name='htdemucs')
    
    # Load the audio file
    wav, sr = torchaudio.load(input_filepath)
    
    # The model expects a specific sample rate
    resampler = torchaudio.transforms.Resample(sr, model.samplerate)
    wav = resampler(wav)
    
    # Separate the audio into stems
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    sources = apply_model(model, wav[None], device="cpu")[0] # Run on CPU for compatibility
    sources = sources * ref.std() + ref.mean()

    # Define the names for the output stems
    stem_names = ['drums', 'bass', 'other', 'vocals']
    output_paths = {}

    # Save each stem to a new file
    for i, name in enumerate(stem_names):
        stem_path = Path(output_directory) / f"{name}.wav"
        torchaudio.save(str(stem_path), sources[i].cpu(), model.samplerate)
        output_paths[name] = str(stem_path)
        print(f"  - Saved stem: {stem_path}")
        
    print("Deconstruction complete.")
    return output_paths

