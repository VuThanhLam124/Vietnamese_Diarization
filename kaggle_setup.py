"""
Kaggle Setup Script for Vietnamese Diarization

Usage on Kaggle:
1. Enable GPU in Settings
2. Clone repo: !git clone https://github.com/YOUR_REPO/Vietnamese_Diarization.git
3. Run this script: !python Vietnamese_Diarization/kaggle_setup.py
4. Then run app: !python Vietnamese_Diarization/app.py
"""

import subprocess
import sys
import os

def run_cmd(cmd, check=True):
    """Run shell command."""
    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if check and result.returncode != 0:
        print(f"Warning: Command failed with code {result.returncode}")
    return result.returncode == 0

def main():
    print("=" * 60)
    print("Vietnamese Diarization - Kaggle Setup")
    print("=" * 60)
    
    # Check GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ No GPU detected, using CPU")
    
    # Install dependencies
    print("\n[1/4] Installing dependencies...")
    deps = [
        "pyannote.audio==3.1.1",
        "gradio==4.19.2",
        "yt-dlp",
        "vn-speaker-profiling>=0.1.7",
        "transformers>=4.36.0",
        "librosa>=0.10.0",
    ]
    run_cmd(f"{sys.executable} -m pip install -q " + " ".join(deps))
    
    # Accept HuggingFace model terms reminder
    print("\n[2/4] HuggingFace Setup")
    print("⚠ IMPORTANT: You need to accept terms for these models:")
    print("  1. https://hf.co/pyannote/speaker-diarization-3.1")
    print("  2. https://hf.co/pyannote/segmentation-3.0")
    print("  3. https://hf.co/pyannote/embedding")
    print("\nSet HF_TOKEN in Kaggle Secrets or create hugging_face_key.txt")
    
    # Check if HF_TOKEN is set
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print("✓ HF_TOKEN detected in environment")
    else:
        print("⚠ HF_TOKEN not found. Set it in Kaggle Secrets > Add Secret")
    
    # Pre-download models (optional, for faster first run)
    print("\n[3/4] Pre-loading models...")
    
    # Test ASR import
    try:
        from transformers import pipeline
        print("✓ Transformers ready")
    except Exception as e:
        print(f"⚠ Transformers import error: {e}")
    
    # Test profiling import
    try:
        from infer import SpeakerProfiler
        print("✓ SpeakerProfiler ready")
    except Exception as e:
        print(f"⚠ SpeakerProfiler import error: {e}")
    
    print("\n[4/4] Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  Option 1 - Run Gradio locally (with share=True for public URL):")
    print("    !python app.py --share")
    print("")
    print("  Option 2 - Run inference directly:")
    print("    !python infer.py your_audio.wav --json output.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
