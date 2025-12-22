# Vietnamese Diarization - Kaggle Notebook

## Hướng dẫn sử dụng trên Kaggle

### Bước 1: Tạo Notebook mới
- Vào Kaggle → New Notebook
- Settings → Accelerator → GPU T4 x2 (hoặc P100)
- Settings → Internet → On

### Bước 2: Thêm HF Token vào Secrets
- Settings → Secrets → Add Secret
- Label: `HF_TOKEN`
- Value: Token từ https://huggingface.co/settings/tokens
- ⚠️ Đảm bảo đã accept terms tại:
  - https://hf.co/pyannote/speaker-diarization-3.1
  - https://hf.co/pyannote/segmentation-3.0

---

## Notebook Code

### Cell 1: Clone và Setup

```python
# Clone repo
!git clone https://github.com/VuThanhLam124/Vietnamese_Diarization.git
%cd Vietnamese_Diarization

# Install dependencies
!pip install -q pyannote.audio==3.1.1 gradio==4.19.2 yt-dlp
!pip install -q vn-speaker-profiling>=0.1.7 transformers>=4.36.0 librosa>=0.10.0

# Set HF Token từ Kaggle Secrets
from kaggle_secrets import UserSecretsClient
import os
try:
    user_secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
    print("✓ HF_TOKEN loaded from Kaggle Secrets")
except:
    print("⚠ HF_TOKEN not found in Secrets. Please add it.")
```

### Cell 2: Kiểm tra GPU

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Cell 3: Chạy Gradio App (Public URL)

```python
# Chạy với share=True để có public URL
!python app.py --share
```

⚠️ Copy public URL (dạng `https://xxxxx.gradio.live`) để truy cập từ trình duyệt.

---

## Option 2: Chạy Inference Trực Tiếp

### Upload audio file

```python
# Upload file từ local
from google.colab import files  # Kaggle cũng hỗ trợ
uploaded = files.upload()
audio_file = list(uploaded.keys())[0]
print(f"Uploaded: {audio_file}")
```

### Hoặc dùng sample audio

```python
# Download sample audio
!wget -q https://www.sample-videos.com/audio/mp3/crowd-cheering.mp3 -O sample.mp3
audio_file = "sample.mp3"
```

### Chạy Diarization

```python
!python infer.py {audio_file} --json output.json --rttm output.rttm
```

### Chạy với Auto-Label (ASR + Profiling)

```python
from app import diarize_file
from src.asr import ASREngine
from src.profiling import ProfilingEngine

# Diarization
segments = diarize_file(audio_file, device="cuda")

# ASR
asr = ASREngine(model_id="vinai/PhoWhisper-base", device="cuda")
profiler = ProfilingEngine(device="cuda")

for seg in segments:
    # Transcribe
    text = asr.transcribe_segment(audio_file, seg.start, seg.end)
    
    # Profile
    profile = profiler.profile_segment(audio_file, seg.start, seg.end)
    
    print(f"{seg.start:.1f}-{seg.end:.1f}s | {seg.speaker}")
    print(f"  Gender: {profile.gender} ({profile.gender_confidence:.0%})")
    print(f"  Dialect: {profile.dialect} ({profile.dialect_confidence:.0%})")
    print(f"  Text: {text}")
    print()
```

---

## Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `Failed to load pipeline` | Accept terms trên HuggingFace |
| `HF_TOKEN not found` | Thêm vào Kaggle Secrets |
| `CUDA out of memory` | Dùng model nhỏ hơn (PhoWhisper-tiny) |
| `No module 'gradio'` | Chạy lại cell install |
