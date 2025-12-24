# Kiến Trúc Hệ Thống Vietnamese Diarization

## Tổng Quan

Hệ thống speaker diarization cho tiếng Việt, xác định "ai nói khi nào" trong audio, kèm khả năng tự động gán nhãn giới tính, vùng miền và transcription.

**Công nghệ chính:**
- pyannote/speaker-diarization-3.1 (diarization)
- PhoWhisper (ASR)
- vn-speaker-profiling (gender + dialect)
- Gradio (web interface)

## Kiến Trúc Components

### 1. Core Engine (src/)

**src/models.py - DiarizationEngine**
- Wrapper cho pyannote pipeline
- Xử lý authentication HuggingFace token
- Auto-detect CUDA/CPU
- Convert audio sang WAV 16kHz mono (ffmpeg)
- Export kết quả: RTTM, JSON

**src/utils.py - Utilities**
- Quản lý token (file/env vars/parameter)
- Audio preprocessing (ffmpeg)
- Download URL (yt-dlp)
- Merge segments liền kề cùng speaker

**src/asr.py - ASREngine**
- Transcription bằng PhoWhisper
- Hỗ trợ cả file và segment
- Model caching với lru_cache

**src/profiling.py - ProfilingEngine**
- Nhận diện giới tính (Nam/Nữ)
- Nhận diện vùng miền (Bắc/Trung/Nam)
- Trả về confidence score

### 2. User Interfaces

**app.py - Gradio Web UI**
- Upload audio hoặc paste YouTube/TikTok URL
- Hiển thị kết quả dạng bảng interactive
- Auto-label với ASR + Profiling
- Gán nhãn thủ công qua dropdown
- Export ZIP chứa segments và metadata.csv
- Merge nhiều ZIP thành dataset lớn

**cli_infer.py - Command Line**
- Chạy diarization qua terminal
- Export RTTM/JSON
- Phù hợp cho batch processing

## Data Flow

```
INPUT
  |
  |-- Audio file hoặc URL
  v
PREPROCESSING
  |
  |-- yt-dlp (nếu URL) → download
  |-- ffmpeg → WAV mono 16kHz
  v
DIARIZATION
  |
  |-- pyannote pipeline
  v
SEGMENTS
  |
  |-- List[{start, end, speaker}]
  v
AUTO LABELING (optional)
  |
  |-- ASREngine → transcription
  |-- ProfilingEngine → gender + dialect
  v
USER EDITING
  |
  |-- Gradio Dataframe (interactive)
  v
EXPORT
  |
  |-- ffmpeg extract segments
  |-- metadata.csv
  |-- ZIP archive
  v
OUTPUT
```

## Authentication Flow

**HuggingFace Token (theo thứ tự ưu tiên):**
1. Function parameter `hf_token`
2. Environment variable `HF_TOKEN`
3. Environment variable `HUGGINGFACE_TOKEN`
4. File `hugging_face_key.txt`

**Yêu cầu:** Accept terms cho 3 models trên HuggingFace:
- pyannote/speaker-diarization-3.1
- pyannote/segmentation-3.0
- pyannote/embedding

## File Format

### Metadata CSV
```csv
id,file_name,start_mmss,end_mmss,gender,region,transcription,speaker,duration_sec
id_0_1_001,id_0_1_001.wav,00:15,00:23,Nam,Trung,"Xin chào",SPEAKER_00,8.0
```

### ID Format
```
id_{gender_code}_{region_code}_{index:03d}
```
- gender_code: 0=Nam, 1=Nữ
- region_code: 0=Bắc, 1=Trung, 2=Nam
- index: 001, 002, 003...

### Label Mappings
```python
GENDER: {"nam": "0", "nữ": "1"}
REGION: {"bắc": "0", "trung": "1", "nam": "2"}
```

## Design Patterns

### Model Caching
```python
@functools.lru_cache(maxsize=2)
def _get_engine(token_key: str, device: str):
    # Tránh reload model nhiều lần
```

### Temp File Management
```python
tmpdir = Path(tempfile.mkdtemp(prefix="diarization_"))
try:
    # Processing...
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
```

### Segment Merging
- Gộp đoạn liền kề cùng speaker nếu gap ≤ 0.5s
- Loại bỏ đoạn quá ngắn < 1.0s

## Dependencies

**Core:**
- pyannote.audio==3.1.1
- torch==2.2.0
- gradio==4.19.2

**Optional:**
- transformers (ASR)
- vn-speaker-profiling (profiling)
- yt-dlp (URL download)

**External Tools (phải có trong PATH):**
- ffmpeg (bắt buộc)
- yt-dlp (optional)

## Usage

### Local Development
```bash
# Cài đặt
pip install -r requirements.txt

# Set token
export HF_TOKEN="hf_xxx..."

# Chạy Gradio
python app.py

# Chạy CLI
python cli_infer.py audio.wav --json output.json
```

### Python API
```python
from app import diarize_file

segments = diarize_file("audio.wav", device="auto")
# Returns: [Segment(start=0.0, end=3.5, speaker="SPEAKER_00"), ...]
```

### Kaggle/Colab
```python
!python kaggle_setup.py
!python app.py --share
```

## Gradio UI Workflow

1. **Upload**: File audio hoặc paste URL
2. **Diarize**: Nhấn "Chạy diarization"
3. **Auto-label**: Chọn model ASR → nhấn "Auto Label"
   - Hiển thị confidence: "Nam (85%)" hoặc "Nam (65%)" với warning
4. **Edit**: Chọn row → gán gender/region/transcription thủ công
5. **Export**: Nhấn "Tách và tải" → download ZIP
6. **Merge**: Upload nhiều ZIP → gộp thành dataset lớn

## Error Handling

**Fallback Mechanisms:**
- Không có ffmpeg → copy file as-is (warning)
- Không có yt-dlp → show error message
- Network error (HF Spaces) → prompt upload file
- Auto-label error → fill "Error" trong cell
- Low confidence → thêm warning icon

## State Management

**Gradio State Variables:**
- `segments_state`: List[dict] - dữ liệu segments từ diarization
- `audio_state`: List[str] - [prepared_path, prep_tmpdir, source_name, download_tmp]
- `selected_row`: int - row đang được chọn trong Dataframe

## Deployment

**Hugging Face Spaces:**
- Set `HF_TOKEN` trong Space secrets
- YAML frontmatter trong README.md configure Space
- Free tier không download URL được (network restriction)

**Kaggle:**
- Enable GPU trong Settings
- Add `HF_TOKEN` vào Kaggle Secrets
- Run với `--share` để có public URL

## Performance Notes

- Model loading: cache với lru_cache (tránh reload)
- CUDA auto-detect: fallback sang CPU nếu không có GPU
- Batch processing: dùng tmpdir chung cho nhiều segments
- Memory: model tiny/base/small/medium/large tùy VRAM

## Future Extensions (placeholder files)

- `eval.py`: Evaluation metrics
- `finetune.py`: Fine-tuning pipeline
- `params/*.yaml`: Configuration files
