# Vietnamese Diarization Project - AI Assistant Guide

## Project Overview
This is a **Vietnamese speaker diarization system** using pyannote/speaker-diarization-3.1. It identifies "who spoke when" in audio files, providing timestamps and speaker labels. The project targets Vietnamese media (YouTube, TikTok, audio files) and supports gender/region labeling for dataset creation.

## Core Architecture

### Component Structure
- **[app.py](app.py)**: Gradio UI + Python API (`diarize_file()`) - main entry point for users
- **[infer.py](infer.py)**: CLI interface for batch processing and scripting
- **[src/models.py](src/models.py)**: `DiarizationEngine` wrapper around pyannote pipeline
- **[src/utils.py](src/utils.py)**: Audio conversion (ffmpeg), token management, segment formatting

### Data Flow
1. **Input**: Audio file OR YouTube/TikTok URL (via yt-dlp)
2. **Preprocessing**: Convert to WAV mono 16kHz using ffmpeg ([src/utils.py#L115-L148](src/utils.py#L115-L148))
3. **Diarization**: pyannote pipeline runs on preprocessed audio ([src/models.py#L88-L103](src/models.py#L88-L103))
4. **Output**: Segments list + RTTM file + JSON export + labeled WAV clips (optional)

### Key Design Decisions
- **Token caching**: `@functools.lru_cache` on `_get_engine()` prevents reloading heavy models ([app.py#L42-L45](app.py#L42-L45))
- **Temp file management**: All conversions use `tempfile.mkdtemp()` with cleanup in `finally` blocks
- **Device handling**: Auto-detect CUDA vs CPU, fallback gracefully ([src/models.py#L80-L87](src/models.py#L80-L87))
- **Vietnamese UI**: All user-facing strings in Vietnamese; error messages use Vietnamese

## Critical Dependencies

### External Tools (Required)
- **ffmpeg**: MUST be in PATH for audio conversion. Without it, fallback copies file as-is (may cause pyannote errors)
- **yt-dlp**: Optional, only for URL downloads. Fails gracefully if missing

### Authentication Flow
Token priority (first found wins):
1. Function parameter `hf_token`
2. Environment: `HF_TOKEN` (Hugging Face Spaces standard)
3. Environment: `HUGGINGFACE_TOKEN` / `HUGGINGFACE_ACCESS_TOKEN`
4. File: `hugging_face_key.txt` (gitignored)

**Critical**: Users MUST accept terms for 3 models (see [src/models.py#L42-L46](src/models.py#L42-L46))

## Development Workflows

### Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure ffmpeg available
which ffmpeg  # must return path

# Set token (one of):
export HF_TOKEN="your_token_here"
echo "your_token_here" > hugging_face_key.txt

# Launch Gradio UI
python app.py

# CLI usage
python infer.py path/to/audio.wav --device cuda --rttm output.rttm
```

### Testing Diarization
```python
from app import diarize_file
segments = diarize_file("audio.wav", device="auto")
# Returns: [Segment(start=0.0, end=3.5, speaker="SPEAKER_00"), ...]
```

### Adding New Features
- **Model parameters**: Modify `segmentation_params` / `clustering_params` in [DiarizationEngine.__init__](src/models.py#L24-L33)
- **Output formats**: Add new export functions in [src/utils.py](src/utils.py) following `export_segments_json()` pattern
- **UI components**: Gradio interface defined at bottom of [app.py](app.py#L350-L495), uses `gr.Blocks()` layout

## Project-Specific Patterns

### Segment Merging
The `merge_adjacent_segments()` function ([src/utils.py#L77-L108](src/utils.py#L77-L108)) combines consecutive segments from same speaker if gap ≤ 0.5s and duration ≥ 1.0s. This reduces false splits from pyannote's aggressive segmentation.

### Gender/Region Labeling
- **Gender codes**: `{"nam"/"male": "0", "nữ"/"nu"/"female": "1"}` ([app.py#L22](app.py#L22))
- **Region codes**: `{"bắc"/"bac"/"north": "0", "trung"/"central": "1", "nam"/"south": "2"}` ([app.py#L23](app.py#L23))
- Normalization always lowercases and strips input ([app.py#L140-L141](app.py#L140-L141))

### WAV Splitting Workflow
Gradio UI allows users to:
1. Run diarization → get segments table
2. Select row → assign gender/region/transcription
3. Click "Tách và tải" → downloads ZIP with:
   - Individual WAV clips (ffmpeg segments)
   - `metadata.csv` with format: `id,file_name,start_mmss,end_mmss,gender,region,transcription,speaker,duration_sec`

ID naming: `id_{gender_code}_{region_code}_{index:03d}` (e.g., `id_0_1_042` = male, central region, segment 42)

## Integration Points

### Hugging Face Spaces Deployment
- YAML frontmatter in [README.md](README.md#L1-L10) configures Spaces
- Set `HF_TOKEN` as Space secret (required)
- **Network limitation**: Free tier can't download URLs - `download_audio_from_url()` detects this and prompts file upload ([src/utils.py#L176-L180](src/utils.py#L176-L180))

### Pipeline Compatibility
Handles both pyannote output formats:
- Old: Returns `Annotation` object directly
- New: Returns object with `.speaker_diarization` attribute
See `_get_annotation()` ([src/models.py#L107-L113](src/models.py#L107-L113))

## Common Pitfalls

1. **Missing ffmpeg**: Causes cryptic pyannote errors - always check `shutil.which("ffmpeg")` first
2. **Unauthenticated**: If token invalid, error message is verbose but check all 3 model terms accepted
3. **Temp cleanup**: Always wrap temp operations in try/finally with `shutil.rmtree(tmpdir, ignore_errors=True)`
4. **Space-in-filenames**: Use `audio_path.stem.replace(" ", "_")` when creating temp files ([src/utils.py#L115](src/utils.py#L115))

## Configuration Files
- **params/*.yaml**: Currently empty placeholders for future eval/finetune configs
- **eval.py / finetune.py**: Empty stubs for planned features
- **data/**: User's local audio files (not version controlled)
