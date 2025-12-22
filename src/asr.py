"""ASR Engine sử dụng PhoWhisper cho tiếng Việt."""
from __future__ import annotations

import functools
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch


@dataclass
class TranscriptionResult:
    """Kết quả transcription."""
    text: str
    language: str = "vi"


@functools.lru_cache(maxsize=2)
def _get_whisper_pipeline(model_id: str, device: str):
    """Cache pipeline để tránh reload model."""
    from transformers import pipeline
    
    torch_device = device
    if device == "auto":
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load ASR pipeline với PhoWhisper
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=torch_device,
        torch_dtype=torch.float16 if torch_device == "cuda" else torch.float32,
    )
    return pipe


class ASREngine:
    """PhoWhisper-based ASR cho tiếng Việt.
    
    Supported models:
    - vinai/PhoWhisper-tiny
    - vinai/PhoWhisper-base
    - vinai/PhoWhisper-small
    - vinai/PhoWhisper-medium
    - vinai/PhoWhisper-large
    """
    
    SUPPORTED_MODELS = [
        "vinai/PhoWhisper-tiny",
        "vinai/PhoWhisper-base",
        "vinai/PhoWhisper-small",
        "vinai/PhoWhisper-medium",
        "vinai/PhoWhisper-large",
    ]
    
    def __init__(
        self,
        model_id: str = "vinai/PhoWhisper-base",
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._pipeline = None
    
    @property
    def pipeline(self):
        """Lazy load pipeline."""
        if self._pipeline is None:
            self._pipeline = _get_whisper_pipeline(self.model_id, self.device)
        return self._pipeline
    
    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """Transcribe toàn bộ file audio.
        
        Args:
            audio_path: Đường dẫn file audio
            
        Returns:
            TranscriptionResult với text và language
        """
        audio_path = str(audio_path)
        result = self.pipeline(audio_path)
        text = result.get("text", "").strip()
        return TranscriptionResult(text=text, language="vi")
    
    def transcribe_segment(
        self,
        audio_path: str | Path,
        start: float,
        end: float,
        tmpdir: Optional[Path] = None,
    ) -> str:
        """Transcribe một segment cụ thể của audio.
        
        Args:
            audio_path: Đường dẫn file audio gốc
            start: Thời điểm bắt đầu (giây)
            end: Thời điểm kết thúc (giây)
            tmpdir: Thư mục tạm (nếu None sẽ tạo mới)
            
        Returns:
            Text transcription của segment
        """
        audio_path = Path(audio_path)
        
        # Tạo thư mục tạm nếu cần
        if tmpdir is None:
            tmpdir = Path(tempfile.mkdtemp(prefix="asr_segment_"))
        
        # Extract segment bằng ffmpeg
        segment_path = tmpdir / f"seg_{start:.3f}_{end:.3f}.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-ac", "1",
            "-ar", "16000",
            "-vn", "-f", "wav",
            str(segment_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg extract segment thất bại: {result.stderr}")
        
        # Transcribe segment
        transcription = self.transcribe(segment_path)
        return transcription.text
    
    def transcribe_segments(
        self,
        audio_path: str | Path,
        segments: List[dict],
    ) -> List[str]:
        """Transcribe nhiều segments.
        
        Args:
            audio_path: Đường dẫn file audio gốc
            segments: List các dict với keys "start" và "end"
            
        Returns:
            List các text transcription
        """
        tmpdir = Path(tempfile.mkdtemp(prefix="asr_batch_"))
        results = []
        
        for seg in segments:
            try:
                text = self.transcribe_segment(
                    audio_path, 
                    seg["start"], 
                    seg["end"],
                    tmpdir=tmpdir,
                )
                results.append(text)
            except Exception as e:
                print(f"ASR error cho segment {seg}: {e}")
                results.append("")
        
        return results
