"""Speaker Profiling Engine wrapper cho vn-speaker-profiling."""
from __future__ import annotations

import functools
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class ProfileResult:
    """Kết quả profiling speaker."""
    gender: str                # "Nam" / "Nữ"
    gender_code: int           # 0 = Male, 1 = Female
    gender_confidence: float   # 0.0 - 1.0
    dialect: str               # "Bắc" / "Trung" / "Nam"
    dialect_code: int          # 0 = North, 1 = Central, 2 = South
    dialect_confidence: float  # 0.0 - 1.0
    
    def is_gender_confident(self, threshold: float = 0.8) -> bool:
        return self.gender_confidence >= threshold
    
    def is_dialect_confident(self, threshold: float = 0.8) -> bool:
        return self.dialect_confidence >= threshold


# Label mappings
GENDER_LABELS = {0: "Nam", 1: "Nữ"}
DIALECT_LABELS = {0: "Bắc", 1: "Trung", 2: "Nam"}


def _import_speaker_profiler():
    """Import SpeakerProfiler từ vn-speaker-profiling package bằng absolute path."""
    import importlib.util
    import site
    import sys
    import os
    
    # Tìm infer.py trong site-packages
    for sp in site.getsitepackages():
        infer_path = Path(sp) / "infer.py"
        if infer_path.exists():
            # Lưu trạng thái
            old_path = sys.path.copy()
            old_modules = dict(sys.modules)
            old_cwd = os.getcwd()
            
            try:
                # Xóa tất cả path có thể conflict với local src/
                cwd = os.getcwd()
                sys.path = [p for p in sys.path 
                           if not (p == '' or p == cwd or 'Vietnamese_Diarization' in p)]
                sys.path.insert(0, sp)
                
                # Xóa cached modules có thể conflict
                for mod_name in list(sys.modules.keys()):
                    if mod_name.startswith('src.') or mod_name == 'src':
                        del sys.modules[mod_name]
                
                # Chuyển cwd để tránh relative import conflict
                os.chdir(sp)
                
                # Load module
                spec = importlib.util.spec_from_file_location("vn_profiling_infer", infer_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.SpeakerProfiler
            finally:
                os.chdir(old_cwd)
                sys.path = old_path
                # Restore only essential modules, không restore src.* để tránh conflict
    
    raise ImportError("Không tìm thấy vn-speaker-profiling package. Cài đặt: pip install vn-speaker-profiling")


@functools.lru_cache(maxsize=2)
def _get_profiler(checkpoint: str, encoder: str, device: str):
    """Cache profiler để tránh reload model."""
    SpeakerProfiler = _import_speaker_profiler()
    
    torch_device = device
    if device == "auto":
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = {
        "model": {
            "checkpoint": checkpoint,
            "name": encoder,
            "head_hidden_dim": 256,
        },
        "audio": {
            "sampling_rate": 16000,
            "max_duration": 30 if "whisper" in encoder.lower() else 5,
        },
        "inference": {
            "batch_size": 1,
            "device": torch_device,
        },
        "input": {
            "audio_path": None,
            "audio_dir": None,
        },
        "output": {
            "dir": "output/predictions",
            "save_results": False,
            "format": "json",
        },
        "labels": {
            "gender": {0: "Male", 1: "Female"},
            "dialect": {0: "North", 1: "Central", 2: "South"},
        },
    }
    
    return SpeakerProfiler(config)


class ProfilingEngine:
    """Vietnamese Speaker Profiling sử dụng vn-speaker-profiling.
    
    Nhận diện giới tính (gender) và vùng miền (dialect) từ audio.
    """
    
    def __init__(
        self,
        checkpoint: str = "hf:Thanh-Lam/profiling-gender-dialect-pho",
        encoder: str = "vinai/PhoWhisper-base",
        device: str = "auto",
    ) -> None:
        self.checkpoint = checkpoint
        self.encoder = encoder
        self.device = device
        self._profiler = None
    
    @property
    def profiler(self):
        """Lazy load profiler."""
        if self._profiler is None:
            self._profiler = _get_profiler(self.checkpoint, self.encoder, self.device)
        return self._profiler
    
    def _parse_result(self, raw_result: dict) -> ProfileResult:
        """Parse kết quả từ SpeakerProfiler.predict()."""
        gender_info = raw_result.get("gender", {})
        dialect_info = raw_result.get("dialect", {})
        
        gender_code = gender_info.get("code", 0)
        dialect_code = dialect_info.get("code", 0)
        
        return ProfileResult(
            gender=GENDER_LABELS.get(gender_code, "Nam"),
            gender_code=gender_code,
            gender_confidence=gender_info.get("confidence", 0.0),
            dialect=DIALECT_LABELS.get(dialect_code, "Bắc"),
            dialect_code=dialect_code,
            dialect_confidence=dialect_info.get("confidence", 0.0),
        )
    
    def profile(self, audio_path: str | Path) -> ProfileResult:
        """Profile một file audio.
        
        Args:
            audio_path: Đường dẫn file audio (WAV 16kHz recommended)
            
        Returns:
            ProfileResult với gender, dialect và confidence scores
        """
        audio_path = str(audio_path)
        raw_result = self.profiler.predict(audio_path)
        
        if "error" in raw_result:
            raise RuntimeError(f"Profiling error: {raw_result['error']}")
        
        return self._parse_result(raw_result)
    
    def profile_segment(
        self,
        audio_path: str | Path,
        start: float,
        end: float,
        tmpdir: Optional[Path] = None,
    ) -> ProfileResult:
        """Profile một segment cụ thể của audio.
        
        Args:
            audio_path: Đường dẫn file audio gốc
            start: Thời điểm bắt đầu (giây)
            end: Thời điểm kết thúc (giây)
            tmpdir: Thư mục tạm (nếu None sẽ tạo mới)
            
        Returns:
            ProfileResult
        """
        audio_path = Path(audio_path)
        
        # Tạo thư mục tạm nếu cần
        if tmpdir is None:
            tmpdir = Path(tempfile.mkdtemp(prefix="profile_segment_"))
        
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
        
        return self.profile(segment_path)
    
    def profile_segments(
        self,
        audio_path: str | Path,
        segments: list[dict],
    ) -> list[ProfileResult]:
        """Profile nhiều segments.
        
        Args:
            audio_path: Đường dẫn file audio gốc
            segments: List các dict với keys "start" và "end"
            
        Returns:
            List các ProfileResult
        """
        tmpdir = Path(tempfile.mkdtemp(prefix="profile_batch_"))
        results = []
        
        for seg in segments:
            try:
                profile = self.profile_segment(
                    audio_path,
                    seg["start"],
                    seg["end"],
                    tmpdir=tmpdir,
                )
                results.append(profile)
            except Exception as e:
                print(f"Profiling error cho segment {seg}: {e}")
                # Return default profile với confidence 0
                results.append(ProfileResult(
                    gender="Nam",
                    gender_code=0,
                    gender_confidence=0.0,
                    dialect="Bắc",
                    dialect_code=0,
                    dialect_confidence=0.0,
                ))
        
        return results
