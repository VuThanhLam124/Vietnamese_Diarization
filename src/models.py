from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from .utils import ensure_audio_path, read_hf_token


@dataclass
class Segment:
    start: float
    end: float
    speaker: str


class DiarizationEngine:
    """Bao gói pipeline diarization của pyannote."""

    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-community-1",
        token: str | None = None,
        key_path: str | Path = "hugging_face_key.txt",
        device: str = "auto",
    ) -> None:
        self.device = self._resolve_device(device)
        auth_token = read_hf_token(token, key_path)
        self.pipeline = Pipeline.from_pretrained(model_id, token=auth_token)
        self.pipeline.to(self.device)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "cpu":
            return torch.device("cpu")
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Yêu cầu CUDA nhưng không phát hiện GPU khả dụng.")
            return torch.device("cuda")
        if device == "auto":
            return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        raise ValueError("Giá trị device hợp lệ: auto, cpu, cuda.")

    def diarize(self, audio_path: str | Path, show_progress: bool = True):
        audio_path = ensure_audio_path(audio_path)
        if show_progress:
            with ProgressHook() as hook:
                return self.pipeline(str(audio_path), hook=hook)
        return self.pipeline(str(audio_path))

    @staticmethod
    def to_segments(diarization) -> List[Segment]:
        segments: List[Segment] = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                Segment(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker=str(speaker),
                )
            )
        return segments

    @staticmethod
    def save_rttm(diarization, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        diarization.write_rttm(path)
        return path

    def run(self, audio_path: str | Path, show_progress: bool = True) -> List[Segment]:
        """Chạy pipeline và trả về danh sách segment."""
        diarization = self.diarize(audio_path, show_progress=show_progress)
        return self.to_segments(diarization)
