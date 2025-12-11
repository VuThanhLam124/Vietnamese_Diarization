from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Any, Dict, Optional
import shutil

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from .utils import ensure_audio_path, read_hf_token, convert_to_wav_16k


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
        segmentation_params: Optional[Dict[str, float]] = None,
        clustering_params: Optional[Dict[str, float]] = None,
    ) -> None:
        import sys
        self.device = self._resolve_device(device)
        auth_token = read_hf_token(token, key_path)
        
        # Handle model config with deprecated parameters
        try:
            pipeline = Pipeline.from_pretrained(model_id, use_auth_token=auth_token)
        except TypeError as e:
            if "plda" in str(e):
                print(f"WARNING: Model config contains deprecated 'plda' parameter. Loading with workaround...", file=sys.stderr)
                # Download and patch config
                from huggingface_hub import hf_hub_download
                import yaml
                from pyannote.audio.pipelines import SpeakerDiarization
                
                # Download original config
                config_path = hf_hub_download(repo_id=model_id, filename="config.yaml", token=auth_token)
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Get pipeline class
                pipeline_class = SpeakerDiarization
                
                # Remove deprecated params from config
                if 'params' in config:
                    deprecated_keys = ['plda']
                    for key in deprecated_keys:
                        if key in config['params']:
                            print(f"  Removing deprecated parameter: {key}", file=sys.stderr)
                            del config['params'][key]
                    
                    # Instantiate pipeline directly with cleaned params
                    pipeline = pipeline_class(**config['params'], use_auth_token=auth_token)
                else:
                    raise ValueError("Invalid config structure")
            else:
                raise
        
        params = pipeline.parameters()
        # Giảm phân mảnh: chỉ cập nhật các khóa thực sự tồn tại để tránh lỗi.
        seg_cfg = params.get("segmentation")
        if seg_cfg:
            default_seg = {"min_duration_on": 1.0, "min_duration_off": 0.8}
            for k, v in default_seg.items():
                if k in seg_cfg:
                    seg_cfg[k] = v
            if segmentation_params:
                for k, v in segmentation_params.items():
                    if k in seg_cfg:
                        seg_cfg[k] = v
        if clustering_params and "clustering" in params:
            params["clustering"].update(clustering_params)
        self.pipeline = pipeline.instantiate(params)
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

    def diarize(
        self, audio_path: str | Path, show_progress: bool = True, keep_audio: bool = False
    ):
        audio_path = ensure_audio_path(audio_path)
        prepared_path, tmpdir = convert_to_wav_16k(audio_path)
        try:
            if show_progress:
                with ProgressHook() as hook:
                    result = self.pipeline(str(prepared_path), hook=hook)
            else:
                result = self.pipeline(str(prepared_path))
            if keep_audio:
                return result, prepared_path, tmpdir
            return result
        finally:
            if tmpdir and not keep_audio:
                shutil.rmtree(tmpdir, ignore_errors=True)

    @staticmethod
    def _get_annotation(diarization: Any):
        """Hỗ trợ cả dạng trả về cũ (Annotation) và mới (có speaker_diarization)."""
        if hasattr(diarization, "itertracks"):
            return diarization
        if hasattr(diarization, "speaker_diarization"):
            return diarization.speaker_diarization
        raise TypeError("Output pipeline không có Annotation hoặc speaker_diarization.")

    def to_segments(self, diarization: Any) -> List[Segment]:
        annotation = self._get_annotation(diarization)
        segments: List[Segment] = []
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                Segment(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker=str(speaker),
                )
            )
        return segments

    def save_rttm(self, diarization: Any, output_path: str | Path) -> Path:
        annotation = self._get_annotation(diarization)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            annotation.write_rttm(f)
        return path

    def run(self, audio_path: str | Path, show_progress: bool = True) -> List[Segment]:
        """Chạy pipeline và trả về danh sách segment."""
        diarization = self.diarize(audio_path, show_progress=show_progress)
        return self.to_segments(diarization)
