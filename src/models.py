from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Any, Dict, Optional
import shutil

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from .utils import ensure_audio_path, read_hf_token, convert_to_wav_16k

torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

@dataclass
class Segment:
    start: float
    end: float
    speaker: str


class DiarizationEngine:
    """Bao gói pipeline diarization của pyannote.
    Pipeline pyannote/speaker-diarization-3.1 có các hyperparameters sau:
    
    1. SEGMENTATION PARAMETERS (segmentation_params):
       - min_duration_off (float): Thời gian tối thiểu (giây) của khoảng im lặng 
         giữa các speech segments.
         + Giá trị nhỏ hơn → nhạy hơn với khoảng dừng ngắn
         + Giá trị lớn hơn → bỏ qua các khoảng dừng ngắn
         + Mặc định: ~0.5s
    
    2. CLUSTERING PARAMETERS (clustering_params):
       - threshold (float): Ngưỡng khoảng cách để quyết định 2 embeddings 
         có thuộc cùng speaker hay không.
         + Giá trị nhỏ hơn → nhiều speaker hơn (dễ tách nhầm)
         + Giá trị lớn hơn → ít speaker hơn (dễ gộp nhầm)
         + Mặc định: ~0.7
       
       - method (str): Phương pháp clustering
         + "centroid", "average", "ward", "complete", "single"
         + Mặc định: "centroid"
       
       - min_cluster_size (int): Số segment tối thiểu để tạo thành 1 speaker cluster
         + Giá trị lớn hơn → loại bỏ speaker xuất hiện ít
         + Mặc định: 15

    """

    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-3.1",
        token: str | None = None,
        key_path: str | Path = "hugging_face_key.txt",
        device: str = "auto",
        segmentation_params: Optional[Dict[str, float]] = None,
        clustering_params: Optional[Dict[str, float]] = None,
    ) -> None:
        import sys
        self.device = self._resolve_device(device)
        auth_token = read_hf_token(token, key_path)
        
        # Load pipeline with authentication
        print(f"DEBUG: Loading model {model_id} with token={'***' if auth_token else 'None'}", file=sys.stderr)
        pipeline = Pipeline.from_pretrained(model_id, token=auth_token)
        
        if pipeline is None:
            raise RuntimeError(
                f"Failed to load pipeline '{model_id}'. "
                f"IMPORTANT: You need to accept terms for ALL these models:\n"
                f"  1. https://hf.co/pyannote/speaker-diarization-3.1\n"
                f"  2. https://hf.co/pyannote/segmentation-3.0\n"
                f"  3. https://hf.co/pyannote/embedding\n"
                f"After accepting, add HF_TOKEN to Space secrets with your token."
            )
        
        print(f"DEBUG: Pipeline loaded successfully", file=sys.stderr)
        
        # Apply hyperparameters if provided
        hyperparams = {}
        if segmentation_params:
            hyperparams["segmentation"] = segmentation_params
        if clustering_params:
            hyperparams["clustering"] = clustering_params
        
        if hyperparams:
            pipeline.instantiate(hyperparams)
            print(f"DEBUG: Applied hyperparameters: {hyperparams}", file=sys.stderr)
        
        # Store and move to device
        self.pipeline = pipeline
        self.pipeline.to(self.device)
        print(f"DEBUG: Pipeline moved to device: {self.device}", file=sys.stderr)

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

    def run(self, audio_path: str | Path, show_progress: bool = True, merge_gap: float = 2.0) -> List[Segment]:
        """Chạy pipeline và trả về danh sách segment.
        
        Args:
            audio_path: Đường dẫn file audio
            show_progress: Hiển thị progress bar
            merge_gap: Khoảng trống tối đa để merge segments cùng speaker (giây)
        """
        from .utils import merge_adjacent_segments
        
        diarization = self.diarize(audio_path, show_progress=show_progress)
        segments = self.to_segments(diarization)
        
        # Merge segments cùng speaker nếu gần nhau
        if merge_gap > 0:
            dict_segments = [{'start': s.start, 'end': s.end, 'speaker': s.speaker} for s in segments]
            merged = merge_adjacent_segments(dict_segments, max_gap=merge_gap, min_duration=0.5)
            segments = [Segment(start=s['start'], end=s['end'], speaker=s['speaker']) for s in merged]
        
        return segments
