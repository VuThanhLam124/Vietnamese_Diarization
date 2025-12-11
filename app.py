from __future__ import annotations

from pathlib import Path
from typing import List

from src.models import DiarizationEngine, Segment


def diarize_file(
    audio_path: str | Path,
    hf_token: str | None = None,
    device: str = "auto",
    show_progress: bool = True,
) -> List[Segment]:
    """API đơn giản để dùng trực tiếp trong Python."""
    engine = DiarizationEngine(token=hf_token, device=device)
    return engine.run(audio_path, show_progress=show_progress)


if __name__ == "__main__":
    # Ví dụ nhanh: python app.py audio.wav
    import argparse

    parser = argparse.ArgumentParser(description="Ví dụ chạy diarization qua hàm Python.")
    parser.add_argument("audio", help="Đường dẫn tới file âm thanh")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Thiết bị ưu tiên khi khởi tạo pipeline",
    )
    args = parser.parse_args()

    segments = diarize_file(args.audio, device=args.device)
    for idx, seg in enumerate(segments, start=1):
        print(f"{idx:02d} | {seg.start:7.2f}s -> {seg.end:7.2f}s | speaker {seg.speaker}")
