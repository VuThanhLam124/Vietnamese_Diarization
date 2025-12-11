from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List


def read_hf_token(token: str | None = None, key_path: str | Path = "hugging_face_key.txt") -> str:
    """Ưu tiên token truyền vào, nếu không thì đọc từ biến môi trường hoặc file."""
    candidates = [
        token,
        os.getenv("HUGGINGFACE_TOKEN"),
        os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
    ]
    for value in candidates:
        if value and value.strip():
            return value.strip()
    path = Path(key_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy token. Truyền biến --hf-token hoặc đặt file {path}."
        )
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"File {path} trống, hãy dán Hugging Face access token vào.")
    return content


def ensure_audio_path(audio_path: str | Path) -> Path:
    """Kiểm tra đường dẫn audio hợp lệ."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file âm thanh: {path}")
    if not path.is_file():
        raise ValueError(f"Đường dẫn không phải file: {path}")
    return path


def export_segments_json(segments: Iterable[dict], output_path: str | Path) -> Path:
    """Lưu danh sách segment thành JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: List[dict] = list(segments)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def format_segments_table(segments: Iterable[dict]) -> str:
    """Trả về chuỗi bảng đơn giản để in ra terminal."""
    lines = []
    for idx, seg in enumerate(segments, start=1):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        speaker = seg.get("speaker", "unknown")
        lines.append(f"{idx:02d} | {start:7.2f}s -> {end:7.2f}s | speaker {speaker}")
    return "\n".join(lines)
