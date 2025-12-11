from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple


def read_hf_token(token: str | None = None, key_path: str | Path = "hugging_face_key.txt") -> str:
    """Ưu tiên token truyền vào, nếu không thì đọc từ biến môi trường hoặc file."""
    candidates = [
        token,
        os.getenv("HF_TOKEN"),  # HF Spaces standard secret name
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


def seconds_to_mmss(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, sec = divmod(total_seconds, 60)
    return f"{minutes:02d}:{sec:02d}"


def format_segments_table(segments: Iterable[dict]) -> str:
    """Trả về chuỗi bảng đơn giản để in ra terminal."""
    lines = []
    for idx, seg in enumerate(segments, start=1):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        speaker = seg.get("speaker", "unknown")
        lines.append(
            f"{idx:02d} | {seconds_to_mmss(start)} -> {seconds_to_mmss(end)} | speaker {speaker}"
        )
    return "\n".join(lines)


def merge_adjacent_segments(
    segments: list[dict],
    max_gap: float = 0.5,
    min_duration: float = 1.0,
) -> list[dict]:
    """
    Ghép các đoạn liên tiếp cùng speaker nếu khoảng trống <= max_gap (giây).
    Đồng thời lọc bỏ đoạn quá ngắn (< min_duration).
    """
    if not segments:
        return []

    merged: list[dict] = []
    # đảm bảo sắp xếp theo thời gian
    segs = sorted(segments, key=lambda s: s.get("start", 0.0))
    current = segs[0].copy()

    for seg in segs[1:]:
        if (
            seg.get("speaker") == current.get("speaker")
            and seg.get("start", 0.0) - current.get("end", 0.0) <= max_gap
        ):
            current["end"] = max(current.get("end", 0.0), seg.get("end", 0.0))
        else:
            if current.get("end", 0.0) - current.get("start", 0.0) >= min_duration:
                merged.append(current)
            current = seg.copy()

    if current.get("end", 0.0) - current.get("start", 0.0) >= min_duration:
        merged.append(current)

    return merged


def convert_to_wav_16k(audio_path: Path) -> Tuple[Path, Path | None]:
    """
    Chuyển audio về WAV mono 16 kHz bằng ffmpeg.
    Trả về (đường dẫn dùng để suy luận, thư mục tạm để dọn dẹp hoặc None nếu không cần).
    """
    safe_stem = audio_path.stem.replace(" ", "_")
    tmpdir = Path(tempfile.mkdtemp(prefix="diarization_audio_"))

    if not shutil.which("ffmpeg"):
        # Sao chép tệp hiện có vào thư mục tạm với tên không dấu cách
        output = tmpdir / audio_path.name.replace(" ", "_")
        shutil.copy2(audio_path, output)
        return output, tmpdir

    output = tmpdir / f"{safe_stem}_16k.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        "-f",
        "wav",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"ffmpeg convert thất bại: {stderr}")
    if not output.exists():
        raise RuntimeError("ffmpeg không tạo được file WAV.")
    return output, tmpdir


def download_audio_from_url(url: str) -> Tuple[Path, Path]:
    """
    Tải audio từ YouTube/TikTok/... dùng yt-dlp, xuất WAV để xử lý tiếp.
    Trả về (đường dẫn file wav, thư mục tạm chứa file).
    """
    if not shutil.which("yt-dlp"):
        raise RuntimeError("Cần cài yt-dlp để tải liên kết (pip install yt-dlp).")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Cần cài ffmpeg để chuyển đổi audio.")

    tmpdir = Path(tempfile.mkdtemp(prefix="download_media_"))
    out_tmpl = tmpdir / "%(title)s.%(ext)s"
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        str(out_tmpl),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        # Check if it's a network error
        if "Failed to resolve" in stderr or "No address associated with hostname" in stderr:
            raise RuntimeError(
                "Không thể kết nối internet để tải video. "
                "Hugging Face Spaces (free tier) không hỗ trợ download từ URL. "
                "Vui lòng tải file audio trực tiếp thay vì dùng URL."
            )
        raise RuntimeError(f"Tải audio thất bại: {stderr}")

    wav_files = list(tmpdir.glob("*.wav"))
    if not wav_files:
        raise RuntimeError("Không tìm thấy file WAV sau khi tải.")
    return wav_files[0], tmpdir
