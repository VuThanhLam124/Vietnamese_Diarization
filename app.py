from __future__ import annotations

import functools
import tempfile
from pathlib import Path
from typing import List, Any
import shutil
import csv
import subprocess
import zipfile

import gradio as gr

from src.models import DiarizationEngine, Segment
from src.utils import (
    export_segments_json,
    format_segments_table,
    seconds_to_mmss,
    download_audio_from_url,
)

DEFAULT_TOKEN_SENTINEL = "__FROM_FILE_OR_ENV__"
GENDER_MAP = {"nam": "0", "male": "0", "nữ": "1", "nu": "1", "female": "1"}
REGION_MAP = {"bắc": "0", "bac": "0", "north": "0", "trung": "1", "central": "1", "nam": "2", "south": "2"}
ALLOWED_GENDER = {"nam", "nữ", "nu", "male", "female"}
ALLOWED_REGION = {"bắc", "trung", "nam", "bac", "north", "central", "south"}


def diarize_file(
    audio_path: str | Path,
    hf_token: str | None = None,
    device: str = "auto",
    show_progress: bool = True,
) -> List[Segment]:
    """API đơn giản để dùng trực tiếp trong Python."""
    engine = DiarizationEngine(token=hf_token, device=device)
    return engine.run(audio_path, show_progress=show_progress)


def _token_key(raw_token: str | None) -> str:
    cleaned = raw_token.strip() if raw_token else None
    return cleaned if cleaned else DEFAULT_TOKEN_SENTINEL


@functools.lru_cache(maxsize=2)
def _get_engine(token_key: str, device: str) -> DiarizationEngine:
    token_value = None if token_key == DEFAULT_TOKEN_SENTINEL else token_key
    return DiarizationEngine(token=token_value, device=device)


def _diarize_action(
    audio_path: str | None,
    hf_token: str | None,
    device: str,
    url: str | None = None,
):
    if not audio_path and not url:
        empty_state = ["", "", "", ""]
        return "Vui lòng tải file âm thanh hoặc nhập URL.", None, None, [], [], empty_state, ""
    try:
        downloaded_path = None
        download_tmp = None
        audio_input = audio_path
        if url:
            downloaded_path, download_tmp = download_audio_from_url(url)
            audio_input = str(downloaded_path)

        engine = _get_engine(_token_key(hf_token), device)
        diarization, prepared_path, prep_tmpdir = engine.diarize(
            audio_input, show_progress=False, keep_audio=True
        )
        segments = engine.to_segments(diarization)
        dict_segments = [
            {"start": float(seg.start), "end": float(seg.end), "speaker": seg.speaker}
            for seg in segments
        ]
        table = format_segments_table(dict_segments)

        output_tmp = Path(tempfile.mkdtemp(prefix="diarization_out_"))
        rttm_path = engine.save_rttm(diarization, output_tmp / "output.rttm")
        json_path = export_segments_json(dict_segments, output_tmp / "segments.json")

        df_rows = [
            [
                seconds_to_mmss(seg["start"]),
                seconds_to_mmss(seg["end"]),
                seg["speaker"],
                "",
                "",
                "",
            ]
            for seg in dict_segments
        ]

        source_name = Path(audio_input).stem if audio_input else "unknown"
        audio_state = [
            str(prepared_path),
            str(prep_tmpdir) if prep_tmpdir else "",
            source_name,
            str(download_tmp) if download_tmp else "",
        ]
        return (
            table,
            str(rttm_path),
            str(json_path),
            df_rows,
            dict_segments,
            audio_state,
            str(prepared_path),
        )
    except Exception as exc:  # pragma: no cover - hiển thị lỗi cho người dùng giao diện
        empty_state = ["", "", "", ""]
        return f"Lỗi: {exc}", None, None, [], [], empty_state, ""


def _normalize_label(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _table_to_rows(table_data: Any) -> list[list[Any]]:
    """Chuyển giá trị DataFrame/ndarray/list sang list of list để thao tác."""
    if table_data is None:
        return []
    if hasattr(table_data, "values"):  # pandas DataFrame hoặc ndarray
        try:
            return table_data.values.tolist()
        except Exception:
            pass
    if isinstance(table_data, list):
        return table_data
    if isinstance(table_data, tuple):
        return list(table_data)
    return []


def _select_row_action(evt: gr.SelectData):
    row_idx = evt.index[0] if evt and evt.index else -1
    if row_idx is None or row_idx < 0:
        return "Chưa chọn hàng", -1
    return f"Đang chọn hàng {row_idx + 1}", row_idx


def _apply_dropdown_action(
    table_rows: list[list[Any]] | None,
    selected_idx: int,
    gender_choice: str,
    region_choice: str,
    transcription_text: str,
):
    rows = _table_to_rows(table_rows)
    if selected_idx is None or selected_idx < 0 or selected_idx >= len(rows):
        return rows, "Chọn một hàng trước."

    gender_val = _normalize_label(gender_choice)
    region_val = _normalize_label(region_choice)
    if gender_val and gender_val not in ALLOWED_GENDER:
        return rows, "Giới tính chỉ được chọn nam/nữ."
    if region_val and region_val not in ALLOWED_REGION:
        return rows, "Vùng miền chỉ được chọn bắc/trung/nam."

    new_rows = [list(r) for r in rows]
    # row order: start_mmss, end_mmss, speaker, gender, region, transcription
    if len(new_rows[selected_idx]) < 6:
        new_rows[selected_idx] = (new_rows[selected_idx] + [""] * 6)[:6]
    new_rows[selected_idx][3] = gender_val
    new_rows[selected_idx][4] = region_val
    new_rows[selected_idx][5] = transcription_text
    return new_rows, f"Đã áp dụng cho hàng {selected_idx + 1}."


def _import_archives_action(files: list[Any] | None, output_root: str = "outputs"):
    if not files:
        return "Chọn ít nhất một file ZIP.", None
    merged_root = Path(tempfile.mkdtemp(prefix="merged_zip_"))
    merged_data = merged_root / "merged"
    merged_data.mkdir(parents=True, exist_ok=True)
    meta_all = merged_data / "metadata_all.csv"
    appended = 0
    extracted = 0

    with meta_all.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "id",
                "file_name",
                "start_mmss",
                "end_mmss",
                "gender",
                "region",
                "transcription",
                "speaker",
                "duration_sec",
                "source",
            ]
        )

    for f in files:
        zip_path = Path(getattr(f, "name", f))
        if not zip_path.exists() and isinstance(f, dict) and "name" in f:
            zip_path = Path(f["name"])
        if not zip_path.exists():
            continue
        extracted += 1
        dest_dir = merged_data / zip_path.stem
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)

        meta_csv = dest_dir / "metadata.csv"
        if meta_csv.exists():
            with meta_all.open("a", newline="", encoding="utf-8") as out_csv:
                writer = csv.writer(out_csv)
                with meta_csv.open("r", encoding="utf-8") as src:
                    next(src, None)  # skip header
                    for line in src:
                        row = line.strip().split(",")
                        if row and any(row):
                            writer.writerow(row + [zip_path.stem])
                            appended += 1

    # Tạo file zip từ thư mục merged_data
    zip_base_name = merged_root / "merged_output"
    merged_zip = shutil.make_archive(str(zip_base_name), "zip", merged_data)
    status = f"Đã gộp {extracted} ZIP, metadata_all.csv có thêm {appended} dòng. Tải merged.zip."
    return status, merged_zip


def _split_segments_action(
    table_rows: list[list[Any]] | None,
    segments_state: list[dict],
    audio_state: list[str],
):
    if not shutil.which("ffmpeg"):
        return "Cần cài ffmpeg để tách đoạn.", None
    if not segments_state:
        return "Chạy diarization trước.", None
    if not audio_state or len(audio_state) < 1 or not audio_state[0]:
        return "Thiếu thông tin file đã chuẩn hóa.", None

    prepared_path = Path(audio_state[0])
    tmp_root = Path(tempfile.mkdtemp(prefix="segments_"))
    output_dir = tmp_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    rows = _table_to_rows(table_rows)

    try:
        with metadata_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "id",
                    "file_name",
                    "start_mmss",
                    "end_mmss",
                    "gender",
                    "region",
                    "transcription",
                    "speaker",
                    "duration_sec",
                ]
            )

            for idx, seg in enumerate(segments_state):
                row = rows[idx] if idx < len(rows) else []
                # row order: start_mmss, end_mmss, speaker, gender, region, transcription
                gender = _normalize_label(row[3] if len(row) > 3 else "")
                region = _normalize_label(row[4] if len(row) > 4 else "")
                transcription = row[5] if len(row) > 5 else ""

                if gender and gender not in ALLOWED_GENDER:
                    return f"Lỗi: giới tính hàng {idx+1} phải là nam/nữ.", None
                if region and region not in ALLOWED_REGION:
                    return f"Lỗi: vùng miền hàng {idx+1} phải là bắc/trung/nam.", None

                gender_code = GENDER_MAP.get(gender, "")
                region_code = REGION_MAP.get(region, "")
                seg_id = f"id_{gender_code or 'x'}_{region_code or 'x'}_{idx:03d}"
                gender_disp = "Nam" if gender_code == "0" else "Nữ" if gender_code == "1" else gender
                region_disp = (
                    "Bắc"
                    if region_code == "0"
                    else "Trung"
                    if region_code == "1"
                    else "Nam"
                    if region_code == "2"
                    else region
                )

                start = float(seg["start"])
                end = float(seg["end"])
                duration = max(end - start, 0.0)
                out_file = output_dir / f"{seg_id}.wav"

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(prepared_path),
                    "-ss",
                    f"{start:.3f}",
                    "-to",
                    f"{end:.3f}",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-vn",
                    "-f",
                    "wav",
                    str(out_file),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    stderr = result.stderr.strip()
                    raise RuntimeError(f"ffmpeg lỗi khi tách đoạn {idx}: {stderr}")

                writer.writerow(
                    [
                        seg_id,
                        out_file.name,
                        seconds_to_mmss(start),
                        seconds_to_mmss(end),
                        gender_disp,
                        region_disp,
                        transcription,
                        seg.get("speaker", ""),
                        duration,
                    ]
                )

        # Tạo file zip từ thư mục output_dir
        zip_base_name = tmp_root / "segments_output"
        archive = shutil.make_archive(str(zip_base_name), "zip", output_dir)
        return f"Tách {len(segments_state)} đoạn thành công. Tải zip bên dưới.", archive
    except Exception as exc:  # pragma: no cover
        return f"Lỗi khi tách: {exc}", None


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Vietnamese Diarization", analytics_enabled=False) as demo:
        gr.Markdown(
            """
### Diarization tiếng Việt với pyannote
- Tải file âm thanh, điền Hugging Face access token (hoặc để trống nếu đã đặt trong môi trường/file).
- Chọn thiết bị chạy, nhấn Chạy. Kết quả hiển thị dạng bảng và file RTTM/JSON tải về.
"""
        )

        segments_state = gr.State([])
        audio_state = gr.State({})

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(label="Tải file audio (tùy chọn)", type="filepath")
                playback = gr.Audio(
                    label="Audio đã chuyển đổi/đang dùng",
                    type="filepath",
                    interactive=False,
                )
            with gr.Column():
                url_input = gr.Textbox(
                    label="URL YouTube/TikTok (tùy chọn)",
                    placeholder="Dán link video nếu không tải file",
                )
                token_input = gr.Textbox(
                    label="Hugging Face access token (tùy chọn)",
                    type="password",
                    placeholder="Để trống nếu đã cấu hình môi trường hoặc hugging_face_key.txt",
                )
                device_input = gr.Dropdown(
                    choices=["auto", "cpu", "cuda"],
                    value="auto",
                    label="Thiết bị",
                )
                run_btn = gr.Button("Chạy diarization")

        gr.Markdown(
            """
#### Gán nhãn và tách đoạn
- Chọn các ô gender (nam/nữ) và region (bắc/trung/nam) bằng dropdown trong bảng, transcription nhập tay.
- Nhấn "Tách và tải" để tải zip gồm các đoạn WAV và metadata.csv (không lưu lại trên server).
"""
        )
        segment_df = gr.DataFrame(
            headers=[
                "start_mmss",
                "end_mmss",
                "speaker",
                "gender",
                "region",
                "transcription",
            ],
            datatype="str",
            interactive=True,
            col_count=(6, "fixed"),
            row_count=(0, "dynamic"),
        )
        gender_dropdown = gr.Dropdown(choices=["", "nam", "nữ"], value="", label="Giới tính chọn nhanh")
        region_dropdown = gr.Dropdown(choices=["", "bắc", "trung", "nam"], value="", label="Vùng miền chọn nhanh")
        transcription_input = gr.Textbox(label="Transcription (áp dụng nhanh)", lines=1, placeholder="Nhập lời thoại")
        selection_info = gr.Textbox(label="Hàng đang chọn", interactive=False, value="Chưa chọn hàng")
        split_btn = gr.Button("Tách và tải")
        split_status = gr.Textbox(label="Trạng thái tách", lines=2)
        split_zip_file = gr.File(label="Tải ZIP các đoạn đã tách")

        gr.Markdown(
            """
#### Nhập ZIP đã tách (gộp nhiều ZIP thành một)
- Tải lên nhiều file ZIP đã tải về trước đó, công cụ sẽ gộp lại và tạo một merged.zip kèm metadata_all.csv.
"""
        )
        import_files = gr.File(label="Chọn nhiều ZIP", file_count="multiple", file_types=[".zip"])
        import_btn = gr.Button("Nhập ZIP vào thư mục chung")
        import_status = gr.Textbox(label="Trạng thái nhập ZIP", lines=2)
        merged_zip_file = gr.File(label="Tải ZIP đã gộp")

        result_box = gr.Textbox(label="Bảng phân đoạn", lines=12)
        rttm_file = gr.File(label="Tải RTTM")
        json_file = gr.File(label="Tải JSON")

        selected_row = gr.State(-1)

        run_btn.click(
            fn=_diarize_action,
            inputs=[audio_input, token_input, device_input, url_input],
            outputs=[result_box, rttm_file, json_file, segment_df, segments_state, audio_state, playback],
        )
        segment_df.select(
            fn=_select_row_action,
            inputs=None,
            outputs=[selection_info, selected_row],
        )
        gender_dropdown.change(
            fn=_apply_dropdown_action,
            inputs=[segment_df, selected_row, gender_dropdown, region_dropdown, transcription_input],
            outputs=[segment_df, selection_info],
        )
        region_dropdown.change(
            fn=_apply_dropdown_action,
            inputs=[segment_df, selected_row, gender_dropdown, region_dropdown, transcription_input],
            outputs=[segment_df, selection_info],
        )
        transcription_input.change(
            fn=_apply_dropdown_action,
            inputs=[segment_df, selected_row, gender_dropdown, region_dropdown, transcription_input],
            outputs=[segment_df, selection_info],
        )
        split_btn.click(
            fn=_split_segments_action,
            inputs=[segment_df, segments_state, audio_state],
            outputs=[split_status, split_zip_file],
        )
        import_btn.click(
            fn=_import_archives_action,
            inputs=[import_files],
            outputs=[import_status, merged_zip_file],
        )
    return demo


if __name__ == "__main__":
    import sys
    import os
    print("=" * 60, file=sys.stderr)
    print("Khởi tạo Vietnamese Diarization App...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    try:
        demo = build_interface()
        print("Interface đã được khởi tạo thành công!", file=sys.stderr)
        
        # Kiểm tra nếu đang chạy trên Hugging Face Space
        is_hf_space = os.getenv("SPACE_ID") is not None
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=is_hf_space,  # Tự động share nếu chạy trên HF Space
        )
    except Exception as e:
        print(f"LỖI khi khởi động app: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
