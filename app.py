from __future__ import annotations

import warnings
# Suppress pyannote TF32 warning và transformers FutureWarning
warnings.filterwarnings("ignore", message="TensorFloat-32")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

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
from src.asr import ASREngine
from src.profiling import ProfilingEngine

DEFAULT_TOKEN_SENTINEL = "__FROM_FILE_OR_ENV__"
GENDER_MAP = {"nam": "0", "male": "0", "nữ": "1", "nu": "1", "female": "1"}
REGION_MAP = {"bắc": "0", "bac": "0", "north": "0", "trung": "1", "central": "1", "nam": "2", "south": "2"}
ALLOWED_GENDER = {"nam", "nữ", "nu", "male", "female"}
ALLOWED_REGION = {"bắc", "trung", "nam", "bac", "north", "central", "south"}

import torch

# Workaround cho PyTorch 2.6 - force weights_only=False
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load


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
    merge_gap: float = 5.0,
):
    import sys
    print(f"DEBUG START: audio_path={audio_path}, url={url}", file=sys.stderr)
    
    if not audio_path and not url:
        empty_state = ["", "", "", ""]
        return "Vui lòng tải file âm thanh hoặc nhập URL.", None, None, [], [], empty_state, None
    try:
        downloaded_path = None
        download_tmp = None
        audio_input = audio_path
        if url:
            print(f"DEBUG: Downloading from URL: {url}", file=sys.stderr)
            downloaded_path, download_tmp = download_audio_from_url(url)
            audio_input = str(downloaded_path)
            print(f"DEBUG: Downloaded to: {audio_input}, tmp={download_tmp}", file=sys.stderr)

        print(f"DEBUG: Getting engine...", file=sys.stderr)
        engine = _get_engine(_token_key(hf_token), device)
        print(f"DEBUG: Running diarization on: {audio_input}, merge_gap={merge_gap}s", file=sys.stderr)
        diarization, prepared_path, prep_tmpdir = engine.diarize(
            audio_input, show_progress=False, keep_audio=True
        )
        print(f"DEBUG: Diarization done. prepared_path={prepared_path}, prep_tmpdir={prep_tmpdir}", file=sys.stderr)
        raw_segments = engine.to_segments(diarization)
        
        # Merge segments cùng speaker
        from src.utils import merge_adjacent_segments
        dict_segments = [{'start': s.start, 'end': s.end, 'speaker': s.speaker} for s in raw_segments]
        merged = merge_adjacent_segments(dict_segments, max_gap=merge_gap, min_duration=0.5)
        segments = [Segment(start=s['start'], end=s['end'], speaker=s['speaker']) for s in merged]
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
        # Đảm bảo prepared_path là file path hợp lệ, không phải None hoặc directory
        audio_file_output = str(prepared_path) if prepared_path and Path(prepared_path).is_file() else None
        
        # Debug logging
        import sys
        print(f"DEBUG: prepared_path={prepared_path}, type={type(prepared_path)}", file=sys.stderr)
        print(f"DEBUG: rttm_path={rttm_path}, exists={Path(rttm_path).exists()}", file=sys.stderr)
        print(f"DEBUG: json_path={json_path}, exists={Path(json_path).exists()}", file=sys.stderr)
        print(f"DEBUG: audio_file_output={audio_file_output}", file=sys.stderr)
        
        return (
            table,
            str(rttm_path),
            str(json_path),
            df_rows,
            dict_segments,
            audio_state,
            audio_file_output,
        )
    except Exception as exc:  # pragma: no cover - hiển thị lỗi cho người dùng giao diện
        import traceback
        print(f"DEBUG ERROR: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        empty_state = ["", "", "", ""]
        return f"Lỗi: {exc}", None, None, [], [], empty_state, None


def _normalize_label(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _extract_label_from_auto(value: Any) -> str:
    """Extract label từ format auto-label có confidence, ví dụ: 'Nam (85%)' -> 'nam'."""
    if value is None:
        return ""
    text = str(value).strip()
    
    # Nếu có dạng "Label (XX%)" hoặc "Label (XX%)⚠️", lấy phần trước dấu (
    if "(" in text:
        text = text.split("(")[0].strip()
    
    # Remove emoji nếu có
    text = text.replace("⚠️", "").strip()
    
    return text.lower()


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


def _auto_label_action(
    table_rows: list[list[Any]] | None,
    segments_state: list[dict],
    audio_state: list[str],
    asr_model: str,
    confidence_threshold: float,
):
    """Auto-label segments với ASR transcription và speaker profiling."""
    import sys
    
    if not segments_state:
        return table_rows or [], "Chạy diarization trước."
    if not audio_state or len(audio_state) < 1 or not audio_state[0]:
        return table_rows or [], "Thiếu thông tin file audio."
    
    prepared_path = Path(audio_state[0])
    if not prepared_path.exists():
        return table_rows or [], f"Không tìm thấy file audio: {prepared_path}"
    
    rows = _table_to_rows(table_rows)
    if len(rows) != len(segments_state):
        rows = [[]] * len(segments_state)
    
    print(f"DEBUG: Auto-labeling {len(segments_state)} segments...", file=sys.stderr)
    print(f"DEBUG: ASR model={asr_model}, threshold={confidence_threshold}", file=sys.stderr)
    
    try:
        # Initialize engines
        asr_engine = ASREngine(model_id=asr_model, device="auto")
        profiler = ProfilingEngine(device="auto")
        
        # Process segments
        tmpdir = Path(tempfile.mkdtemp(prefix="auto_label_"))
        new_rows = []
        low_confidence_count = 0
        
        for idx, seg in enumerate(segments_state):
            start = float(seg["start"])
            end = float(seg["end"])
            speaker = seg.get("speaker", "")
            
            # Get existing row data
            old_row = rows[idx] if idx < len(rows) else []
            start_mmss = old_row[0] if len(old_row) > 0 else seconds_to_mmss(start)
            end_mmss = old_row[1] if len(old_row) > 1 else seconds_to_mmss(end)
            
            try:
                # ASR transcription
                transcription = asr_engine.transcribe_segment(
                    prepared_path, start, end, tmpdir=tmpdir
                )
                
                # Speaker profiling
                profile = profiler.profile_segment(
                    prepared_path, start, end, tmpdir=tmpdir
                )
                
                # Format với confidence indicators
                gender_str = profile.gender
                dialect_str = profile.dialect
                gender_conf = profile.gender_confidence
                dialect_conf = profile.dialect_confidence
                
                # Add confidence % và marker nếu thấp
                if gender_conf < confidence_threshold:
                    gender_str = f"{profile.gender} ({gender_conf:.0%})⚠️"
                    low_confidence_count += 1
                else:
                    gender_str = f"{profile.gender} ({gender_conf:.0%})"
                
                if dialect_conf < confidence_threshold:
                    dialect_str = f"{profile.dialect} ({dialect_conf:.0%})⚠️"
                    low_confidence_count += 1
                else:
                    dialect_str = f"{profile.dialect} ({dialect_conf:.0%})"
                
                new_rows.append([
                    start_mmss,
                    end_mmss,
                    speaker,
                    gender_str,
                    dialect_str,
                    transcription,
                ])
                
            except Exception as e:
                print(f"DEBUG: Error processing segment {idx}: {e}", file=sys.stderr)
                new_rows.append([
                    start_mmss,
                    end_mmss,
                    speaker,
                    "Error",
                    "Error",
                    f"[Lỗi: {str(e)[:50]}]",
                ])
        
        status = f"Auto-label hoàn tất {len(segments_state)} đoạn."
        if low_confidence_count > 0:
            status += f" ⚠️ {low_confidence_count} nhãn có confidence thấp (< {confidence_threshold:.0%}) cần review."
        
        return new_rows, status
        
    except Exception as e:
        import traceback
        print(f"DEBUG: Auto-label error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return rows, f"Lỗi auto-label: {e}"


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
                # Sử dụng _extract_label_from_auto để xử lý format "Nam (85%)" -> "nam"
                gender = _extract_label_from_auto(row[3] if len(row) > 3 else "")
                region = _extract_label_from_auto(row[4] if len(row) > 4 else "")
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
                    info="⚠️ Lưu ý: URL download không hoạt động trên HF Spaces (free tier). Vui lòng tải file trực tiếp.",
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
                merge_gap_slider = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    value=2.0,
                    step=0.5,
                    label="Merge Gap (giây) - khoảng trống tối đa để gộp đoạn cùng speaker",
                    info="Tăng giá trị để gộp nhiều đoạn nhỏ thành đoạn dài hơn. Khuyến nghị: 2-3s",
                )
                run_btn = gr.Button("Chạy diarization")

        gr.Markdown(
            """
#### Gán nhãn và tách đoạn
- Chọn các ô gender (nam/nữ) và region (bắc/trung/nam) bằng dropdown trong bảng, transcription nhập tay.
- Nhấn "Tách và tải" để tải zip gồm các đoạn WAV và metadata.csv (không lưu lại trên server).
"""
        )
        segment_df = gr.Dataframe(
            value=[],
            headers=["start_mmss", "end_mmss", "speaker", "gender", "region", "transcription"],
            datatype=["str", "str", "str", "str", "str", "str"],
            interactive=True,
            type="array",
        )
        gender_dropdown = gr.Dropdown(choices=["", "nam", "nữ"], value="", label="Giới tính chọn nhanh")
        region_dropdown = gr.Dropdown(choices=["", "bắc", "trung", "nam"], value="", label="Vùng miền chọn nhanh")
        transcription_input = gr.Textbox(label="Transcription (áp dụng nhanh)", lines=1, placeholder="Nhập lời thoại")
        selection_info = gr.Textbox(label="Hàng đang chọn", interactive=False, value="Chưa chọn hàng")
        
        gr.Markdown(
            """
#### Auto Label (ASR + Gender + Dialect)
- Chọn model ASR và ngưỡng confidence, nhấn "Auto Label" để tự động nhận diện.
- Ô có ⚠️ nghĩa là confidence thấp, cần review thủ công.
"""
        )
        with gr.Row():
            asr_model_dropdown = gr.Dropdown(
                choices=[
                    "vinai/PhoWhisper-tiny",
                    "vinai/PhoWhisper-base",
                    "vinai/PhoWhisper-small",
                    "vinai/PhoWhisper-medium",
                    "vinai/PhoWhisper-large",
                ],
                value="vinai/PhoWhisper-base",
                label="ASR Model",
            )
            confidence_slider = gr.Slider(
                minimum=0.5,
                maximum=1.0,
                value=0.8,
                step=0.05,
                label="Ngưỡng confidence (⚠️ nếu thấp hơn)",
            )
        auto_label_btn = gr.Button("Auto Label (ASR + Gender + Dialect)", variant="primary")
        auto_label_status = gr.Textbox(label="Trạng thái Auto Label", lines=2)
        
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
            inputs=[audio_input, token_input, device_input, url_input, merge_gap_slider],
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
        auto_label_btn.click(
            fn=_auto_label_action,
            inputs=[segment_df, segments_state, audio_state, asr_model_dropdown, confidence_slider],
            outputs=[segment_df, auto_label_status],
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Vietnamese Diarization Gradio App")
    parser.add_argument("--share", action="store_true", 
                        help="Create public URL (useful for Kaggle/Colab)")
    parser.add_argument("--port", type=int, default=7860, 
                        help="Server port (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host (default: 0.0.0.0)")
    args = parser.parse_args()
    
    print("=" * 60, file=sys.stderr)
    print("Khởi tạo Vietnamese Diarization App...", file=sys.stderr)
    if args.share:
        print("🌐 Public URL sẽ được tạo (share=True)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    try:
        demo = build_interface()
        print("Interface đã được khởi tạo thành công!", file=sys.stderr)
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
        )
    except Exception as e:
        print(f"LỖI khi khởi động app: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
