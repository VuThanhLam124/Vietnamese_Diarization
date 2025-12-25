from __future__ import annotations

import argparse
from pathlib import Path

from src.models import DiarizationEngine
from src.utils import export_segments_json, format_segments_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy diarization bằng pyannote/speaker-diarization-community-1"
    )
    parser.add_argument("audio", help="Đường dẫn file âm thanh (wav, mp3, flac...)")
    parser.add_argument(
        "--hf-token",
        dest="hf_token",
        default=None,
        help="Hugging Face access token, nếu bỏ trống sẽ đọc từ hugging_face_key.txt",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Ưu tiên thiết bị chạy pipeline",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Tắt hiển thị tiến trình tải model/feature",
    )
    parser.add_argument(
        "--rttm",
        default=None,
        help="Đường dẫn lưu file RTTM (tùy chọn)",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="Đường dẫn lưu kết quả dạng JSON (tùy chọn)",
    )
    # Pipeline hyperparameters
    parser.add_argument(
        "--min-duration-off",
        type=float,
        default=None,
        help="Segmentation: thời gian im lặng tối thiểu (giây). Nhỏ hơn = nhạy hơn với khoảng dừng ngắn",
    )
    parser.add_argument(
        "--clustering-threshold",
        type=float,
        default=None,
        help="Clustering: ngưỡng khoảng cách speaker (0-1). Nhỏ hơn = nhiều speaker hơn",
    )
    parser.add_argument(
        "--clustering-method",
        choices=["centroid", "average", "ward", "complete", "single"],
        default=None,
        help="Clustering: phương pháp gom nhóm",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Clustering: số segment tối thiểu để tạo speaker cluster",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Build hyperparameters from CLI args
    segmentation_params = {}
    if args.min_duration_off is not None:
        segmentation_params["min_duration_off"] = args.min_duration_off
    
    clustering_params = {}
    if args.clustering_threshold is not None:
        clustering_params["threshold"] = args.clustering_threshold
    if args.clustering_method is not None:
        clustering_params["method"] = args.clustering_method
    if args.min_cluster_size is not None:
        clustering_params["min_cluster_size"] = args.min_cluster_size
    
    engine = DiarizationEngine(
        token=args.hf_token,
        device=args.device,
        segmentation_params=segmentation_params if segmentation_params else None,
        clustering_params=clustering_params if clustering_params else None,
    )
    diarization = engine.diarize(args.audio, show_progress=not args.no_progress)
    segments = engine.to_segments(diarization)

    print("Kết quả phân đoạn:")
    print(format_segments_table([seg.__dict__ for seg in segments]))

    if args.rttm:
        rttm_path = engine.save_rttm(diarization, args.rttm)
        print(f"Đã lưu RTTM tại: {rttm_path}")

    if args.json_out:
        json_path = export_segments_json([seg.__dict__ for seg in segments], args.json_out)
        print(f"Đã lưu JSON tại: {json_path}")


if __name__ == "__main__":
    main()
