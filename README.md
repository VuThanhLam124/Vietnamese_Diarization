---
title: Diarization Labeling
emoji: "\U0001F4E3"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.39.0"
app_file: app.py
pinned: false
---

# Vietnamese_Diarization

Kho mã mẫu diarization tiếng Việt dùng pyannote/speaker-diarization-community-1.

## Yêu cầu
- Python 3.10+
- ffmpeg (bắt buộc cho torchcodec audio decoding)
- Đã chấp nhận điều khoản model tại https://huggingface.co/pyannote/speaker-diarization-community-1
- Hugging Face access token (dán vào hugging_face_key.txt hoặc đặt biến môi trường HUGGINGFACE_TOKEN/HUGGINGFACE_ACCESS_TOKEN)

## Cài đặt nhanh
- Cài thư viện: `pip install pyannote.audio gradio` hoặc `uv add pyannote.audio gradio`
- Đảm bảo ffmpeg đã có trong PATH

## Chạy Gradio
- Lệnh: `python app.py`
- Trình duyệt mở tại http://localhost:7860 (hoặc địa chỉ máy chủ nếu chạy từ xa)
- Điền token nếu chưa đặt sẵn, tải file âm thanh, chọn thiết bị rồi nhấn Chạy
- Bảng kết quả hiển thị dạng phút:giây; có thể gán nhãn giới tính (nam/nữ), vùng miền (bắc/trung/nam) và transcription, sau đó bấm "Tách và tải" để nhận zip gồm các đoạn WAV và metadata.csv

## Chạy mẫu
- Diarization và in kết quả: `python infer.py path/to/audio.wav`
- Lưu thêm RTTM: `python infer.py path/to/audio.wav --rttm outputs/audio.rttm`
- Lưu JSON: `python infer.py path/to/audio.wav --json outputs/audio.json`
- Chọn thiết bị: thêm `--device cpu` hoặc `--device cuda` (mặc định auto)

## API Python
```
from app import diarize_file
segments = diarize_file("audio.wav", device="auto")
```

## Cấu trúc
- app.py: API Python và giao diện Gradio
- infer.py: CLI chạy diarization
- src/models.py: Bao gói pipeline pyannote
- src/utils.py: Hỗ trợ đọc token, định dạng kết quả
- hugging_face_key.txt: nơi dán Hugging Face access token (không commit token thật)
