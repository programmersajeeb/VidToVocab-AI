# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import json
import tempfile
import urllib.request
import zipfile
from datetime import timedelta

import cv2
import pytesseract
import ffmpeg
import whisper
from ultralytics import YOLO

# CONFIG
video_url = input("🎬 Enter YouTube Video URL: ").strip()
video_path = "video.mp4"
audio_path = "audio.wav"
output_json = "outputs/video_data.json"
os.makedirs("outputs", exist_ok=True)

def ensure_ffmpeg_local():
    """
    Ensure ffmpeg is available. If not found on PATH, download a portable build
    into a local folder 'ffmpeg_portable' and prepend its bin folder to PATH.
    Returns True if ffmpeg is available after this function, False otherwise.
    """
    if shutil.which("ffmpeg"):
        return True

    print("ffmpeg not found on PATH — downloading portable ffmpeg into project folder...")
    target_root = os.path.join(os.getcwd(), "ffmpeg_portable")
    bin_dir = None

    # If already downloaded previously, try to use it
    for root, dirs, files in os.walk(target_root if os.path.isdir(target_root) else os.getcwd()):
        if "ffmpeg.exe" in files:
            bin_dir = root
            break

    if bin_dir is None:
        try:
            os.makedirs(target_root, exist_ok=True)
            url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            print("Downloading:", url)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmpf:
                tmp_path = tmpf.name
            urllib.request.urlretrieve(url, tmp_path)
            print("Extracting to", target_root)
            with zipfile.ZipFile(tmp_path, "r") as z:
                z.extractall(target_root)
            os.unlink(tmp_path)
            # Search for ffmpeg.exe inside extracted archive
            for root, dirs, files in os.walk(target_root):
                if "ffmpeg.exe" in files:
                    bin_dir = root
                    break
        except Exception as e:
            print("Auto-download of ffmpeg failed:", e)

    if bin_dir:
        # Prepend bin_dir to PATH for current process
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        if shutil.which("ffmpeg"):
            print("Portable ffmpeg ready.")
            return True
        else:
            print("ffmpeg executable found but not available via PATH after extraction.")
            return False

    print("Failed to prepare ffmpeg.")
    return False

# Check required external tools
def check_exec(name, install_hint=None, required=True):
    if shutil.which(name) is None:
        msg = f"'{name}' not found on PATH."
        if install_hint:
            msg += " " + install_hint
        if required:
            print("Error:", msg)
            sys.exit(1)
        else:
            print("Warning:", msg)

# Ensure yt-dlp (required) and tesseract (optional)
check_exec("yt-dlp", "Install with: choco install -y yt-dlp OR see https://github.com/yt-dlp/yt-dlp", required=True)
check_exec("tesseract", "Install with: choco install -y tesseract OR see https://github.com/tesseract-ocr/tesseract", required=False)

# Ensure ffmpeg (auto-download if missing)
if not ensure_ffmpeg_local():
    print("Please install ffmpeg manually and ensure ffmpeg.exe is on your PATH.")
    print("Examples (PowerShell, run as Administrator):")
    print("  winget install --id Gyan.FFmpeg.FFmpeg -e")
    print("  choco install -y ffmpeg")
    sys.exit(1)

# Optional: if tesseract is in non-standard location on Windows uncomment and set the path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# STEP 1: Download video
print("Downloading video...")
try:
    subprocess.run(
        ["yt-dlp", "-f", "best", video_url, "-o", video_path],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
except subprocess.CalledProcessError as e:
    print("yt-dlp download failed:", e.stderr or e)
    sys.exit(1)

# STEP 2: Extract audio
print("🎧 Extracting audio with ffmpeg...")
if not os.path.exists(video_path):
    print("Error: downloaded video not found:", video_path)
    sys.exit(1)

try:
    ffmpeg.input(video_path).output(audio_path, ac=1, ar=16000).run(overwrite_output=True)
except Exception as e:
    print("Error extracting audio with ffmpeg:", e)
    sys.exit(1)

# STEP 3: Transcribe audio with Whisper
print("🗣️ Transcribing audio (whisper)...")
if not os.path.exists(audio_path):
    print("Error: audio file not found:", audio_path)
    sys.exit(1)

try:
    model_whisper = whisper.load_model("small")
    transcript = model_whisper.transcribe(audio_path)
except Exception as e:
    print("Whisper transcription failed:", e)
    transcript = {"text": "", "segments": []}

# STEP 4: Video frame analysis (YOLO + OCR)
print("🧠 Analysing video frames (objects + text)...")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: cannot open video:", video_path)
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
frame_interval = max(1, int(round(fps)))  # sample ~1 frame per second; at least every frame
try:
    model_yolo = YOLO("yolov8n.pt")
except Exception as e:
    print("YOLO model load failed (ensure ultralytics installed and weights available):", e)
    model_yolo = None

frame_data = []
frame_no = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_no % frame_interval == 0:
        timestamp = frame_no / (fps if fps > 0 else 1)
        detections = []

        # Object detection (if model loaded)
        if model_yolo is not None:
            try:
                results = model_yolo(frame)
                for r in results:
                    for box in getattr(r, "boxes", []):
                        try:
                            raw_cls = getattr(box, "cls", None)
                            if raw_cls is None:
                                cls_name = "unknown"
                            else:
                                try:
                                    idx = int(raw_cls[0]) if hasattr(raw_cls, "__getitem__") else int(raw_cls)
                                except Exception:
                                    idx = int(raw_cls)
                                cls_name = model_yolo.names.get(idx, str(idx)) if hasattr(model_yolo, "names") else str(idx)
                        except Exception:
                            cls_name = "unknown"

                        try:
                            conf_val = float(getattr(box, "conf", 0.0)[0])
                        except Exception:
                            try:
                                conf_val = float(getattr(box, "conf", 0.0))
                            except Exception:
                                conf_val = 0.0

                        detections.append({"object": cls_name, "confidence": conf_val})
            except Exception as e:
                print("YOLO inference error on frame", frame_no, ":", e)

        # OCR (pytesseract)
        text_found = ""
        try:
            text_found = pytesseract.image_to_string(frame).strip()
        except Exception as e:
            print("pytesseract error on frame", frame_no, ":", e)

        frame_data.append(
            {
                "time": str(timedelta(seconds=int(timestamp))),
                "objects": detections,
                "text_on_screen": text_found,
            }
        )

    frame_no += 1

cap.release()

# STEP 5: Save results
data = {
    "video_url": video_url,
    "audio_transcript": transcript.get("text", ""),
    "segments": transcript.get("segments", []),
    "frames": frame_data,
}

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n✅ All data saved successfully to {output_json}")