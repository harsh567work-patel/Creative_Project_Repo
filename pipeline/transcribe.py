
import os
import json
import gc
import torch
from faster_whisper import WhisperModel

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOCAL_IN = _ROOT / "data" / "raw_downloads"
_DEFAULT_LOCAL_OUT = _ROOT / "data" / "whisper_transcripts"

WORKDIR = os.getenv("WHISPER_WORKDIR", "/content/whisper_job" if os.path.exists("/content/whisper_job") else str(_DEFAULT_LOCAL_IN))
OUTDIR = os.getenv("WHISPER_OUTDIR", os.path.join(WORKDIR, "transcripts_batch") if os.path.exists("/content/whisper_job") else str(_DEFAULT_LOCAL_OUT))
os.makedirs(OUTDIR, exist_ok=True)

# Find every uploaded/downloaded WEBM file, excluding output folders
video_files = sorted([
    os.path.join(WORKDIR, f)
    for f in os.listdir(WORKDIR)
    if f.lower().endswith((".webm", ".mp4", ".mp3"))
])

print(f"Found {len(video_files)} WEBM files")

# Use the same model as your successful transcription
model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",
    download_root="/content/whisper_models"
)

for index, video_path in enumerate(video_files, start=1):
    filename = os.path.basename(video_path)
    base = os.path.splitext(filename)[0]

    json_path = os.path.join(OUTDIR, base + ".json")
    txt_path = os.path.join(OUTDIR, base + ".txt")

    # Skip files already transcribed successfully
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        print(f"\n[{index}/{len(video_files)}] Skipping existing: {filename}")
        continue

    print(f"\n[{index}/{len(video_files)}] Transcribing: {filename}")

    segments, info = model.transcribe(
        video_path,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True
    )

    segment_list = []
    full_text = []

    for s in segments:
        text = s.text.strip()
        full_text.append(text)
        segment_list.append({
            "start": s.start,
            "end": s.end,
            "text": text,
            "words": [
                {
                    "start": w.start,
                    "end": w.end,
                    "word": w.word,
                    "probability": w.probability
                }
                for w in (s.words or [])
            ]
        })

    transcript_text = " ".join(full_text)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_file": filename,
            "video_id": (
                filename.rsplit("[", 1)[-1].split("]")[0]
                if "[" in filename and "]" in filename
                else None
            ),
            "language": info.language,
            "duration_sec": info.duration,
            "text": transcript_text,
            "segments": segment_list
        }, f, ensure_ascii=False, indent=2)

    print(
        f"Saved: {len(segment_list)} segments | "
        f"{len(transcript_text.split())} words | "
        f"language: {info.language}"
    )

print("\nBatch transcription finished!")
print(f"Transcripts saved in: {OUTDIR}")

del model
gc.collect()
torch.cuda.empty_cache()
