from faster_whisper import WhisperModel
from pathlib import Path
import time

# ==========================================
# SETTINGS
# ==========================================

INPUT_FOLDER = Path("mp3_files")
OUTPUT_FOLDER = Path("transcripts")

MODEL_SIZE = "large-v3"
LANGUAGE = "el"

SUPPORTED_EXTENSIONS = [
    "*.mp3",
    "*.m4a",
    "*.wav",
    "*.flac"
]

# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

OUTPUT_FOLDER.mkdir(exist_ok=True)

# ==========================================
# LOAD MODEL ON GPU
# ==========================================

print("Loading Whisper model on GPU...")

model = WhisperModel(
    MODEL_SIZE,
    device="cuda",
    compute_type="float16"
)

print("Model loaded successfully.\n")

# ==========================================
# FIND AUDIO FILES
# ==========================================

audio_files = []

for ext in SUPPORTED_EXTENSIONS:
    audio_files.extend(INPUT_FOLDER.glob(ext))

if not audio_files:
    print("No supported audio files found.")
    exit()

# ==========================================
# TRANSCRIBE FILES
# ==========================================

for audio_file in audio_files:

    print("=" * 60)
    print(f"Transcribing: {audio_file.name}")

    start_time = time.time()

    segments, info = model.transcribe(
        str(audio_file),
        language=LANGUAGE,
        task="transcribe",
        beam_size=5,
        best_of=5,
        vad_filter=True
    )

    lines = []

    for segment in segments:

        start = round(segment.start, 2)
        end = round(segment.end, 2)

        text = segment.text.strip()

        line = f"[{start}s -> {end}s] {text}"

        print(line)

        lines.append(line)

    # Save transcript
    output_file = OUTPUT_FOLDER / f"{audio_file.stem}.txt"

    output_file.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    elapsed = round(time.time() - start_time, 2)

    print(f"\nSaved transcript to:")
    print(output_file)

    print(f"Processing time: {elapsed} seconds")
    print("=" * 60)

print("\nAll files completed.")