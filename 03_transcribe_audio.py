"""
Script 3: 03_transcribe_audio.py
Purpose: Transcribe German audio to text using WhisperX with word-level timestamps.
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
import whisperx
from config import (
    AUDIO_DIR,
    TRANSCRIPTS_DIR,
    RESULTS_DIR,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE,
    WHISPER_BATCH_SIZE
)
import soundfile as sf
import numpy as np

# Configure logging
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# File handler
fh = logging.FileHandler(RESULTS_DIR / "transcription_errors.log", encoding="utf-8")
fh.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

def load_audio_sf(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load wav using soundfile — no FFmpeg required."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != sample_rate:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
    return data

def main():
    """Main function to transcribe all extracted audio files."""
    # Load model once
    logger.info(f"Loading WhisperX model: {WHISPER_MODEL} on {WHISPER_DEVICE}...")
    try:
        model = whisperx.load_model(
            WHISPER_MODEL,
            WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            language=WHISPER_LANGUAGE
        )
        
        # Load align model
        model_a, metadata = whisperx.load_align_model(
            language_code=WHISPER_LANGUAGE,
            device=WHISPER_DEVICE
        )
    except Exception as e:
        logger.error(f"Failed to load WhisperX models: {e}")
        return

    audio_files = list(AUDIO_DIR.glob("*.wav"))
    
    transcribed = 0
    skipped = 0
    failed = 0

    for audio_path in tqdm(audio_files, desc="Transcribing audio"):
        audio_id = audio_path.stem
        json_output = TRANSCRIPTS_DIR / f"{audio_id}.json"
        txt_output = TRANSCRIPTS_DIR / f"{audio_id}.txt"

        if json_output.exists():
            skipped += 1
            continue

        try:
            audio = load_audio_sf(str(audio_path))
            result = model.transcribe(audio, batch_size=WHISPER_BATCH_SIZE, language=WHISPER_LANGUAGE)

            # Word-level alignment
            result = whisperx.align(
                result["segments"], model_a, metadata, audio, WHISPER_DEVICE
            )

            # Save full JSON output
            with open(json_output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Save plain text
            full_text = " ".join(seg["text"].strip() for seg in result["segments"])
            with open(txt_output, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            transcribed += 1
        except Exception as e:
            logger.error(f"Error transcribing ID {audio_id}: {e}")
            failed += 1

    print(f"\nSummary:")
    print(f"Total transcribed: {transcribed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
