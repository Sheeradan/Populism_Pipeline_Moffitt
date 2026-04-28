"""
Script 2: 02_extract_audio.py
Purpose: Extract 16kHz mono WAV audio from each downloaded video using MoviePy.
# REQUIREMENTS: moviepy imageio-ffmpeg
"""

import logging
from pathlib import Path
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from config import (
    VIDEOS_DIR,
    AUDIO_DIR,
    RESULTS_DIR,
    SAMPLE_RATE
)

# Configure logging
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# File handler
fh = logging.FileHandler(RESULTS_DIR / "audio_errors.log", encoding="utf-8")
fh.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

def extract_audio(video_path: Path, output_path: Path, sample_rate: int = 16000) -> str:
    """Extract mono WAV audio from video using MoviePy. Returns 'success', 'photo', or 'failed'."""
    try:
        with VideoFileClip(str(video_path)) as clip:
            if clip.audio is None:
                return "photo"
            clip.audio.write_audiofile(
                str(output_path),
                fps=sample_rate,
                nbytes=2,        # 16-bit PCM
                ffmpeg_params=["-ac", "1"],  # force mono
                logger=None      # suppress MoviePy progress spam
            )
        return "success"
    except Exception as e:
        if "video_fps" in str(e):
            return "photo"
        logger.error(f"Error extracting audio from {video_path}: {e}")
        return "failed"

def main():
    """Main function to extract audio from all downloaded videos."""
    video_files = list(VIDEOS_DIR.glob("*.mp4"))
    
    extracted = 0
    skipped = 0
    photos = 0
    failed = 0

    for video_path in tqdm(video_files, desc="Extracting audio"):
        video_id = video_path.stem
        output_path = AUDIO_DIR / f"{video_id}.wav"

        if output_path.exists():
            skipped += 1
            continue

        res = extract_audio(video_path, output_path, SAMPLE_RATE)
        if res == "success":
            extracted += 1
        elif res == "photo":
            photos += 1
        else:
            failed += 1

    print(f"\nSummary:")
    print(f"Total extracted: {extracted}")
    print(f"Skipped: {skipped}")
    print(f"Photo posts skipped: {photos}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
