"""
Script 1: 01_download_videos.py
Purpose: Download all 139 TikTok videos via RapidAPI.
"""

import time
import logging
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config import (
    RAPIDAPI_KEY,
    RAPIDAPI_HOST,
    DATA_CSV,
    VIDEOS_DIR,
    RESULTS_DIR
)

# Configure logging
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# File handler
fh = logging.FileHandler(RESULTS_DIR / "download_errors.log", encoding="utf-8")
fh.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

def download_video(video_url: str, output_path: Path) -> bool:
    """Stream-download video_url with requests.get(..., stream=True) and write in 8192-byte chunks."""
    try:
        with requests.get(video_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Error downloading video from {video_url}: {e}")
        return False

def get_tiktok_data(tiktok_url: str) -> dict | None:
    """Call the RapidAPI TikTok Scraper endpoint to retrieve the direct download URL."""
    url = "https://tiktok-scraper7.p.rapidapi.com/"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    params = {"url": tiktok_url}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error calling RapidAPI for {tiktok_url}: {e}")
        return None

def main():
    """Main function to download all 139 TikTok videos."""
    if not DATA_CSV.exists():
        logger.error(f"Input file {DATA_CSV} not found.")
        return

    df = pd.read_csv(DATA_CSV)
    
    downloaded = 0
    skipped = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading videos"):
        video_id = str(row['ID'])
        tiktok_url = row['Link']
        output_path = VIDEOS_DIR / f"{video_id}.mp4"

        if output_path.exists():
            skipped += 1
            continue

        # Rate limit to 1 request/second
        time.sleep(1)

        data = get_tiktok_data(tiktok_url)
        if data and data.get("code") == 0:
            # Try no-watermark URL first, fall back to watermarked
            video_url = (
                data.get("data", {}).get("play")
                or data.get("data", {}).get("wmplay")
            )
            
            if video_url:
                if download_video(video_url, output_path):
                    downloaded += 1
                else:
                    failed += 1
                    logger.error(f"Failed to download video ID {video_id} from {video_url}")
            else:
                failed += 1
                logger.error(f"No video URL found for ID {video_id} in API response")
        else:
            failed += 1
            error_msg = data.get("msg") if data else "No response data"
            logger.error(f"API call failed for ID {video_id}: {error_msg}")

    print(f"\nSummary:")
    print(f"Total downloaded: {downloaded}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
