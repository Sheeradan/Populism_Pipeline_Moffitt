"""
Script 4: 04_prosody_analysis.py
Purpose: Extract acoustic/prosodic features from each audio file using Praat via parselmouth.

Theoretical grounding:
  - Prosodic feature selection follows Braga & Marques (2004) "prosodic code" framework:
    six maxims (Pitch, Emphasis, Pitch Range, Phrasing, Silence) that jointly construct
    argumentative stance in political speech.
  - Composite score interpretation draws on Artero Abellan (2025), who demonstrates that
    populist delivery (Moffitt 2016) is prosodically characterized by steep pitch falls,
    compressed tempo, high intensity, and creaky/pressed phonation — whereas institutional
    register features controlled pitch, slowed tempo, moderate intensity, and breath-supported
    phonation. This contrast motivates the three-component composite.
  - Prosodic features are supplied to the LLM classifier as an explicit auxiliary channel,
    following Tsiamas et al. (2024) finding that text-based models lack direct access to
    prosodic information and require explicit prosodic encoding to leverage it.

Changes from original:
  1. Reference ranges computed empirically (corpus min/max) — not hardcoded.
  2. Composite renamed to 'prosodic_formality_score' to reflect what is measured:
     the degree of vocal control/formality, NOT Bad Manners per se.
     Score direction: 1.0 = maximally controlled/formal, 0.0 = maximally agitated/informal.
  3. Syllable-detection indexing bug fixed.
  4. Two-pass architecture: Pass 1 extracts raw features, Pass 2 normalizes.
  5. Added pitch contour slope (mean F0 gradient per prosodic phrase) as a feature,
     following Artero Abellan (2025): steep terminal falls index assertive/populist stance,
     while gradual rise-fall contours index affiliative/institutional stance.
  6. Added speech-rate variability (CV of local speech rate) following Braga & Marques (2004)
     Maxim of Phrasing: rhythmic regularity vs. irregularity is a pragmatic signal.
"""

import json
import logging
import numpy as np
import parselmouth
from parselmouth.praat import call
from pathlib import Path
from tqdm import tqdm
from config import (
    AUDIO_DIR,
    PROSODY_DIR,
    RESULTS_DIR
)

# Configure logging
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROSODY_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fh = logging.FileHandler(RESULTS_DIR / "prosody_errors.log", encoding="utf-8")
fh.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)


def normalize(val, low, high, invert=False):
    """Normalize a value to [0.0, 1.0] range based on reference bounds."""
    if high == low:
        return 0.5
    score = max(0.0, min(1.0, (val - low) / (high - low)))
    return 1.0 - score if invert else score


def compute_pitch_slope(snd, pitch_obj) -> float:
    """
    Compute mean F0 gradient (Hz/s) across the utterance.

    Motivation (Artero Abellan 2025): steep negative slopes characterize
    assertive/populist terminal falls; gentle slopes characterize
    affiliative/institutional contours.

    Returns absolute mean slope — higher values = more dramatic pitch movement.
    """
    f0_values = pitch_obj.selected_array["frequency"]
    times = pitch_obj.xs()
    # Keep only voiced frames
    voiced_mask = f0_values > 0
    f0_voiced = f0_values[voiced_mask]
    t_voiced = np.array(times)[voiced_mask]

    if len(f0_voiced) < 2:
        return 0.0

    # Local gradient: |df0/dt| between consecutive voiced frames
    df0 = np.diff(f0_voiced)
    dt = np.diff(t_voiced)
    dt[dt == 0] = 1e-6  # avoid division by zero
    slopes = np.abs(df0 / dt)

    return float(np.mean(slopes))


def compute_speech_rate_variability(intensity_obj, mean_db: float, duration: float) -> float:
    """
    Compute coefficient of variation of local speech rate (syllable peaks per window).

    Motivation (Braga & Marques 2004, Maxim of Phrasing): rhythmic regularity
    signals controlled/institutional delivery; high variability signals
    agitated or dramatized delivery.

    Uses 2-second sliding windows with 1-second step.
    Returns CV (std/mean) of per-window syllable counts. Higher = more irregular.
    """
    intensity_values = intensity_obj.values.T.flatten()
    times = np.array(intensity_obj.xs())
    silence_threshold = mean_db - 15

    window_size = 2.0  # seconds
    step = 1.0

    counts = []
    t_start = times[0]
    while t_start + window_size <= times[-1]:
        mask = (times >= t_start) & (times < t_start + window_size)
        window_vals = intensity_values[mask]

        # Count peaks in window
        n_peaks = 0
        for i in range(1, len(window_vals) - 1):
            if (window_vals[i] > window_vals[i - 1]
                    and window_vals[i] > window_vals[i + 1]
                    and window_vals[i] > silence_threshold):
                n_peaks += 1
        counts.append(n_peaks)
        t_start += step

    if len(counts) < 2 or np.mean(counts) == 0:
        return 0.0

    return float(np.std(counts) / np.mean(counts))


def extract_raw_features(audio_path: Path) -> dict | None:
    """
    Pass 1: Extract raw prosodic features without normalization.
    Returns a dict of raw values, or None on failure.
    """
    try:
        snd = parselmouth.Sound(str(audio_path))
        duration = snd.duration

        # --- Pitch ---
        pitch = snd.to_pitch()
        f0_values = pitch.selected_array["frequency"]
        f0_voiced = f0_values[f0_values > 0]

        if len(f0_voiced) == 0:
            mean_f0 = std_f0 = min_f0 = max_f0 = 0.0
        else:
            mean_f0 = float(np.mean(f0_voiced))
            std_f0 = float(np.std(f0_voiced))
            min_f0 = float(np.min(f0_voiced))
            max_f0 = float(np.max(f0_voiced))

        # --- Pitch slope (NEW — Artero Abellan 2025) ---
        mean_pitch_slope = compute_pitch_slope(snd, pitch)

        # --- Intensity ---
        intensity = snd.to_intensity()
        intensity_values = intensity.values.T.flatten()
        mean_db = float(np.mean(intensity_values))
        std_db = float(np.std(intensity_values))
        min_db = float(np.min(intensity_values))
        max_db = float(np.max(intensity_values))

        # --- HNR (voice quality — higher = cleaner/more controlled) ---
        # Braga & Marques (2004): voice quality indexes power and assertiveness
        # Artero Abellan (2025): pressed/creaky phonation → low HNR → populist style;
        #                        breath-supported phonation → high HNR → institutional style
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = float(call(harmonicity, "Get mean", 0, 0))

        # --- Jitter and shimmer ---
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        jitter = float(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        shimmer = float(call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

        # --- Pause detection (Braga & Marques 2004: Maxim of Silence) ---
        silence_threshold_db = mean_db - 15
        pauses = []
        in_pause = False
        pause_start = 0.0
        for t, val in zip(intensity.xs(), intensity_values):
            if val < silence_threshold_db and not in_pause:
                in_pause = True
                pause_start = t
            elif val >= silence_threshold_db and in_pause:
                duration_pause = t - pause_start
                if duration_pause > 0.3:
                    pauses.append(duration_pause)
                in_pause = False

        num_pauses = len(pauses)
        total_pause_duration = sum(pauses)
        mean_pause_duration = total_pause_duration / num_pauses if num_pauses > 0 else 0.0

        # --- Speech rate (syllable approximation, fixed indexing) ---
        num_syllables_approx = 0
        for i in range(1, len(intensity_values) - 1):
            if (intensity_values[i] > intensity_values[i - 1]
                    and intensity_values[i] > intensity_values[i + 1]
                    and intensity_values[i] > silence_threshold_db):
                num_syllables_approx += 1

        speech_rate = num_syllables_approx / duration if duration > 0 else 0.0

        # --- Speech rate variability (NEW — Braga & Marques 2004) ---
        speech_rate_cv = compute_speech_rate_variability(intensity, mean_db, duration)

        return {
            "id": audio_path.stem,
            "duration_sec": round(duration, 2),
            "pitch": {
                "mean_f0_hz": round(mean_f0, 2),
                "std_f0_hz": round(std_f0, 2),
                "min_f0_hz": round(min_f0, 2),
                "max_f0_hz": round(max_f0, 2),
                "f0_range_hz": round(max_f0 - min_f0, 2),
                "mean_slope_hz_per_sec": round(mean_pitch_slope, 2),
            },
            "intensity": {
                "mean_db": round(mean_db, 2),
                "std_db": round(std_db, 2),
                "min_db": round(min_db, 2),
                "max_db": round(max_db, 2)
            },
            "speech_rate": {
                "num_syllables_approx": num_syllables_approx,
                "rate_syllables_per_sec": round(speech_rate, 2),
                "rate_cv": round(speech_rate_cv, 4),
            },
            "voice_quality": {
                "mean_hnr_db": round(hnr, 2),
                "mean_jitter_local": round(jitter, 4),
                "mean_shimmer_local": round(shimmer, 4)
            },
            "pauses": {
                "num_pauses": num_pauses,
                "total_pause_duration_sec": round(total_pause_duration, 2),
                "mean_pause_duration_sec": round(mean_pause_duration, 2)
            },
            # Raw values for Pass 2 normalization
            "_raw_std_f0": std_f0,
            "_raw_hnr": hnr,
            "_raw_std_db": std_db,
        }
    except Exception as e:
        logger.error(f"Error analyzing prosody for {audio_path}: {e}")
        return None


def compute_corpus_ranges(all_features: list[dict]) -> dict:
    """
    Compute empirical min/max for each composite score component across the corpus.
    """
    std_f0_vals = [f["_raw_std_f0"] for f in all_features]
    hnr_vals = [f["_raw_hnr"] for f in all_features]
    std_db_vals = [f["_raw_std_db"] for f in all_features]

    ranges = {
        "pitch_std": (min(std_f0_vals), max(std_f0_vals)),
        "hnr":       (min(hnr_vals),    max(hnr_vals)),
        "int_std":   (min(std_db_vals), max(std_db_vals)),
    }

    logger.info(f"Empirical reference ranges (corpus N={len(all_features)}):")
    logger.info(f"  pitch_std_hz : {ranges['pitch_std'][0]:.2f} – {ranges['pitch_std'][1]:.2f}")
    logger.info(f"  hnr_db       : {ranges['hnr'][0]:.2f} – {ranges['hnr'][1]:.2f}")
    logger.info(f"  intensity_std: {ranges['int_std'][0]:.2f} – {ranges['int_std'][1]:.2f}")

    return ranges


def add_composite_score(features: dict, ranges: dict) -> dict:
    """
    Pass 2: Compute the prosodic_formality_score using empirical corpus ranges.

    Score direction:
        1.0 = maximally controlled/formal delivery
              (institutional style: narrow pitch range, clean voice, even intensity)
        0.0 = maximally agitated/informal delivery
              (populist style: wide pitch range, rough voice, dynamic intensity)

    Theoretical basis:
        - Artero Abellan (2025): populist prosody = steep pitch, high intensity,
          pressed phonation; institutional prosody = controlled pitch, moderate
          intensity, breath-supported phonation
        - Braga & Marques (2004): assertive modality uses all prosodic maxims
          to convey determination; emphasis/pitch range mark key content

    Components (equal weight — justified by parsimony in absence of
    theoretically motivated weighting; cf. discussion in methodology):
        - pitch_std  (inverted): high pitch variability → lower score
        - HNR        (direct):   high HNR (clean voice) → higher score
        - int_std    (inverted): high intensity variability → lower score
    """
    pitch_score = normalize(features["_raw_std_f0"], *ranges["pitch_std"], invert=True)
    hnr_score = normalize(features["_raw_hnr"], *ranges["hnr"])
    intens_score = normalize(features["_raw_std_db"], *ranges["int_std"], invert=True)

    composite = (pitch_score + hnr_score + intens_score) / 3.0

    # Remove temporary raw fields
    features.pop("_raw_std_f0", None)
    features.pop("_raw_hnr", None)
    features.pop("_raw_std_db", None)

    # Add final score with transparent sub-components
    features["prosodic_formality_score"] = {
        "composite": round(composite, 2),
        "pitch_variability_norm": round(pitch_score, 2),
        "hnr_norm": round(hnr_score, 2),
        "intensity_variability_norm": round(intens_score, 2),
    }

    return features


def main():
    """
    Two-pass prosody analysis:
      Pass 1 — Extract raw features for all audio files.
      Pass 2 — Compute empirical ranges, normalize, and write output JSONs.
    """
    audio_files = sorted(AUDIO_DIR.glob("*.wav"))

    if not audio_files:
        logger.warning(f"No .wav files found in {AUDIO_DIR}")
        return

    # ------------------------------------------------------------------
    # Pass 1: Raw extraction
    # ------------------------------------------------------------------
    logger.info(f"Pass 1: Extracting raw features from {len(audio_files)} files...")
    all_features: list[dict] = []
    failed = 0

    for audio_path in tqdm(audio_files, desc="Pass 1 – raw extraction"):
        features = extract_raw_features(audio_path)
        if features:
            all_features.append(features)
        else:
            failed += 1

    if not all_features:
        logger.error("No features extracted — aborting.")
        return

    # ------------------------------------------------------------------
    # Pass 2: Empirical normalization + composite score
    # ------------------------------------------------------------------
    logger.info("Pass 2: Computing empirical ranges and composite scores...")
    ranges = compute_corpus_ranges(all_features)

    # Save ranges for reproducibility
    ranges_serializable = {k: {"min": round(v[0], 4), "max": round(v[1], 4)}
                           for k, v in ranges.items()}
    with open(PROSODY_DIR / "_corpus_ranges.json", "w", encoding="utf-8") as f:
        json.dump(ranges_serializable, f, indent=2)
    logger.info(f"Saved corpus ranges to {PROSODY_DIR / '_corpus_ranges.json'}")

    written = 0
    skipped = 0
    for features in tqdm(all_features, desc="Pass 2 – normalization"):
        output_path = PROSODY_DIR / f"{features['id']}.json"
        if output_path.exists():
            skipped += 1
            continue

        features = add_composite_score(features, ranges)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(features, f, indent=2)
        written += 1

    print(f"\nSummary:")
    print(f"  Total audio files : {len(audio_files)}")
    print(f"  Features extracted: {len(all_features)}")
    print(f"  JSONs written     : {written}")
    print(f"  Skipped (exist)   : {skipped}")
    print(f"  Failed            : {failed}")


if __name__ == "__main__":
    main()
