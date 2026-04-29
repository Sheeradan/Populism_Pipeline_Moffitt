# Populism Pipeline — Moffitt (2016)

A replicable research pipeline for classifying populist style in political TikTok videos. Given a spreadsheet with TikTok video URLs, the pipeline downloads videos, extracts audio, transcribes speech, and classifies each transcript for four binary populist style variables using a locally running large language model.

It was built for a BA thesis analysing AfD politicians on TikTok using Benjamin Moffitt's (2016) populist style framework, but the design is general. You can apply it to any set of political TikTok accounts, swap the language, and define whatever grouping variable you want.

## What the pipeline produces

A validated CSV file with four binary Moffitt (2016) populist style codes per video. Each transcript is classified twice with differently structured prompts; where both passes agree the code is accepted, where they disagree the case is flagged for manual review. The output CSV includes per-variable scores, reasoning text from both passes, and a disagreement column identifying which variables need human verification.

Statistical analysis and visualization were performed separately in RStudio and Python and are not included in this repository.

## The four stages

**Stage 1 — Video download** (`01_download_videos.py`)
Reads your URL list and downloads each video as an .mp4 using the TikTok API (via RapidAPI). Photo slideshows are detected and excluded automatically.

**Stage 2 — Audio extraction** (`02_extract_audio.py`)
Pulls audio from each .mp4 and saves it as a standardised .wav file (16 kHz, mono, 16-bit PCM) using FFmpeg via MoviePy.

**Stage 3 — Transcription** (`03_transcribe_audio.py`)
Transcribes each audio file using WhisperX with the large-v2 model. Configured for German but works with any language Whisper supports. Output is a plain-text transcript and a JSON file with word-level timestamps.

**Stage 4 — LLM classification** (`04_llm_classify.py`)
Classifies each transcript for four binary populist style variables using Gemma 3 27b via Ollama, so no API costs and no data leaves your machine. Classification runs twice:

- Pass A presents full theoretical definitions first, then asks for scores.
- Pass B opens with known error patterns and calibration rules, then reverses the variable order to stress-test Bad Manners first.

Where both passes agree, the code is accepted. Where they disagree, the case is flagged for manual review against the original video. The script checkpoints every 10 videos, so a crash never loses more than 10 videos of work.

## What gets coded

**Populist style (LLM-coded):**

- Appeal to the People — does the speaker construct a shared "people" and position themselves as one of them?
- Anti-Elitism — does the speaker explicitly name and delegitimise an elite antagonist?
- Bad Manners — does the speaker use mockery, sarcasm, colloquial register, hyperbolic labels, or performative outrage?
- Crisis / Breakdown / Threat — does the speaker frame the current situation as an emergency or existential threat?

**Performative register** (human-coded, entered in your spreadsheet — not processed by this pipeline):

- Dress code (formal vs. informal)
- Setting (studio/office vs. casual/outdoor)
- Production quality (professional vs. smartphone)
- Video format (talking-head vs. other)

## What you need

- A spreadsheet (CSV) with at least one column of TikTok URLs and a unique video ID per row.
- A RapidAPI key for the TikTok scraper (free tier covers small corpora).
- Python 3.11+ with a virtual environment.
- Ollama installed locally with gemma3:27b pulled (~17 GB download, runs on a modern GPU or slowly on CPU).
- FFmpeg installed on your system.
- A CUDA-capable GPU for WhisperX (strongly recommended; CPU transcription is very slow).

## How to adapt it

- **Different language:** change the language parameter in the WhisperX call and translate the classification prompts.
- **Different grouping variable:** replace the tier assignment in your spreadsheet with your own logic — party affiliation, country, gender, time period.
- **Different theoretical framework:** the prompts in `04_llm_classify.py` are the only framework-specific part. Rewrite them for any binary coding scheme and the rest works unchanged.
- **Larger corpus:** the checkpoint system lets you run the classifier overnight across hundreds of videos without supervision.
