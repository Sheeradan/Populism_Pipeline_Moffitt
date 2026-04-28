from pathlib import Path

# API KEYS
RAPIDAPI_KEY  = "*" # use your API keys
RAPIDAPI_HOST = "*" # use your API keys
HF_TOKEN      = "*" # use your API keys

# PATHS
DATA_CSV        = Path("Data.csv")
LABELED_CSV     = Path("Labeled_Data.csv")
VIDEOS_DIR      = Path("videos")
AUDIO_DIR       = Path("audio")
TRANSCRIPTS_DIR = Path("transcripts")
PROSODY_DIR     = Path("prosody")
RESULTS_DIR     = Path("results")
MODELS_DIR      = Path("models")

# WHISPERX — RTX 3080 Ti 16 GB
WHISPER_MODEL        = "large-v2"
WHISPER_LANGUAGE     = "de"
WHISPER_DEVICE       = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_BATCH_SIZE   = 12 # batch_size=8 is safe; bump to 12 if no OOM, drop to 4 if OOM

# LLM SETTINGS (Ollama)
LLM_MODEL = "gemma3:27b"
LLM_MAX_TOKENS = 512

# AUDIO
SAMPLE_RATE = 16000
