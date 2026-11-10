"""Paths and defaults. Every setting can be overridden with an environment variable (or the repo's .env)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000


def _load_dotenv():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

MODELS_DIR = Path(os.environ.get("DIARAI_MODELS_DIR", ROOT / "models"))
BENCH_DIR = Path(os.environ.get("DIARAI_BENCH_DIR", ROOT / "bench"))  # zdial comparison results + audio
MAYA_URL = os.environ.get("DIARAI_MAYA_URL", "http://127.0.0.1:8086")  # maya-transcribe service (diarai/maya_server.py)
DEVICE = os.environ.get("DIARAI_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("DIARAI_COMPUTE_TYPE", "int8_float16")

# Default Whisper: transcribes every language without a dedicated model, and does language ID.
ASR_MODEL = os.environ.get("DIARAI_ASR_MODEL", "large-v3")
# Dedicated per-language CTranslate2 Whisper models (fine-tuned without timestamp tokens).
# IndicWhisper large (AI4Bharat Vistaar, MIT): Hindi cpWER vs Sarvam 17.6% against 24.6% for large-v3.
LANG_MODELS = {
    lang: path
    for lang, path in {"hi": Path(os.environ.get("DIARAI_HI_MODEL", MODELS_DIR / "indicwhisper-large-hi-ct2"))}.items()
    if path.exists()
}
DIAR_MODEL = os.environ.get("DIARAI_DIAR_MODEL", "BUT-FIT/diarizen-wavlm-large-s80-md")

# Languages language-ID may choose from ("all" = any Whisper language). Hindi speech is often scored as
# Urdu by Whisper; restricting to the languages you serve avoids that.
_langs = os.environ.get("DIARAI_LANGUAGES", "en,hi").strip()
LANGUAGES = None if _langs in ("", "all") else [x.strip() for x in _langs.split(",") if x.strip()]

# Whisper initial_prompt per language: steers transcription style. Hindi: English loanwords and numbers
# in Devanagari, the usual convention for Hindi transcripts. Generic on purpose (no test-clip words).
PROMPTS = {"hi": "हाँ सर, मैंने कंप्यूटर साइंस में पीएचडी की है और अब एआई टेक्नोलॉजी पर काम करता हूँ। पाँच सौ एम एल।"}

# Server
API_KEY_HASHES = [h for h in os.environ.get("DIARAI_API_KEY_HASH", "").split(",") if h.strip()]
MAX_UPLOAD_MB = int(os.environ.get("DIARAI_MAX_UPLOAD_MB", "1024"))
