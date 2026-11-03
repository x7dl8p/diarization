"""Sarvam-AI-compatible response shapes (speech-to-text REST + streaming WebSocket).

REST (`POST /speech-to-text`), same keys as Sarvam saaras:
    {"request_id", "transcript", "timestamps": {"words", "start_time_seconds", "end_time_seconds"} | null,
     "diarized_transcript": {"entries": [{"transcript", "start_time_seconds", "end_time_seconds",
                                          "speaker_id"}]} | null,
     "language_code", "language_probability", "metrics": {"audio_duration", "processing_latency"}}
`timestamps` here is word-level (Sarvam returns phrase chunks under the same keys).
Speaker ids are "0", "1", ... in order of first appearance.
"""

from typing import Dict, Iterable, Optional

# Whisper language code -> BCP-47 code as Sarvam uses it
_INDIC = {"hi", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "as", "ur", "ne", "sa", "sd"}


def lang_code(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    if lang == "or":
        return "od-IN"
    return f"{lang}-IN" if lang in _INDIC or lang == "en" else lang


def parse_lang(code: Optional[str]) -> Optional[str]:
    """'hi-IN' -> 'hi'; 'unknown' / '' / None -> None (auto-detect)."""
    if not code or code.lower() in ("unknown", "auto"):
        return None
    code = code.split("-")[0].lower()
    return "or" if code == "od" else code


class SpeakerIds:
    """Maps internal speaker labels to "0", "1", ... in order of first appearance."""

    def __init__(self):
        self.ids: Dict[str, str] = {}

    def __call__(self, label: str) -> str:
        if label not in self.ids:
            self.ids[label] = str(len(self.ids))
        return self.ids[label]


def _r(x: float) -> float:
    return round(float(x), 2)


def entry(text: str, start: float, end: float, speaker_id: str) -> dict:
    return {"transcript": text, "start_time_seconds": _r(start), "end_time_seconds": _r(end), "speaker_id": speaker_id}


def word_timestamps(words: Iterable) -> dict:
    words = list(words)
    return {"words": [w.text for w in words], "start_time_seconds": [_r(w.start) for w in words],
            "end_time_seconds": [_r(w.end) for w in words]}


def sarvam_response(result, request_id: str, with_timestamps: bool = True, with_diarization: bool = True) -> dict:
    utts = sorted(result.utterances, key=lambda u: u.start)
    ids = SpeakerIds()
    return {
        "request_id": request_id,
        "transcript": " ".join(u.text for u in utts),
        "timestamps": word_timestamps(w for u in utts for w in u.words) if with_timestamps else None,
        "diarized_transcript": {"entries": [entry(u.text, u.start, u.end, ids(u.speaker)) for u in utts]}
        if with_diarization else None,
        "language_code": lang_code(result.language),
        "language_probability": result.language_probability,
        "metrics": {"audio_duration": _r(result.duration), "processing_latency": _r(result.timings.get("total", 0))},
    }
