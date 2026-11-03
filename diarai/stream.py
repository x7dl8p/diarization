"""Live (streaming) transcription + speaker attribution.

Audio arrives in small chunks. A streaming Silero VAD + endpointer cuts it into utterances (end of speech =
`end_silence` s of non-speech; utterances longer than `max_utterance` s are cut at the quietest point).
Each finished utterance is then, on the GPU:
  * language-identified (until the session language is locked),
  * transcribed with that language's Whisper (IndicWhisper for Hindi, large-v3 otherwise),
  * attributed to a speaker by its WeSpeaker embedding against running speaker centroids.
Optionally, partial transcripts of the utterance in progress are produced while it is being spoken,
and at the end of the session the whole recording can be re-processed by the offline pipeline
(DiariZen + per-turn ASR) for the most accurate final transcript.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from diarai.config import LANGUAGES, SAMPLE_RATE
from diarai.models import PARTIAL, STREAM, BATCH, Models
from diarai.pipeline import DiarAI, Result, Turn, Utterance, dedupe_crosstalk, transcribe_unit

FRAME = 512  # Silero VAD window at 16 kHz (32 ms)
HOP = FRAME / SAMPLE_RATE


class StreamingVAD:
    """Silero VAD v5 (faster-whisper's ONNX export), run incrementally with carried state."""

    CTX = 64

    def __init__(self):
        from faster_whisper.vad import get_vad_model

        m = get_vad_model()
        self.enc, self.dec = m.encoder_session, m.decoder_session
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.ctx = np.zeros(self.CTX, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Speech probability for every complete 32 ms frame of audio received so far."""
        buf = np.concatenate([self.pending, x.astype(np.float32)])
        n = len(buf) // FRAME
        self.pending = buf[n * FRAME:]
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        frames = buf[: n * FRAME].reshape(n, FRAME)
        ctx = np.concatenate([self.ctx[None], frames[:-1, -self.CTX:]], 0)
        self.ctx = frames[-1, -self.CTX:].copy()
        enc = self.enc.run(None, {"input": np.concatenate([ctx, frames], 1)})[0].reshape(n, -1)
        probs = np.empty(n, dtype=np.float32)
        for i in range(n):
            out, self.state = self.dec.run(None, {"input": enc[i: i + 1], "state": self.state})
            probs[i] = float(np.asarray(out).reshape(-1)[0])
        return probs


@dataclass
class Event:
    kind: str  # "start" | "end" | "segment"
    start: float
    end: float = 0.0


class Endpointer:
    """Turns per-frame speech probabilities into speech start/end events and utterance segments."""

    def __init__(self, threshold: float = 0.5, neg_threshold: float = 0.35, min_speech: float = 0.1,
                 end_silence: float = 0.6, max_utterance: float = 20.0, pre_pad: float = 0.3,
                 post_pad: float = 0.25):
        self.thr, self.neg = threshold, neg_threshold
        self.min_frames = max(1, round(min_speech / HOP))
        self.end_frames = max(1, round(end_silence / HOP))
        self.max_frames = round(max_utterance / HOP)
        self.pre_pad, self.post_pad = pre_pad, post_pad
        self.f = 0  # frames seen
        self.in_speech, self.run = False, 0
        self.start_f = self.last_f = 0
        self.probs: List[float] = []  # since start_f
        self.committed = 0.0  # seconds; segments never overlap

    def _segment(self, end_f: int, pad: float) -> Event:
        a = max(self.committed, self.start_f * HOP - self.pre_pad)
        b = min(self.f * HOP, (end_f + 1) * HOP + pad)
        self.committed = b
        return Event("segment", a, b)

    def push(self, probs: np.ndarray) -> List[Event]:
        out = []
        for p in probs:
            f = self.f
            self.f += 1
            if not self.in_speech:
                self.run = self.run + 1 if p >= self.thr else 0
                if self.run >= self.min_frames:
                    self.in_speech = True
                    self.start_f = self.last_f = f - self.run + 1
                    self.probs = [self.thr] * self.run
                    out.append(Event("start", self.start_f * HOP))
                continue
            self.probs.append(float(p))
            if p >= self.neg:
                self.last_f = f
            if f - self.last_f >= self.end_frames:
                out.append(self._segment(self.last_f, self.post_pad))
                out.append(Event("end", (self.last_f + 1) * HOP))
                self.in_speech, self.run = False, 0
            elif f - self.start_f >= self.max_frames:
                # too long: cut at the quietest frame of the last ~4 s (not the very last 0.5 s)
                lo = max(0, len(self.probs) - round(4.0 / HOP))
                hi = max(lo + 1, len(self.probs) - round(0.5 / HOP))
                c = self.start_f + lo + int(np.argmin(self.probs[lo:hi]))
                out.append(self._segment(c, 0.0))
                self.probs = self.probs[c + 1 - self.start_f:]
                self.start_f = c + 1
        return out

    def flush(self) -> List[Event]:
        """Force-finish the utterance in progress (client flush / end of stream)."""
        if not self.in_speech:
            return []
        self.in_speech, self.run = False, 0
        return [self._segment(self.last_f, self.post_pad), Event("end", (self.last_f + 1) * HOP)]

    def open_region(self) -> Optional[Tuple[float, float]]:
        if not self.in_speech:
            return None
        return max(self.committed, self.start_f * HOP - self.pre_pad), self.f * HOP


class OnlineSpeakers:
    """Running speaker centroids (duration-weighted sums of L2-normalised embeddings)."""

    def __init__(self, threshold: float = 0.3, min_new_dur: float = 2.0, max_weight: float = 10.0):
        self.threshold, self.min_new_dur, self.max_weight = threshold, min_new_dur, max_weight
        self.sums: List[np.ndarray] = []

    def __len__(self):
        return len(self.sums)

    def assign(self, emb: Optional[np.ndarray], dur: float, exclude=()) -> int:
        """Global speaker for an embedding. A new speaker is opened only if nothing is similar enough AND
        the voice has >= min_new_dur s of clean speech (short snippets have unreliable embeddings).
        `exclude`: speakers already taken by other voices in this window - unless one of them is the only
        good match (EEND occasionally splits one voice into two local speakers)."""
        if not self.sums:
            return self._new(emb, dur)
        if emb is None:
            cand = [k for k in range(len(self.sums)) if k not in exclude]
            return cand[-1] if cand else len(self.sums) - 1
        cents = np.stack([s / (np.linalg.norm(s) + 1e-9) for s in self.sums])
        sims = cents @ emb
        free = [k for k in range(len(self.sums)) if k not in exclude]
        best_free = max(free, key=lambda k: sims[k]) if free else None
        if best_free is not None and sims[best_free] >= self.threshold:
            k = best_free
        elif sims.max() >= self.threshold:
            k = int(np.argmax(sims))  # the voice of a speaker already in this window (EEND split it)
        elif dur >= self.min_new_dur:
            return self._new(emb, dur)
        else:  # too short to trust as a new voice
            k = best_free if best_free is not None else int(np.argmax(sims))
        self.update(k, emb, dur)
        return k

    def update(self, k: int, emb: Optional[np.ndarray], dur: float):
        if emb is not None and dur >= 0.5 * self.min_new_dur:  # short snippets don't move centroids
            if not self.sums[k].any():
                self.sums[k] = min(dur, self.max_weight) * emb
            else:
                self.sums[k] = self.sums[k] + min(dur, self.max_weight) * emb

    def _new(self, emb, dur) -> int:
        self.sums.append(min(max(dur, 0.1), self.max_weight) * emb if emb is not None else np.zeros(256))
        return len(self.sums) - 1


class Timeline:
    """Which global speaker was active when (50 Hz), for speaker continuity across windows."""

    RATE = 50

    def __init__(self):
        self.acts: List[np.ndarray] = []

    def mark(self, g: int, start: float, end: float):
        while len(self.acts) <= g:
            self.acts.append(np.zeros(0, dtype=bool))
        i, j = int(start * self.RATE), int(np.ceil(end * self.RATE))
        if j > len(self.acts[g]):
            self.acts[g] = np.concatenate([self.acts[g], np.zeros(j - len(self.acts[g]) + 30 * self.RATE, bool)])
        self.acts[g][i:j] = True

    def at(self, g: int, times: np.ndarray) -> np.ndarray:
        out = np.zeros(len(times), dtype=bool)
        if g < len(self.acts):
            idx = (times * self.RATE).astype(int)
            ok = idx < len(self.acts[g])
            out[ok] = self.acts[g][idx[ok]]
        return out


def runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[(i0, i1)) index ranges where mask is True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


class AudioBuffer:
    def __init__(self, max_seconds: float):
        self.data = np.zeros(SAMPLE_RATE * 60, dtype=np.float32)
        self.n = 0
        self.max = int(max_seconds * SAMPLE_RATE)

    def append(self, x: np.ndarray):
        if self.n + len(x) > self.max:
            raise OverflowError("stream too long")
        if self.n + len(x) > len(self.data):
            self.data = np.resize(self.data, max(2 * len(self.data), self.n + len(x)))
        self.data[self.n: self.n + len(x)] = x
        self.n += len(x)

    def __getitem__(self, ab: Tuple[float, float]) -> np.ndarray:
        a, b = ab
        return self.data[max(0, int(a * SAMPLE_RATE)): min(self.n, int(b * SAMPLE_RATE))]

    @property
    def seconds(self) -> float:
        return self.n / SAMPLE_RATE

    def all(self) -> np.ndarray:
        return self.data[: self.n]


class StreamSession:
    """State of one live stream. `feed`/`flush` are cheap (CPU); `process`, `partial` and `finalize` use the
    GPU and are called from worker threads (they serialise on models.gpu)."""

    def __init__(self, models: Models, language: Optional[str] = None, beam_size: int = 5,
                 end_silence: float = 0.6, max_utterance: float = 10.0, vad_threshold: float = 0.5,
                 speaker_threshold: float = 0.3, window: float = 16.0, max_seconds: float = 4 * 3600):
        self.models = models
        self.language, self.locked = language, language is not None
        self.beam_size = beam_size
        self.vad = StreamingVAD()
        self.ep = Endpointer(threshold=vad_threshold, neg_threshold=max(0.05, vad_threshold - 0.15),
                             end_silence=end_silence, max_utterance=max_utterance)
        self.speakers = OnlineSpeakers(speaker_threshold)
        self.timeline = Timeline()
        self.window = window
        self.audio = AudioBuffer(max_seconds)
        self._lid_audio: List[np.ndarray] = []
        self._lid_tries = 0
        self.language_probability: Optional[float] = None
        self.utterances: List[Utterance] = []

    # ------------------------------------------------------------- cheap
    def feed(self, x: np.ndarray) -> List[Event]:
        self.audio.append(x)
        return self.ep.push(self.vad(x))

    def flush(self) -> List[Event]:
        return self.ep.flush()

    # --------------------------------------------------------------- GPU
    def _language(self, audio: np.ndarray) -> str:
        if self.locked:
            return self.language
        self._lid_audio.append(audio)
        ctx = np.concatenate(self._lid_audio)[-30 * SAMPLE_RATE:]
        lang, p = self.models.detect_language(ctx, allowed=LANGUAGES)
        self._lid_tries += 1
        self.language, self.language_probability = lang, round(float(p), 4)
        if p >= 0.75 or len(ctx) >= 10 * SAMPLE_RATE or self._lid_tries >= 3:
            self.locked, self._lid_audio = True, []
        return lang

    def process(self, seg: Event, priority: int = STREAM) -> List[Utterance]:
        """Diarize + transcribe one finished segment. Returns its utterances (possibly several speakers).

        DiariZen's EEND runs on the last `window` s (segment + already-labelled context), which gives
        speaker changes and overlaps inside the segment. Local speakers are linked to global ones by their
        overlap with the labelled context, otherwise by WeSpeaker embedding against running centroids.
        Each speaker's part is then transcribed on its own (as in the offline pipeline).
        """
        a, b = seg.start, seg.end
        if b - a < 0.2:
            return []
        ws = max(0.0, b - self.window)
        win = self.audio[ws, b]
        sr = SAMPLE_RATE
        with self.models.gpu(priority):
            lang = self._language(self.audio[a, b])
            y, fd = self.models.segment(win)
            t = ws + (np.arange(len(y)) + 0.5) * fd
            new = t >= a
            local = [k for k in range(y.shape[1]) if y[new, k].sum() * fd >= 0.2]

            # 1) continuity: a local speaker active in the context is the global speaker labelled there
            mapping, pairs = {}, []
            for k in local:
                act = y[~new, k].sum() * fd
                for g in range(len(self.speakers)):
                    ov = (y[~new, k] & self.timeline.at(g, t[~new])).sum() * fd
                    if ov >= 0.5 and ov >= 0.5 * act:
                        pairs.append((ov, k, g))
            for _, k, g in sorted(pairs, reverse=True):
                if k not in mapping and g not in mapping.values():
                    mapping[k] = g

            # 2) embeddings (single-speaker frames only) for the others, and to keep centroids current
            solo = y.sum(1) == 1
            for k in sorted(local, key=lambda k: -y[new, k].sum()):
                m = y[:, k] & solo
                clip = [win[int(i * fd * sr): int(j * fd * sr)] for i, j in runs(m)]
                emb = self.models.embed(np.concatenate(clip)) if clip else None
                if k in mapping:
                    self.speakers.update(mapping[k], emb, (m & new).sum() * fd)
                else:
                    mapping[k] = self.speakers.assign(emb, m.sum() * fd, exclude=set(mapping.values()))

            # 3) speaker turns inside the segment
            turns = []
            for k in local:
                for i, j in runs(y[:, k] & new):
                    s_, e_ = max(a, ws + i * fd), min(b, ws + j * fd)
                    if e_ - s_ >= 0.1:
                        turns.append(Turn(f"SPEAKER_{mapping[k]:02d}", s_, e_))
                        self.timeline.mark(mapping[k], s_, e_)
            if not turns:  # EEND heard nobody, VAD did: one turn, speaker by embedding
                g = self.speakers.assign(self.models.embed(self.audio[a, b]), b - a)
                turns = [Turn(f"SPEAKER_{g:02d}", a, b)]
                self.timeline.mark(g, a, b)

            # 4) ASR per speaker unit, never reaching outside the segment (no duplicated words)
            units = []
            for spk, s_, e_ in DiarAI.speaker_segments(turns):
                ua, ub = max(a, s_ - 0.2), min(b, e_ + 0.2)
                units.append((e_ - s_, transcribe_unit(self.models, self.audio[ua, ub], lang, ua, spk,
                                                       self.beam_size)))
        utts = DiarAI.group(dedupe_crosstalk(units))
        self.utterances += utts
        return utts

    def partial(self) -> Optional[Tuple[float, str]]:
        """Greedy transcript of the utterance still being spoken (for live captions)."""
        region = self.ep.open_region()
        if not self.locked or region is None or region[1] - region[0] < 0.8:
            return None  # no guessing the language from a fragment (and no model swaps for it)
        a, b = max(region[0], region[1] - 10.0), region[1]
        audio = self.audio[a, b]
        with self.models.gpu(PARTIAL):
            words = transcribe_unit(self.models, audio, self.language, a, "", beam_size=1, word_timestamps=False)
        return (a, " ".join(w.text for w in words)) if words else None

    def finalize(self, num_speakers: Optional[int] = None) -> Result:
        """Re-process the whole session with the offline pipeline (most accurate transcript + speakers)."""
        return DiarAI(self.models)(self.audio.all(), num_speakers=num_speakers,
                                   language=self.language if self.locked else None, priority=BATCH)

    def live_result(self) -> Result:
        return Result(list(self.utterances), [], self.language, round(self.audio.seconds, 3), {},
                      self.language_probability)
