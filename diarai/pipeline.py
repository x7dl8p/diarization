"""Offline (whole-file) transcription + speaker diarization.

1. DiariZen (WavLM + powerset EEND + VBx) finds who speaks when; every speaker turn is then re-embedded
   and re-assigned to its nearest speaker centroid (refine).
2. Language ID over several speech windows (restricted to config.LANGUAGES).
3. Each speaker's turns are merged into <= 25 s units and decoded on their own with the language's
   Whisper (IndicWhisper large for Hindi, large-v3 otherwise), so words can't cross speaker changes and
   long-form Whisper can't drop or hallucinate sentences.
"""

import itertools
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
import torch

from diarai import config
from diarai.audio import load_audio
from diarai.config import SAMPLE_RATE
from diarai.models import BATCH, Models, get_models

log = logging.getLogger("diarai")


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float
    speaker: Optional[str] = None


@dataclass
class Utterance:
    speaker: str
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)


@dataclass
class Turn:
    speaker: str
    start: float
    end: float


@dataclass
class Result:
    utterances: List[Utterance]
    turns: List[Turn]
    language: Optional[str]
    duration: float
    timings: dict
    language_probability: Optional[float] = None

    def to_dict(self):
        return asdict(self)

    def to_sarvam(self, request_id: Optional[str] = None, with_timestamps: bool = True,
                  with_diarization: bool = True) -> dict:
        from diarai.formats import sarvam_response

        return sarvam_response(self, request_id or uuid.uuid4().hex, with_timestamps, with_diarization)

    def to_rttm(self, uri: str) -> str:
        return "".join(
            f"SPEAKER {uri} 1 {t.start:.3f} {t.end - t.start:.3f} <NA> <NA> {t.speaker} <NA> <NA>\n"
            for t in self.turns
        )

    def to_srt(self) -> str:
        def ts(x):
            ms = int(round(x * 1000))
            return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

        return "\n".join(
            f"{i}\n{ts(u.start)} --> {ts(u.end)}\n[{u.speaker}] {u.text}\n"
            for i, u in enumerate(self.utterances, 1)
        )

    def to_text(self) -> str:
        return "\n".join(f"[{u.start:8.2f} - {u.end:8.2f}] {u.speaker}: {u.text}" for u in self.utterances)


def decode(model, audio, max_segments=None, **kw):
    """Run Whisper and materialise the segments. On a CUDA OOM: free PyTorch's cache and retry, then retry
    greedily (beam search keeps beam_size copies of the decoder state)."""
    for attempt in range(3):
        try:
            segments, _ = model.transcribe(audio, **kw)
            return list(itertools.islice(segments, max_segments))  # lazy: errors surface while iterating
        except RuntimeError as e:
            if attempt == 2 or "out of memory" not in str(e):
                raise
            torch.cuda.empty_cache()
            if attempt == 1:
                kw = {**kw, "beam_size": 1}


def transcribe_unit(models: Models, audio: np.ndarray, language: str, offset: float, speaker: str,
                    beam_size: int = 5, max_words_per_sec: float = 6.0, prompt: Optional[str] = "default",
                    word_timestamps: bool = True) -> List[Word]:
    """Decode one single-speaker unit (< 30 s). Call inside `with models.gpu(...)`."""
    model, kw = models.whisper(language)
    if prompt == "default":
        prompt = config.PROMPTS.get(language)
    dur = len(audio) / SAMPLE_RATE
    # Bound the output to what the audio can hold (~20 tokens/s covers Devanagari's byte-level tokens).
    # On noisy phone audio Whisper can otherwise loop on a phrase until the 448-token limit, and with beam
    # search that decoder state outgrows a 4 GB GPU.
    prompt_tokens = 2 * len(prompt) if prompt else 0
    kw["max_new_tokens"] = max(16, min(440 - prompt_tokens // 2, int(24 + 20 * dur)))
    try:
        segments = decode(model, audio, language=language, beam_size=beam_size, initial_prompt=prompt,
                          word_timestamps=word_timestamps, condition_on_previous_text=False, vad_filter=False,
                          **kw,
                          # without timestamps the first segment covers the whole (< 30 s) unit; later ones
                          # come from faster-whisper re-seeking after the last aligned word and decoding the
                          # padding (hallucinated tails) - don't even decode them
                          max_segments=1 if kw.get("without_timestamps") else None)
    except RuntimeError as e:
        if "out of memory" not in str(e):
            raise
        log.warning("unit at %.1fs (%.1fs) skipped: GPU out of memory", offset, dur)
        torch.cuda.empty_cache()
        return []
    segments = [s for s in segments if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
    if word_timestamps:
        words = [Word(w.start + offset, w.end + offset, w.word.strip(), w.probability, speaker)
                 for s in segments for w in (s.words or []) if w.word.strip()]
    else:
        words = [Word(s.start + offset, s.end + offset, s.text.strip(), float(np.exp(s.avg_logprob)), speaker)
                 for s in segments if s.text.strip()]
    # a tiny overlapped turn decoded on its own often yields junk: no one says 6+ words per second
    n = sum(len(w.text.split()) for w in words)
    if n > max(1.0, max_words_per_sec * dur):
        return []
    # Whisper sometimes echoes the prompt on near-silence
    text = " ".join(w.text for w in words).strip(" ।.,")
    if prompt and len(text) >= 0.5 * len(prompt) and text in prompt:  # not short real words like "सर"
        return []
    return words


def dedupe_crosstalk(units) -> List[Word]:
    """units: [(duration, words)]. Overlapping turns of two speakers are decoded separately, so the louder
    speaker's words can appear in both; keep a duplicated word only in the longer unit."""
    words, kept = [], []
    for dur, ws in sorted(units, key=lambda u: -u[0]):
        for w in ws:
            key = w.text.strip(".,?!।").lower()
            if any(k == key and o.speaker != w.speaker and abs((o.start + o.end) - (w.start + w.end)) < 1.0
                   for k, o in kept):
                continue
            kept.append((key, w))
            words.append(w)
    return sorted(words, key=lambda w: w.start)


class DiarAI:
    def __init__(self, models: Optional[Models] = None, language: Optional[str] = None, beam_size: int = 5,
                 refine: bool = True, **model_kw):
        """models: shared Models (default: the process-wide instance, created with **model_kw)."""
        self.models = models or get_models(**model_kw)
        self.language = language
        self.beam_size = beam_size
        self.do_refine = refine

    # --------------------------------------------------------- diarization
    @torch.inference_mode()
    def diarize(self, wav: np.ndarray, num_speakers=None, min_speakers=None, max_speakers=None) -> List[Turn]:
        from pyannote.audio.utils.signal import Binarize
        from scipy.ndimage import median_filter

        p = self.models.diar
        lo, hi = self.models.default_speakers
        if num_speakers:
            lo = hi = num_speakers
        lo, hi = min_speakers or lo, max_speakers or hi

        file = {"waveform": torch.from_numpy(wav)[None], "sample_rate": SAMPLE_RATE}
        # Mirrors DiariZenPipeline.__call__, but on an in-memory waveform shared with ASR.
        seg = p.get_segmentations(file, soft=False)
        if p.apply_median_filtering:
            seg.data = median_filter(seg.data, size=(1, 11, 1), mode="reflect")
        if not seg.data.any():  # no speech at all
            return []
        count = p.speaker_count(seg, p._segmentation.model._receptive_field, warm_up=(0.0, 0.0))
        emb = p.get_embeddings(file, seg, exclude_overlap=p.embedding_exclude_overlap)
        hard_clusters, _, centroids = self._cluster(emb, seg, num_speakers)
        hard_clusters = self._merge_to_max(hard_clusters, centroids, hi)
        count.data = np.minimum(count.data, hi).astype(np.int8)
        hard_clusters[np.sum(seg.data, axis=1) == 0] = -2
        discrete, _ = p.reconstruct(seg, hard_clusters, count)
        ann = Binarize(onset=0.5, offset=0.5, min_duration_on=0.0, min_duration_off=0.0)(discrete)
        turns = [
            Turn(f"SPEAKER_{int(spk):02d}", round(s.start, 3), round(s.end, 3))
            for s, _, spk in ann.itertracks(yield_label=True)
        ]
        return self.refine(wav, turns) if self.do_refine else turns

    @torch.inference_mode()
    def refine(self, wav: np.ndarray, turns: List[Turn], min_dur: float = 1.0, max_dur: float = 20.0,
               max_gap: float = 1.0, iters: int = 3, margin: float = 0.0) -> List[Turn]:
        """Turn-level speaker re-assignment.

        VBx links speakers across 16 s chunks from short, noisy chunk embeddings, so short turns of
        a similar-sounding speaker occasionally land in the wrong cluster. Here every speaker turn
        (overlapped speech removed) gets its own embedding, speaker centroids are re-estimated from
        those, and each turn moves to its nearest centroid. Turns shorter than `min_dur` keep their label.
        """
        from pyannote.core import Annotation, Segment

        labels = sorted({t.speaker for t in turns})
        if len(labels) < 2:
            return turns
        ann = Annotation()
        for i, t in enumerate(turns):
            ann[Segment(t.start, t.end), i] = t.speaker
        overlap = ann.get_overlap()

        # group consecutive same-label segments (gap < max_gap) into speaker turns
        groups: List[List[int]] = []
        for i in sorted(range(len(turns)), key=lambda i: turns[i].start):
            prev = turns[groups[-1][-1]] if groups else None
            if prev and prev.speaker == turns[i].speaker and turns[i].start - prev.end < max_gap:
                groups[-1].append(i)
            else:
                groups.append([i])

        idx, embs, durs = [], [], []
        for gi, g in enumerate(groups):
            clean = Annotation()
            for i in g:
                clean[Segment(turns[i].start, turns[i].end), i] = "x"
            clean = clean.get_timeline().support().extrude(overlap)
            dur = clean.duration()
            if dur < min_dur:
                continue
            chunks, total = [], 0.0
            for s in clean:
                take = min(s.duration, max_dur - total)
                chunks.append(wav[int(s.start * SAMPLE_RATE): int((s.start + take) * SAMPLE_RATE)])
                total += take
                if total >= max_dur:
                    break
            e = self.models.embed(np.concatenate(chunks), min_seconds=0.0)
            if e is not None:
                idx.append(gi)
                embs.append(e)
                durs.append(min(dur, max_dur))
        if len(idx) < 2:
            return turns
        embs, durs = np.stack(embs), np.array(durs)
        assign = np.array([labels.index(turns[groups[gi][0]].speaker) for gi in idx])
        assign, _ = self._reassign(embs, durs, assign, len(labels), iters, margin)
        out = list(turns)
        for gi, k in zip(idx, assign):
            for i in groups[gi]:
                out[i] = Turn(labels[k], turns[i].start, turns[i].end)
        return out

    @staticmethod
    def _reassign(embs, durs, assign, n_labels, iters, margin):
        """Alternate centroid estimation and nearest-centroid reassignment of turn embeddings."""

        def centroids(assign):
            cents = np.zeros((n_labels, embs.shape[1]))
            for k in range(n_labels):
                m = assign == k
                if m.any():
                    c = (embs[m] * durs[m, None]).sum(0)
                    cents[k] = c / (np.linalg.norm(c) + 1e-9)
            return cents

        for _ in range(iters):
            sim = embs @ centroids(assign).T
            cur = sim[np.arange(len(assign)), assign]
            new = np.where(sim.max(1) - cur > margin, sim.argmax(1), assign)
            if np.array_equal(new, assign):
                break
            assign = new
        return assign, centroids(assign)

    def _cluster(self, emb, seg, num_speakers=None):
        """DiariZen's VBx ignores min/max cluster hints; with a known speaker count, seed its AHC with it."""
        c = self.models.diar.clustering
        if not num_speakers:
            return c(embeddings=emb, segmentations=seg)
        saved = c.ahc_criterion, c.ahc_threshold
        c.ahc_criterion, c.ahc_threshold = "maxclust", num_speakers
        try:
            return c(embeddings=emb, segmentations=seg)
        finally:
            c.ahc_criterion, c.ahc_threshold = saved

    @staticmethod
    def _merge_to_max(hard_clusters, centroids, max_speakers):
        """Merge the most similar clusters (cosine) until at most max_speakers remain."""
        labels = sorted(set(np.unique(hard_clusters)) - {-2})
        if max_speakers is None or len(labels) <= max_speakers or len(centroids) < len(labels):
            return hard_clusters
        cent = {k: centroids[k] / np.linalg.norm(centroids[k]) for k in labels}
        weight = {k: int(np.sum(hard_clusters == k)) for k in labels}
        while len(cent) > max_speakers:
            keys = list(cent)
            a, b = max(((a, b) for i, a in enumerate(keys) for b in keys[i + 1:]),
                       key=lambda ab: float(cent[ab[0]] @ cent[ab[1]]))
            merged = weight[a] * cent[a] + weight[b] * cent[b]
            cent[a], weight[a] = merged / np.linalg.norm(merged), weight[a] + weight[b]
            del cent[b]
            hard_clusters[hard_clusters == b] = a
        out = hard_clusters.copy()  # renumber 0..K-1
        for new, old in enumerate(sorted(set(np.unique(hard_clusters)) - {-2})):
            out[hard_clusters == old] = new
        return out

    # ------------------------------------------------------------------ ASR
    @staticmethod
    def speaker_segments(turns: List[Turn], max_gap: float = 0.6, max_len: float = 25.0):
        """Merge each speaker's diarization turns across short pauses into decoding units of <= max_len s,
        splitting long monologues at their pauses."""
        units = []
        for spk in sorted({t.speaker for t in turns}):
            cur = None
            for t in sorted((t for t in turns if t.speaker == spk), key=lambda t: t.start):
                if cur and t.start - cur[2] <= max_gap and t.end - cur[1] <= max_len:
                    cur[2] = max(cur[2], t.end)
                else:
                    cur = [spk, t.start, t.end]
                    units.append(cur)
        out = []
        for spk, a, b in units:  # a single turn longer than max_len: cut into equal pieces
            n = int(np.ceil((b - a) / max_len))
            out += [(spk, a + i * (b - a) / n, a + (i + 1) * (b - a) / n) for i in range(n)]
        return sorted(out, key=lambda u: u[1])

    def transcribe_turns(self, wav, turns: List[Turn], language: str, pad: float = 0.2,
                         priority: int = BATCH) -> List[Word]:
        units = []
        for spk, a, b in self.speaker_segments(turns):
            i, j = max(0, int((a - pad) * SAMPLE_RATE)), min(len(wav), int((b + pad) * SAMPLE_RATE))
            with self.models.gpu(priority):  # per unit, so live streams can cut in between
                ws = transcribe_unit(self.models, wav[i:j], language, i / SAMPLE_RATE, spk, self.beam_size)
            units.append((b - a, ws))

        return dedupe_crosstalk(units)

    @staticmethod
    def group(words: List[Word], max_gap: float = 1.5, max_len: float = 20.0,
              min_interruption: float = 1.0) -> List[Utterance]:
        """Words -> utterances, per speaker. A speaker's utterance continues across words of others that
        overlap it or total < min_interruption s (backchannels, cross-talk), so overlapped speech reads as
        two overlapping utterances instead of word-by-word ping-pong. Past max_len s an utterance is split
        at a sentence end or pause. Utterances are returned in start-time order."""
        utts: List[Utterance] = []
        open_: dict = {}  # speaker -> current utterance
        others: dict = {}  # speaker -> seconds others spoke after that speaker's last word
        for w in words:
            for spk, u in open_.items():
                if spk != w.speaker and w.start >= u.end:
                    others[spk] = others.get(spk, 0.0) + (w.end - w.start)
            u = open_.get(w.speaker)
            long = u is not None and (w.end - u.start > 2 * max_len or (
                u.end - u.start >= max_len and (u.words[-1].text[-1:] in ".?!।" or w.start - u.end >= 0.3)))
            if u and w.start - u.end <= max_gap and others.get(w.speaker, 0.0) < min_interruption and not long:
                u.words.append(w)
                u.end = max(u.end, w.end)
            else:
                u = Utterance(w.speaker, w.start, w.end, "", [w])
                utts.append(u)
                open_[w.speaker] = u
            others[w.speaker] = 0.0
        for u in utts:
            u.text = " ".join(w.text for w in u.words)
        return sorted(utts, key=lambda u: u.start)

    # ---------------------------------------------------------------- main
    def __call__(self, audio, num_speakers=None, min_speakers=None, max_speakers=None,
                 language: Optional[str] = None, priority: int = BATCH) -> Result:
        t0 = time.perf_counter()
        wav = load_audio(audio)
        timings = {"load": round(time.perf_counter() - t0, 3)}

        def timed(name, fn, *a, **kw):
            s = time.perf_counter()
            out = fn(*a, **kw)
            timings[name] = round(time.perf_counter() - s, 3)
            return out

        with self.models.gpu(priority):
            turns = timed("diarization", self.diarize, wav, num_speakers, min_speakers, max_speakers)
            torch.cuda.empty_cache()  # give the activation memory back before Whisper runs
        language, lang_p = language or self.language, None
        if language is None:
            with self.models.gpu(priority):  # vote over several 30 s speech windows
                language, lang_p = timed("language_id", self.models.detect_language, wav, segments=4,
                                         vad_filter=True)
        words = timed("asr", self.transcribe_turns, wav, turns, language, priority=priority)
        utts = self.group(words)
        duration = len(wav) / SAMPLE_RATE
        timings["total"] = round(time.perf_counter() - t0, 3)
        timings["rtf"] = round(timings["total"] / max(duration, 1e-6), 4)
        return Result(utts, turns, language, round(duration, 3), timings,
                      None if lang_p is None else round(float(lang_p), 4))
