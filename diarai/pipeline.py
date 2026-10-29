"""Joint transcription + speaker diarization on a single local GPU.

DiariZen (WavLM + powerset EEND + VBx) and faster-whisper run concurrently on the same decoded
waveform, in two threads (both release the GIL inside their GPU kernels). Each transcribed word
is then attributed to the speaker whose diarization turns overlap it the most.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import ctypes
import glob
import os

import numpy as np
import torch

SAMPLE_RATE = 16000


def _preload_cudnn():
    """CTranslate2 4.4 dlopen()s cuDNN 8 by soname; make the copy bundled with torch visible to it."""
    for lib in ("libcudnn_ops_infer.so.8", "libcudnn_cnn_infer.so.8"):
        for path in glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", lib)):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


_preload_cudnn()


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

    def to_dict(self):
        return asdict(self)

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


def load_audio(audio: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono float32 (PyAV, no ffmpeg binary needed)."""
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32)
    from faster_whisper import decode_audio

    return decode_audio(str(audio), sampling_rate=SAMPLE_RATE)


class DiarAI:
    def __init__(
        self,
        asr_model: str = "large-v3-turbo",
        diar_model: str = "BUT-FIT/diarizen-wavlm-large-s80-md",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        asr_batch_size: int = 1,
        diar_batch_size: int = 2,
        diar_step: float = 0.1,
        ahc_threshold: Optional[float] = 0.8,
        refine: bool = True,
        beam_size: int = 5,
        language: Optional[str] = None,
        smooth_words: bool = True,
        fill_short_turns: bool = False,
    ):
        from faster_whisper import BatchedInferencePipeline, WhisperModel
        from diarizen.pipelines.inference import DiariZenPipeline

        self.language = language
        self.beam_size = beam_size
        self.asr_batch_size = asr_batch_size
        self.smooth_words = smooth_words
        self.fill_short_turns = fill_short_turns

        self.asr = None
        if asr_model is not None:
            self.asr = WhisperModel(asr_model, device=device, compute_type=compute_type)
            if asr_batch_size > 1:
                self.asr = BatchedInferencePipeline(self.asr)

        self.diar = None
        self.do_refine = refine
        if diar_model is not None:
            self.diar = DiariZenPipeline.from_pretrained(diar_model)
            self.diar.segmentation_batch_size = diar_batch_size
            self.diar.embedding_batch_size = diar_batch_size
            # sliding-window hop as a fraction of the 16 s window (DiariZen default 0.1 = 10x overlap)
            self.diar._segmentation.step = diar_step * self.diar._segmentation.duration
            self._default_speakers = (self.diar.min_speakers, self.diar.max_speakers)
            if ahc_threshold is not None:  # DiariZen ships 0.6; 0.8 + turn refinement was best on our dev split
                self.diar.clustering.ahc_threshold = ahc_threshold

    # ------------------------------------------------------------------ ASR
    def transcribe(self, wav: np.ndarray):
        common = dict(language=self.language, beam_size=self.beam_size, word_timestamps=True,
                      condition_on_previous_text=False)
        if self.asr_batch_size > 1:
            segments, info = self.asr.transcribe(wav, batch_size=self.asr_batch_size, **common)
        else:
            # Sequential long-form decoding: on our dev split it never skips whole sentences the way
            # batched 30 s windows occasionally do (WER 2.0% vs 2.9%), and it is hidden behind diarization.
            segments, info = self.asr.transcribe(wav, vad_filter=True, **common)
        words = [
            Word(w.start, w.end, w.word.strip(), w.probability)
            for s in segments
            for w in (s.words or [])
            if w.word.strip()
        ]
        return words, info.language

    # --------------------------------------------------------- diarization
    @torch.inference_mode()
    def diarize(self, wav: np.ndarray, num_speakers=None, min_speakers=None, max_speakers=None) -> List[Turn]:
        from pyannote.audio.utils.signal import Binarize
        from scipy.ndimage import median_filter

        p = self.diar
        lo, hi = self._default_speakers
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
            x = torch.from_numpy(np.concatenate(chunks))[None, None]
            e = self.diar._embedding(x)[0]
            if np.all(np.isfinite(e)):
                idx.append(gi)
                embs.append(e / np.linalg.norm(e))
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
        c = self.diar.clustering
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

    def fill_missing(self, wav, words: List[Word], turns: List[Turn], language=None,
                     min_dur: float = 0.3, max_dur: float = 15.0, max_coverage: float = 0.2) -> List[Word]:
        """Second ASR pass over diarized turns that received (almost) no words.

        Long-form Whisper favours the main speaker and drops short backchannels ("yeah", "okay")
        and interjections inside overlaps. DiariZen does find those turns, so each one that has
        < max_coverage of its duration covered by words of its own speaker is decoded on its own
        and the result is attributed to that turn's speaker.
        """
        from faster_whisper import BatchedInferencePipeline

        model = self.asr.model if isinstance(self.asr, BatchedInferencePipeline) else self.asr
        added = []
        for t in turns:
            dur = t.end - t.start
            if not (min_dur <= dur <= max_dur):
                continue
            covered = sum(max(0.0, min(w.end, t.end) - max(w.start, t.start)) for w in words if w.speaker == t.speaker)
            if covered >= max_coverage * dur:
                continue
            a, b = max(0, int((t.start - 0.15) * SAMPLE_RATE)), int((t.end + 0.15) * SAMPLE_RATE)
            segments, _ = model.transcribe(wav[a:b], language=language or self.language, beam_size=self.beam_size,
                                           word_timestamps=True, condition_on_previous_text=False, vad_filter=False)
            for seg in segments:
                if seg.no_speech_prob > 0.6 or seg.avg_logprob < -1.0:
                    continue
                off = a / SAMPLE_RATE
                added += [Word(w.start + off, w.end + off, w.word.strip(), w.probability, t.speaker)
                          for w in (seg.words or []) if w.word.strip()]
        return sorted(words + added, key=lambda w: w.start)

    # --------------------------------------------------------------- merge
    @staticmethod
    def assign_speakers(words: List[Word], turns: List[Turn], smooth: bool = True, overlap_tie: float = 0.8) -> None:
        if not turns:
            for w in words:
                w.speaker = "SPEAKER_00"
            return
        starts = np.array([t.start for t in turns])
        ends = np.array([t.end for t in turns])
        labels = [t.speaker for t in turns]
        for w in words:
            ov = np.minimum(ends, w.end) - np.maximum(starts, w.start)
            score, latest = {}, {}
            for i in np.flatnonzero(ov > 0):
                score[labels[i]] = score.get(labels[i], 0.0) + ov[i]
                latest[labels[i]] = max(latest.get(labels[i], -1.0), starts[i])
            if score:
                # Overlapped speech: when several speakers cover the word about equally, Whisper has
                # usually followed the speaker who just took the floor -> prefer the most recent turn.
                best = max(score.values())
                tied = [k for k in score if score[k] >= overlap_tie * best]
                w.speaker = max(tied, key=latest.get)
            else:  # word falls in a diarization gap: take the closest turn
                mid = 0.5 * (w.start + w.end)
                dist = np.maximum(starts - mid, 0) + np.maximum(mid - ends, 0)
                w.speaker = labels[int(np.argmin(dist))]
        if smooth:
            DiarAI._smooth(words)

    @staticmethod
    def _smooth(words: List[Word], max_island: int = 2, max_island_dur: float = 0.6) -> None:
        """Relabel tiny speaker 'islands' (<= max_island words) sandwiched inside another speaker's run.

        Whisper word timestamps are typically off by ~0.1-0.3 s; at speaker changes this flips a
        word or two onto the wrong side. Real interjections are usually separated by pauses, so
        only islands that are tightly packed between the neighbouring words are merged.
        """
        i = 0
        while i < len(words):
            j = i
            while j + 1 < len(words) and words[j + 1].speaker == words[i].speaker:
                j += 1
            if 0 < i and j + 1 < len(words) and j - i + 1 <= max_island:
                left, right = words[i - 1], words[j + 1]
                if (
                    left.speaker == right.speaker
                    and words[j].end - words[i].start <= max_island_dur
                    and words[i].start - left.end < 0.3
                    and right.start - words[j].end < 0.3
                ):
                    for k in range(i, j + 1):
                        words[k].speaker = left.speaker
            i = j + 1

    @staticmethod
    def group(words: List[Word], max_gap: float = 1.5) -> List[Utterance]:
        utts: List[Utterance] = []
        for w in words:
            u = utts[-1] if utts else None
            if u and u.speaker == w.speaker and w.start - u.end <= max_gap:
                u.words.append(w)
                u.end = max(u.end, w.end)
            else:
                utts.append(Utterance(w.speaker, w.start, w.end, "", [w]))
        for u in utts:
            u.text = " ".join(w.text for w in u.words)
        return utts

    # ---------------------------------------------------------------- main
    def __call__(self, audio, num_speakers=None, min_speakers=None, max_speakers=None) -> Result:
        t0 = time.perf_counter()
        wav = load_audio(audio)
        t_load = time.perf_counter() - t0
        timings = {}

        def timed(name, fn, *a, **kw):
            s = time.perf_counter()
            out = fn(*a, **kw)
            timings[name] = round(time.perf_counter() - s, 3)
            return out

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_asr = ex.submit(timed, "asr", self.transcribe, wav)
            f_diar = ex.submit(timed, "diarization", self.diarize, wav, num_speakers, min_speakers, max_speakers)
            words, language = f_asr.result()
            turns = f_diar.result()

        s = time.perf_counter()
        self.assign_speakers(words, turns, self.smooth_words)
        if self.fill_short_turns and self.asr is not None:
            words = timed("fill", self.fill_missing, wav, words, turns, language)
        utts = self.group(words)
        timings["merge"] = round(time.perf_counter() - s, 3)
        timings["load"] = round(t_load, 3)
        timings["total"] = round(time.perf_counter() - t0, 3)
        duration = len(wav) / SAMPLE_RATE
        timings["rtf"] = round(timings["total"] / max(duration, 1e-6), 4)
        return Result(utts, turns, language, round(duration, 3), timings)
