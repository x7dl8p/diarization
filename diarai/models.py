"""Process-wide model registry shared by the CLI, the batch API and live streams.

GPU budget (4 GB card): one Whisper large (int8, ~1.9 GB) + DiariZen/WeSpeaker (~0.4 GB) + activations.
The default Whisper (large-v3) and the per-language ones (IndicWhisper for Hindi) are all loaded at
start; only one is resident on the GPU, the others are parked in CPU RAM (CTranslate2 unload_model
to_cpu), so a switch costs ~1 s instead of a reload.

All GPU work goes through `Models.gpu(priority)`, a re-entrant priority lock: live-stream work (priority
0) overtakes queued batch work (priority 2) at the next unit boundary.
"""

import ctypes
import glob
import heapq
import itertools
import logging
import os
import threading
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from diarai import config

log = logging.getLogger("diarai")

STREAM, PARTIAL, BATCH = 0, 1, 2


def _preload_cudnn():
    """CTranslate2 4.4 dlopen()s cuDNN 8 by soname; make the copy bundled with torch visible to it."""
    for lib in ("libcudnn_ops_infer.so.8", "libcudnn_cnn_infer.so.8"):
        for path in glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", lib)):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


_preload_cudnn()


class PriorityLock:
    """Re-entrant lock granting access to the waiter with the lowest priority value (FIFO within one)."""

    def __init__(self):
        self._cv = threading.Condition()
        self._owner, self._depth = None, 0
        self._waiting, self._seq = [], itertools.count()

    @contextmanager
    def __call__(self, priority: int = BATCH):
        me = threading.get_ident()
        with self._cv:
            if self._owner == me:
                self._depth += 1
            else:
                ticket = (priority, next(self._seq))
                heapq.heappush(self._waiting, ticket)
                while self._owner is not None or self._waiting[0] != ticket:
                    self._cv.wait()
                heapq.heappop(self._waiting)
                self._owner, self._depth = me, 1
        try:
            yield
        finally:
            with self._cv:
                self._depth -= 1
                if self._depth == 0:
                    self._owner = None
                    self._cv.notify_all()


class Models:
    def __init__(self, asr_model: str = config.ASR_MODEL, lang_models: Optional[Dict[str, str]] = None,
                 diar_model: Optional[str] = config.DIAR_MODEL, device: str = config.DEVICE,
                 compute_type: str = config.COMPUTE_TYPE, diar_batch_size: int = 2, diar_step: float = 0.1,
                 ahc_threshold: Optional[float] = 0.8):
        from faster_whisper import WhisperModel

        self.gpu = PriorityLock()
        self.device = device
        lang_models = config.LANG_MODELS if lang_models is None else lang_models
        self.whispers = {}
        for key, path in [("default", asr_model)] + sorted(lang_models.items()):
            log.info("loading whisper %s: %s", key, path)
            self.whispers[key] = WhisperModel(str(path), device=device, compute_type=compute_type)
            if device == "cuda":
                self.whispers[key].model.unload_model(to_cpu=True)
        self.active = None
        self.lang_models = set(lang_models)
        self._activate("default")

        self.diar = None
        if diar_model:
            from diarizen.pipelines.inference import DiariZenPipeline

            log.info("loading diarization: %s", diar_model)
            d = DiariZenPipeline.from_pretrained(diar_model)
            d.segmentation_batch_size = d.embedding_batch_size = diar_batch_size
            # sliding-window hop as a fraction of the 16 s window (DiariZen default 0.1 = 10x overlap)
            d._segmentation.step = diar_step * d._segmentation.duration
            if ahc_threshold is not None:  # DiariZen ships 0.6; 0.8 + turn refinement was best on our dev split
                d.clustering.ahc_threshold = ahc_threshold
            self.default_speakers = (d.min_speakers, d.max_speakers)
            self.diar = d

    # ---------------------------------------------------------------- whisper
    def _activate(self, key):
        if key == self.active or self.device != "cuda":
            self.active = key
            return
        if self.active is not None:
            self.whispers[self.active].model.unload_model(to_cpu=True)
        self.whispers[key].model.load_model()
        self.active = key

    def whisper(self, language: Optional[str]) -> Tuple[object, dict]:
        """(WhisperModel on the GPU, extra decode kwargs) for this language. Call inside `with self.gpu()`."""
        key = language if language in self.lang_models else "default"
        self._activate(key)
        # dedicated models are fine-tuned without timestamp tokens
        return self.whispers[key], ({"without_timestamps": True} if key != "default" else {})

    def detect_language(self, audio: np.ndarray, segments: int = 1, vad_filter: bool = False,
                        allowed=config.LANGUAGES) -> Tuple[str, float]:
        """Language ID with the default Whisper, restricted to `allowed` languages (renormalised)."""
        model, _ = self.whisper(None)
        _, _, probs = model.detect_language(audio, vad_filter=vad_filter, language_detection_segments=segments)
        if allowed:
            probs = [(l, p) for l, p in probs if l in allowed] or probs
        total = sum(p for _, p in probs) or 1.0
        lang, p = max(probs, key=lambda lp: lp[1])
        return lang, p / total

    # --------------------------------------------------------------- speakers
    @torch.inference_mode()
    def segment(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """DiariZen EEND on one window (<= 16 s): (frames x local speakers) 0/1 activity, frame duration (s).
        Local speaker indices are consistent within the window only."""
        from scipy.ndimage import median_filter

        seg = self.diar._segmentation
        y = seg.infer(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None, None])[0]
        if self.diar.apply_median_filtering and len(y) > 11:
            y = median_filter(y, size=(11, 1), mode="reflect")
        return y.astype(bool), len(audio) / config.SAMPLE_RATE / max(len(y), 1)

    @torch.inference_mode()
    def embed(self, audio: np.ndarray, min_seconds: float = 0.4) -> Optional[np.ndarray]:
        """L2-normalised WeSpeaker ResNet34 embedding of a speech snippet (None if too short/invalid)."""
        if self.diar is None or len(audio) < min_seconds * config.SAMPLE_RATE:
            return None
        e = self.diar._embedding(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None, None])[0]
        if not np.all(np.isfinite(e)):
            return None
        return e / (np.linalg.norm(e) + 1e-9)


_shared: Optional[Models] = None
_shared_lock = threading.Lock()


def get_models(**kw) -> Models:
    """The process-wide Models instance (created on first use)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = Models(**kw)
        return _shared
