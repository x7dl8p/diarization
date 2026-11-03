"""Evaluation metrics.

* DER   - diarization error rate (pyannote.metrics), reported with no collar and with 0.25 s collar.
* WER   - speaker-agnostic word error rate on the whole transcript (Whisper English normalizer).
* cpWER - concatenated minimum-permutation WER (CHiME-6 / NOTSOFAR metric): each hypothesis
          speaker's words are concatenated and optimally matched to a reference speaker. It
          penalises recognition AND speaker-attribution errors at once, so 1 - cpWER is the
          "speaker-attributed transcript accuracy".
"""

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment
from whisper_normalizer.english import EnglishTextNormalizer

_norm = EnglishTextNormalizer()


def normalize(text: str):
    return _norm(text).split()


def wer(ref_words, hyp_words):
    errors = Levenshtein.distance(ref_words, hyp_words)
    return errors, len(ref_words)


def cpwer(ref_by_spk: dict, hyp_by_spk: dict):
    """Returns (errors, ref_len, mapping hyp->ref)."""
    refs, hyps = list(ref_by_spk), list(hyp_by_spk)
    n = max(len(refs), len(hyps))
    r_words = [ref_by_spk[k] for k in refs] + [[]] * (n - len(refs))
    h_words = [hyp_by_spk[k] for k in hyps] + [[]] * (n - len(hyps))
    cost = np.array([[Levenshtein.distance(r, h) for h in h_words] for r in r_words])
    ri, hi = linear_sum_assignment(cost)
    mapping = {hyps[h]: refs[r] for r, h in zip(ri, hi) if r < len(refs) and h < len(hyps)}
    return int(cost[ri, hi].sum()), sum(len(r) for r in r_words), mapping


def der(ref_rttm: str, hyp_turns, uri: str, collar: float = 0.0, uem=None, fill_gaps: float = 0.0):
    """fill_gaps > 0 merges same-speaker hypothesis turns separated by less than that many seconds,
    to match references that count short pauses as speech (AMI `only_words` almost never leaves a
    same-speaker gap below ~0.7 s)."""
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref = Annotation(uri=uri)
    for line in open(ref_rttm):
        p = line.split()
        ref[Segment(float(p[3]), float(p[3]) + float(p[4]))] = p[7]
    hyp = Annotation(uri=uri)
    for t in hyp_turns:
        hyp[Segment(t["start"], t["end"])] = t["speaker"]
    if fill_gaps > 0:
        hyp = hyp.support(collar=fill_gaps)
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    d = metric(ref, hyp, uem=uem, detailed=True)
    return {k: d[k] for k in ("diarization error rate", "missed detection", "false alarm", "confusion", "total")}


def summarize(values, higher_is_better=True):
    v = np.asarray(values, dtype=float)
    out = {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}
    # "p95" = the level reached by 95% of files: 5th percentile for accuracies, 95th for errors/latency
    out["p95"] = float(np.percentile(v, 5 if higher_is_better else 95))
    out["frac_ge_0.95"] = float(np.mean(v >= 0.95)) if higher_is_better else None
    return out
