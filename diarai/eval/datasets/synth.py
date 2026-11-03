"""Build synthetic multi-speaker conversations from LibriSpeech test-clean.

Every conversation has exact ground truth for both tasks:
  * who spoke when (RTTM, split on intra-utterance pauses with an energy VAD)
  * what each speaker said (per-speaker transcripts, in order)

Usage:
    python -m diarai.eval.datasets.synth --librispeech data/raw/librispeech/LibriSpeech/test-clean \
        --out data/synth --num 40
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 320  # 20 ms


def load_corpus(root: Path):
    """Return {speaker_id: [(flac_path, text), ...]}, one chapter (= one recording session) per speaker.

    Mixing chapters would change a speaker's microphone/room mid-conversation, which never happens
    in a real recording and makes clustering artificially hard.
    """
    chapters = defaultdict(list)
    for trans in sorted(root.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt, text = line.split(" ", 1)
            chapters[tuple(utt.split("-")[:2])].append((trans.parent / f"{utt}.flac", text))
    corpus = {}
    for (spk, _), utts in sorted(chapters.items(), key=lambda kv: -len(kv[1])):
        corpus.setdefault(spk, utts)  # largest chapter of each speaker
    return corpus


def frame_activity(wav: np.ndarray, rel_db: float = -35.0) -> np.ndarray:
    n = len(wav) // FRAME
    rms = np.sqrt(np.mean(wav[: n * FRAME].reshape(n, FRAME) ** 2, axis=1) + 1e-12)
    return rms > rms.max() * 10 ** (rel_db / 20)


def trim(wav: np.ndarray) -> np.ndarray:
    act = np.flatnonzero(frame_activity(wav))
    if len(act) == 0:
        return wav
    return wav[max(0, act[0] - 2) * FRAME : min(len(wav), (act[-1] + 3) * FRAME)]


class Aligner:
    """CTC forced alignment (torchaudio wav2vec2 960h) -> word-level speech regions, ~20 ms resolution."""

    def __init__(self):
        import torch
        import torchaudio

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.torch, self.F = torch, torchaudio.functional
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = bundle.get_model().to(self.device).eval()
        self.vocab = {c: i for i, c in enumerate(bundle.get_labels())}

    def regions(self, wav: np.ndarray, text: str, min_pause: float = 0.3):
        torch = self.torch
        with torch.inference_mode():
            emission, _ = self.model(torch.from_numpy(wav)[None].to(self.device))
            emission = torch.log_softmax(emission, dim=-1).cpu()
        tokens = [self.vocab[c] for c in "|".join(text.split()) if c in self.vocab]
        ali, _ = self.F.forced_align(emission, torch.tensor([tokens], dtype=torch.int32), blank=0)
        ali = ali[0].tolist()
        sec = len(wav) / SR / emission.shape[1]
        # word spans = runs of non-blank, non-separator frames between "|" tokens
        words, cur, sep = [], None, self.vocab["|"]
        for f, t in enumerate(ali):
            if t == 0:
                continue
            if t == sep:
                if cur:
                    words.append(cur)
                cur = None
            elif cur is None:
                cur = [f, f + 1]
            else:
                cur[1] = f + 1
        if cur:
            words.append(cur)
        merged = []
        for s, e in words:
            if merged and (s - merged[-1][1]) * sec < min_pause:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        return [(s * sec, e * sec) for s, e in merged]


def speech_regions(wav: np.ndarray, min_pause: float = 0.3):
    """Fallback energy VAD: split an utterance into speech regions separated by pauses >= min_pause seconds."""
    act = frame_activity(wav)
    regions, start, silence = [], None, 0
    max_sil = int(min_pause * SR / FRAME)
    for i, a in enumerate(act):
        if a:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= max_sil:
                regions.append((start, i - silence + 1))
                start, silence = None, 0
    if start is not None:
        regions.append((start, len(act) - silence))
    return [(s * FRAME / SR, e * FRAME / SR) for s, e in regions if e > s]


def make_conversation(corpus, used, rng: random.Random, n_spk: int, target_dur: float, overlap_p: float, aligner=None):
    speakers = rng.sample(sorted(corpus), n_spk)
    gains = {s: 10 ** (rng.uniform(-4, 4) / 20) for s in speakers}
    turns, cursor, prev, prev_end = [], 0.0, None, 0.0
    pieces = []
    while cursor < target_dur:
        spk = rng.choice([s for s in speakers if s != prev])
        pool = [u for u in corpus[spk] if u[0] not in used]
        if not pool:
            pool = corpus[spk]
        path, text = rng.choice(pool)
        used.add(path)
        wav, sr = sf.read(path, dtype="float32")
        assert sr == SR
        wav = trim(wav)
        dur = len(wav) / SR
        if dur > 20:  # keep turns conversational
            continue
        if prev is not None and rng.random() < overlap_p:
            start = max(0.0, prev_end - rng.uniform(0.2, min(1.5, 0.5 * dur)))
        else:
            start = prev_end + (rng.uniform(0.1, 1.0) if prev is not None else 0.5)
        pieces.append((start, wav * gains[spk]))
        turns.append({"speaker": spk, "start": round(start, 3), "end": round(start + dur, 3), "text": text,
                      "regions": [(start + s, start + e) for s, e in (aligner.regions(wav, text) if aligner else speech_regions(wav))]})
        prev, prev_end, cursor = spk, start + dur, start + dur
    total = int((cursor + 0.5) * SR)
    mix = np.zeros(total, dtype=np.float32)
    for start, wav in pieces:
        s = int(start * SR)
        mix[s : s + len(wav)] += wav
    # mild background noise at 25-40 dB SNR
    snr = rng.uniform(25, 40)
    sig_p = np.mean(mix[mix != 0] ** 2)
    mix += np.random.default_rng(rng.randrange(2**32)).normal(0, np.sqrt(sig_p / 10 ** (snr / 10)), total).astype(np.float32)
    mix /= max(1.0, np.abs(mix).max() / 0.95)
    return mix, turns, speakers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlap-p", type=float, default=0.15)
    ap.add_argument("--no-align", action="store_true", help="energy VAD instead of CTC forced alignment for RTTM")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    corpus = load_corpus(args.librispeech)
    (args.out / "wav").mkdir(parents=True, exist_ok=True)
    (args.out / "ref").mkdir(parents=True, exist_ok=True)
    used, manifest = set(), []
    aligner = None if args.no_align else Aligner()
    for i in range(args.num):
        n_spk = rng.choices([2, 3, 4], weights=[0.4, 0.35, 0.25])[0]
        mix, turns, speakers = make_conversation(corpus, used, rng, n_spk, rng.uniform(90, 240), args.overlap_p, aligner)
        uri = f"synth{i:03d}"
        wav_path = args.out / "wav" / f"{uri}.wav"
        sf.write(wav_path, mix, SR)
        with open(args.out / "ref" / f"{uri}.rttm", "w") as f:
            for t in turns:
                for s, e in t["regions"]:
                    f.write(f"SPEAKER {uri} 1 {s:.3f} {e - s:.3f} <NA> <NA> {t['speaker']} <NA> <NA>\n")
        ref = {"uri": uri, "speakers": speakers,
               "segments": [{k: t[k] for k in ("speaker", "start", "end", "text")} for t in turns]}
        (args.out / "ref" / f"{uri}.json").write_text(json.dumps(ref, indent=1))
        manifest.append({"uri": uri, "audio": str(wav_path), "rttm": str(args.out / "ref" / f"{uri}.rttm"),
                         "transcript": str(args.out / "ref" / f"{uri}.json"), "num_speakers": n_spk,
                         "duration": round(len(mix) / SR, 2)})
        print(f"{uri}: {n_spk} spk, {len(mix) / SR:.0f}s, {len(turns)} turns")
    (args.out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    print(f"total audio: {sum(m['duration'] for m in manifest) / 60:.1f} min")


if __name__ == "__main__":
    main()
