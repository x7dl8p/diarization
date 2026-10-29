# diarai — local GPU transcription + speaker diarization (DiariZen + Whisper)

Who said what, fully offline on one GPU. [DiariZen](https://github.com/BUTSpeechFIT/DiariZen)
(WavLM-Large-s80 EEND + VBx) and Whisper large-v3-turbo (faster-whisper/CTranslate2) run
**concurrently** on the same decoded waveform; every word is then attributed to a speaker.

```
python -m diarai meeting.wav --out out/            # -> out/meeting.{json,rttm,srt,txt}
python -m diarai call.mp3 --num-speakers 2 --language en
python -m diarai long.m4a --fast                   # ~3x faster diarization, somewhat less accurate
```

```python
from diarai import DiarAI
pipe = DiarAI(language="en")
res = pipe("meeting.wav")
for u in res.utterances:
    print(f"[{u.start:7.2f}] {u.speaker}: {u.text}")
```

## How it works

```
audio ──decode once (16 kHz)──┬──> Whisper large-v3-turbo (int8/fp16, VAD, word timestamps) ──┐
                               └──> DiariZen: WavLM EEND segmentation → WeSpeaker embeddings   ├──> word→speaker merge ──> utterances
                                    → VBx clustering → turn-level re-clustering (ours) ─────────┘
```

Both branches run in parallel threads (CTranslate2 and PyTorch release the GIL in their kernels).
On top of the stock DiariZen pipeline we added, each one checked on a dev split before being kept:

| change | why | dev effect |
|---|---|---|
| **turn-level refinement**: re-embed every speaker turn (overlap removed) with the WeSpeaker model and re-assign turns to the nearest duration-weighted speaker centroid | VBx links speakers across 16 s chunks with short, noisy embeddings, so short turns of a similar-sounding speaker land in the wrong cluster | speaker confusion 3.6% → 0.7%, worst file 10.9% → 2.9% |
| VBx AHC threshold 0.6 → 0.8 | fewer spurious extra speakers | speaker count exact 90% → 100% |
| word→speaker merge: max time overlap; on ties inside overlapped speech prefer the speaker who **just took the floor**; fix 1–2-word islands at turn boundaries | 77% of mis-attributed words were in overlaps | cpWER 6.9% → 5.9% (confirmed on held-out test: 8.0% → 6.9%) |
| sequential (not batched) Whisper decoding with Silero VAD | batched 30 s windows occasionally skip a whole sentence | WER 2.9% → 2.0%, worst file 8.3% → 4.3% |
| `num_speakers` actually honoured | DiariZen's VBx silently ignores min/max speaker hints | — |

Tried and rejected (no gain): shorter Whisper chunks (10/15/20 s), keep-previous-speaker overlap
rule (worse), merging clusters by turn-centroid similarity (no threshold separates split speakers from
genuinely different ones), coarser diarization hop (0.5) — kept only as `--fast`.

## Test data

| set | what | ground truth |
|---|---|---|
| **synth** (40 conversations, 1.8 h) | 2–4 speakers, built from LibriSpeech test-clean (one recording session per speaker), 0.1–1 s gaps, 15% of turns overlap the previous one by 0.2–1.5 s, ±4 dB speaker gain, 25–40 dB SNR noise. `synth000-009` = dev (all tuning), `synth010-039` = held-out test | exact transcripts per speaker; RTTM from wav2vec2 CTC forced alignment (words merged across pauses < 0.3 s) |
| **AMI test** (16 meetings, 9.1 h) | real 4-person meetings, Mix-Headset | official AMI manual word transcripts + the standard `only_words` RTTMs (pyannote/AMI-diarization-setup) |

Rebuild: `python -m diarai.data.synth --librispeech .../LibriSpeech/test-clean --out data/synth` and
`python -m diarai.data.ami --raw data/raw --out data/ami` (download commands below).

## Metrics

* **cpWER** – concatenated minimum-permutation WER (CHiME-6/NOTSOFAR). Punishes recognition *and* speaker attribution errors at once; **1 − cpWER = speaker-attributed transcript accuracy** — the number that says "the right words under the right speaker".
* **WER** – speaker-agnostic, Whisper English normalizer on both sides.
* **DER** – pyannote.metrics, no collar and 0.25 s collar, overlap scored.
* **p95** – the accuracy that 95% of files reach (5th percentile of per-file accuracy); for latency/RTF the 95th percentile.

## Results  (RTX 2050 laptop GPU, 4 GB, 50 W)

RESULTS_PLACEHOLDER

## Install

```bash
uv venv -p 3.10 .venv && source .venv/bin/activate
git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen third_party/DiariZen
uv pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt -c third_party/DiariZen/constraints.txt
uv pip install -e third_party/DiariZen -e third_party/DiariZen/pyannote-audio -c third_party/DiariZen/constraints.txt
# glibc >= 2.41 refuses CTranslate2 4.4's executable-stack flag:
patchelf --clear-execstack .venv/lib/python3.10/site-packages/ctranslate2.libs/libctranslate2-*.so*
pytest -q tests        # CPU-only unit tests for the merge logic + metrics
```

Version pins matter: DiariZen needs torch 2.1.1 (cuDNN 8), so ASR uses CTranslate2 4.4.0 — the last
release built against cuDNN 8. `diarai.pipeline` preloads torch's bundled cuDNN for it.

Benchmarks: `./scripts/run_benchmarks.sh`, then `python scripts/summarize.py results/synth results/ami`.
Data downloads: LibriSpeech `https://www.openslr.org/resources/12/test-clean.tar.gz`; AMI audio
`https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<MEETING>/audio/<MEETING>.Mix-Headset.wav`,
annotations `https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip`,
RTTMs/meeting lists from `github.com/pyannote/AMI-diarization-setup` (`only_words`).

## Licence note

Code: MIT (as DiariZen). The DiariZen **model weights are CC BY-NC 4.0 — non-commercial use only.**
