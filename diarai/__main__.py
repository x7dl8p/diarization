"""CLI: python -m diarai audio.wav [more files...] --out out/ [--num-speakers N]"""

import argparse
import json
from pathlib import Path

from diarai.pipeline import DiarAI


def main():
    ap = argparse.ArgumentParser(description="Joint local GPU transcription + speaker diarization (DiariZen + Whisper)")
    ap.add_argument("audio", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--num-speakers", type=int)
    ap.add_argument("--min-speakers", type=int)
    ap.add_argument("--max-speakers", type=int)
    ap.add_argument("--language", default=None, help="e.g. 'en'; auto-detected when omitted")
    ap.add_argument("--asr-model", default="large-v3-turbo")
    ap.add_argument("--compute-type", default="int8_float16")
    ap.add_argument("--fast", action="store_true", help="5x coarser diarization window hop: ~3x faster, less accurate")
    args = ap.parse_args()

    pipe = DiarAI(asr_model=args.asr_model, compute_type=args.compute_type, language=args.language,
                  diar_step=0.5 if args.fast else 0.1)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.audio:
        res = pipe(path, args.num_speakers, args.min_speakers, args.max_speakers)
        stem = args.out / path.stem
        stem.with_suffix(".json").write_text(json.dumps(res.to_dict(), indent=1))
        stem.with_suffix(".rttm").write_text(res.to_rttm(path.stem))
        stem.with_suffix(".srt").write_text(res.to_srt())
        stem.with_suffix(".txt").write_text(res.to_text())
        print(res.to_text())
        print(f"\n{path.name}: {res.duration:.1f}s audio in {res.timings['total']:.1f}s "
              f"({res.duration / res.timings['total']:.1f}x realtime) -> {stem}.{{json,rttm,srt,txt}}")


if __name__ == "__main__":
    main()
