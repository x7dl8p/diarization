"""Run the joint pipeline over a test manifest and score it.

Usage:
    python -m diarai.evaluate --manifest data/synth/manifest.jsonl --out results/synth
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from diarai import metrics


def ref_words(ref_json):
    ref = json.loads(Path(ref_json).read_text())
    by_spk = defaultdict(list)
    all_words = []
    for s in sorted(ref["segments"], key=lambda s: s["start"]):
        w = metrics.normalize(s["text"])
        by_spk[s["speaker"]] += w
        all_words += w
    return all_words, dict(by_spk)


def hyp_words(result):
    by_spk = defaultdict(list)
    all_words = []
    for u in result["utterances"]:
        w = metrics.normalize(u["text"])
        by_spk[u["speaker"]] += w
        all_words += w
    return all_words, dict(by_spk)


def remerge(result, smooth=True, filler=None, audio=None):
    from dataclasses import asdict

    from diarai.pipeline import DiarAI, Turn, Word

    words = sorted((Word(w["start"], w["end"], w["text"], w["probability"])
                    for u in result["utterances"] for w in u["words"]), key=lambda w: w.start)
    turns = [Turn(**t) for t in result["turns"]]
    DiarAI.assign_speakers(words, turns, smooth)
    if filler is not None:
        from diarai.pipeline import load_audio

        words = filler.fill_missing(load_audio(audio), words, turns, result.get("language"))
    return {**result, "utterances": [asdict(u) for u in DiarAI.group(words)]}


def score(item, result, fill_gaps=0.0):
    r_all, r_spk = ref_words(item["transcript"])
    h_all, h_spk = hyp_words(result)
    wer_e, wer_n = metrics.wer(r_all, h_all)
    cp_e, cp_n, _ = metrics.cpwer(r_spk, h_spk)
    d0 = metrics.der(item["rttm"], result["turns"], item["uri"], collar=0.0)
    d25 = metrics.der(item["rttm"], result["turns"], item["uri"], collar=0.25)
    dfill = metrics.der(item["rttm"], result["turns"], item["uri"], fill_gaps=fill_gaps) if fill_gaps else d0
    return {
        "uri": item["uri"], "duration": item["duration"],
        "ref_speakers": len(r_spk), "hyp_speakers": len({t["speaker"] for t in result["turns"]}),
        "wer": wer_e / max(wer_n, 1), "cpwer": cp_e / max(cp_n, 1),
        "wer_err": wer_e, "cpwer_err": cp_e, "ref_words": wer_n,
        "der": d0["diarization error rate"], "der_collar": d25["diarization error rate"],
        "der_filled": dfill["diarization error rate"],
        "der_parts": {k: d0[k] / max(d0["total"], 1e-9) for k in ("missed detection", "false alarm", "confusion")},
        **{f"t_{k}": v for k, v in result["timings"].items()},
    }


def report(rows):
    tot = lambda k: sum(r[k] for r in rows)
    acc = {
        "speaker_attributed_accuracy (1-cpWER)": [1 - r["cpwer"] for r in rows],
        "word_accuracy (1-WER)": [1 - r["wer"] for r in rows],
        "diarization_accuracy (1-DER, no collar)": [1 - r["der"] for r in rows],
        "diarization_accuracy (1-DER, 0.25s collar)": [1 - r["der_collar"] for r in rows],
        "diarization_accuracy (1-DER, ref gap convention)": [1 - r["der_filled"] for r in rows],
    }
    out = {
        "files": len(rows),
        "audio_hours": tot("duration") / 3600,
        "pooled": {
            "WER": tot("wer_err") / tot("ref_words"),
            "cpWER": tot("cpwer_err") / tot("ref_words"),
            "DER_no_collar": float(np.average([r["der"] for r in rows], weights=[r["duration"] for r in rows])),
            "DER_collar_0.25": float(np.average([r["der_collar"] for r in rows], weights=[r["duration"] for r in rows])),
            "DER_gaps_filled": float(np.average([r["der_filled"] for r in rows], weights=[r["duration"] for r in rows])),
            "speaker_count_exact": float(np.mean([r["ref_speakers"] == r["hyp_speakers"] for r in rows])),
        },
        "per_file_accuracy": {k: metrics.summarize(v) for k, v in acc.items()},
        "speed": {
            "RTF_overall": tot("t_total") / tot("duration"),
            "x_realtime": tot("duration") / tot("t_total"),
            "rtf": metrics.summarize([r["t_rtf"] for r in rows], higher_is_better=False),
            "latency_s": metrics.summarize([r["t_total"] for r in rows], higher_is_better=False),
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--der-fill-gaps", type=float, default=0.0,
                    help="also score DER after merging same-speaker hyp gaps < this (reference annotation convention)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--oracle-speakers", action="store_true", help="give the true speaker count to diarization")
    ap.add_argument("--rescore", action="store_true", help="only rescore existing hypotheses")
    ap.add_argument("--remerge", action="store_true",
                    help="with --rescore: redo the word->speaker merge on the saved words/turns with the current code")
    ap.add_argument("--asr-model", default="large-v3-turbo")
    ap.add_argument("--compute-type", default="int8_float16")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--fill-short-turns", action="store_true", help="second ASR pass over word-less diarized turns")
    ap.add_argument("--diar-step", type=float, default=0.1)
    ap.add_argument("--ahc-threshold", type=float, default=0.8)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--asr-batch-size", type=int, default=1, help="1 = sequential decoding (default), >1 = batched")
    args = ap.parse_args()

    items = [json.loads(l) for l in args.manifest.read_text().splitlines()][args.offset:][: args.limit]
    hyp_dir = args.out / "hyp"
    hyp_dir.mkdir(parents=True, exist_ok=True)

    pipe = filler = None
    rows = []
    for item in items:
        hyp_path = hyp_dir / f"{item['uri']}.json"
        if args.rescore and hyp_path.exists():
            result = json.loads(hyp_path.read_text())
            if args.remerge:
                if args.fill_short_turns and filler is None:
                    from diarai.pipeline import DiarAI

                    filler = DiarAI(diar_model=None, language="en", asr_model=args.asr_model,
                                    compute_type=args.compute_type)
                result = remerge(result, smooth=not args.no_smooth,
                                 filler=filler if args.fill_short_turns else None, audio=item["audio"])
        else:
            if pipe is None:
                from diarai.pipeline import DiarAI

                t = time.perf_counter()
                pipe = DiarAI(asr_model=args.asr_model, compute_type=args.compute_type,
                              smooth_words=not args.no_smooth, language="en", fill_short_turns=args.fill_short_turns,
                              diar_step=args.diar_step, asr_batch_size=args.asr_batch_size,
                              ahc_threshold=args.ahc_threshold, refine=not args.no_refine)
                print(f"models loaded in {time.perf_counter() - t:.1f}s", flush=True)
                pipe(np.zeros(16000 * 5, dtype=np.float32))  # silent input must not crash
                from diarai.pipeline import load_audio

                pipe(load_audio(item["audio"])[: 16000 * 30])  # warm-up (CUDA ctx, cuDNN autotune)
            res = pipe(item["audio"], num_speakers=item["num_speakers"] if args.oracle_speakers else None)
            result = res.to_dict()
            hyp_path.write_text(json.dumps(result, indent=1))
            (hyp_dir / f"{item['uri']}.rttm").write_text(res.to_rttm(item["uri"]))
        row = score(item, result, args.der_fill_gaps)
        rows.append(row)
        (args.out / "per_file.json").write_text(json.dumps(rows, indent=1))
        print(f"{row['uri']}: dur={row['duration']:.0f}s WER={row['wer']:.3f} cpWER={row['cpwer']:.3f} "
              f"DER={row['der']:.3f} DERc={row['der_collar']:.3f} spk={row['hyp_speakers']}/{row['ref_speakers']} "
              f"RTF={row['t_rtf']:.3f}", flush=True)

    summary = report(rows)
    (args.out / "per_file.json").write_text(json.dumps(rows, indent=1))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
