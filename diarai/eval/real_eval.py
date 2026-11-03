"""Compare the pipeline against Sarvam AI (saaras:v4) on the real clips, for transcription and diarization.

Sarvam is a strong reference, not ground truth, so these numbers measure *agreement*.

Usage:
    python -m diarai.eval.real_eval --out results/real/offline [--mode stream] [options]
    python -m diarai.eval.real_eval --out results/real/offline --rescore     # score saved hypotheses only
"""

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from diarai.eval import metrics

_HI_PUNCT = re.compile(r"[।॥\-–—]")


def normalize(text: str, lang: str):
    if lang == "en":
        return metrics.normalize(text)
    # Hindi: Whisper's normalizers strip Devanagari vowel signs (Unicode category M), so do it by hand
    t = unicodedata.normalize("NFC", text).lower()
    t = _HI_PUNCT.sub(" ", t)
    t = "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in t)
    t = t.replace("़", "").replace("ँ", "ं").replace("‌", "").replace("‍", "")
    return t.split()


def load_ref(path: str):
    d = json.loads(Path(path).read_text())
    return [{"speaker": f"S{e['speaker_id']}", "start": e["start_time_seconds"], "end": e["end_time_seconds"],
             "text": e["transcript"]} for e in d["diarized_transcript"]["entries"]]


def per_speaker(segments, lang):
    by, flat = defaultdict(list), []
    for s in sorted(segments, key=lambda s: s["start"]):
        w = normalize(s["text"], lang)
        by[s["speaker"]] += w
        flat += w
    return flat, dict(by)


def chars(words):
    return list("".join(words))


def score(item, result, out_dir: Path):
    lang = item["language"]
    ref_segs = load_ref(item["sarvam"])
    hyp_segs = [{"speaker": u["speaker"], "start": u["start"], "end": u["end"], "text": u["text"]}
                for u in result["utterances"]]
    r_all, r_spk = per_speaker(ref_segs, lang)
    h_all, h_spk = per_speaker(hyp_segs, lang)
    rttm = out_dir / f"{item['uri']}.ref.rttm"
    rttm.write_text("".join(f"SPEAKER {item['uri']} 1 {s['start']:.3f} {s['end'] - s['start']:.3f} <NA> <NA> "
                            f"{s['speaker']} <NA> <NA>\n" for s in ref_segs))
    we, wn = metrics.wer(r_all, h_all)
    ce, cn = metrics.wer(chars(r_all), chars(h_all))
    cpe, _, mapping = metrics.cpwer(r_spk, h_spk)
    cpce, cpcn, _ = metrics.cpwer({k: chars(v) for k, v in r_spk.items()}, {k: chars(v) for k, v in h_spk.items()})
    # Sarvam segments are utterance-level (pauses inside count as speech): score with a 0.25 s collar
    # and with our same-speaker gaps < 1 s filled to match that convention.
    d = metrics.der(str(rttm), result["turns"], item["uri"], collar=0.25, fill_gaps=1.0)
    return {
        "uri": item["uri"], "language": lang, "detected_language": result.get("language"),
        "duration": result["duration"], "ref_words": wn, "ref_chars": cn,
        "wer": we / max(wn, 1), "cer": ce / max(cn, 1), "cpwer": cpe / max(wn, 1), "cpcer": cpce / max(cpcn, 1),
        "der": d["diarization error rate"],
        "der_parts": {k: d[k] / max(d["total"], 1e-9) for k in ("missed detection", "false alarm", "confusion")},
        "ref_speakers": len(r_spk), "hyp_speakers": len(h_spk), "speaker_map": mapping,
        "seconds": result["timings"]["total"],
    }


def run_stream(models, item, args):
    """Feed the clip through a StreamSession in 100 ms chunks, as a live client would."""
    import time

    from diarai.audio import load_audio
    from diarai.stream import StreamSession

    wav = load_audio(item["audio"])
    sess = StreamSession(models, language=None if args.auto_language else item["language"],
                         end_silence=args.end_silence, speaker_threshold=args.speaker_threshold, beam_size=args.beam_size,
                         max_utterance=args.max_utterance)
    lat, t0 = [], time.perf_counter()
    step = int(0.1 * 16000)
    for i in range(0, len(wav) + step, step):
        events = sess.feed(wav[i:i + step]) if i < len(wav) else sess.flush()
        for ev in events:
            if ev.kind == "segment":
                s = time.perf_counter()
                sess.process(ev)
                lat.append(time.perf_counter() - s)
    total = time.perf_counter() - t0
    res = sess.live_result().to_dict()
    res["turns"] = [{"speaker": u["speaker"], "start": u["start"], "end": u["end"]} for u in res["utterances"]]
    res["timings"] = {"total": total, "utterance_latency": lat}
    return res, sess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/real/manifest.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--auto-language", action="store_true", help="detect the language instead of using the manifest's")
    ap.add_argument("--mode", choices=["offline", "stream"], default="offline")
    ap.add_argument("--asr-model", default=None, help="default Whisper (default: config.ASR_MODEL)")
    ap.add_argument("--lang-model", action="append", default=None, metavar="LANG=DIR",
                    help="per-language Whisper (default: config.LANG_MODELS); 'none' disables them")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--no-prompt", action="store_true", help="disable the per-language style prompts")
    ap.add_argument("--only-language", default=None, help="score only clips of this language (e.g. hi)")
    ap.add_argument("--end-silence", type=float, default=0.6, help="stream mode: endpointing silence (s)")
    ap.add_argument("--speaker-threshold", type=float, default=0.3, help="stream mode: new-speaker cosine threshold")
    ap.add_argument("--max-utterance", type=float, default=10.0, help="stream mode: longest segment (s)")
    args = ap.parse_args()

    items = [json.loads(l) for l in args.manifest.read_text().splitlines()]
    items = [i for i in items if args.only_language in (None, i["language"])]
    args.out.mkdir(parents=True, exist_ok=True)
    models, rows = None, []
    for item in items:
        hyp_path = args.out / f"{item['uri']}.json"
        if args.rescore and hyp_path.exists():
            result = json.loads(hyp_path.read_text())
        else:
            if models is None:
                from diarai import config
                from diarai.models import Models

                kw = {}
                if args.asr_model:
                    kw["asr_model"] = args.asr_model
                if args.lang_model is not None:
                    kw["lang_models"] = {} if args.lang_model == ["none"] else dict(
                        x.split("=", 1) for x in args.lang_model)
                if args.no_prompt:
                    config.PROMPTS.clear()
                models = Models(**kw)
            if args.mode == "stream":
                result, _ = run_stream(models, item, args)
            else:
                from diarai.pipeline import DiarAI

                res = DiarAI(models, beam_size=args.beam_size, refine=not args.no_refine)(
                    item["audio"], language=None if args.auto_language else item["language"])
                result = res.to_dict()
            hyp_path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        rows.append(score(item, result, args.out))
        r = rows[-1]
        lat = result["timings"].get("utterance_latency")
        extra = f" utt-latency p50={np.median(lat):.2f}s p95={np.percentile(lat, 95):.2f}s" if lat else ""
        print(f"{r['uri']:11s} lang={r['detected_language']} WER={r['wer']:.3f} CER={r['cer']:.3f} "
              f"cpWER={r['cpwer']:.3f} cpCER={r['cpcer']:.3f} DER={r['der']:.3f} "
              f"spk={r['hyp_speakers']}/{r['ref_speakers']} {r['seconds']:.1f}s{extra}", flush=True)
    pooled = pool(rows)
    (args.out / "scores.json").write_text(json.dumps({"pooled": pooled, "files": rows}, indent=1, ensure_ascii=False))
    print("POOLED", {k: round(v, 4) for k, v in pooled.items()})


def pool(rows):
    weight = {"wer": "ref_words", "cpwer": "ref_words", "cer": "ref_chars", "cpcer": "ref_chars"}
    pooled = {k: float(np.average([r[k] for r in rows], weights=[r[w] for r in rows])) for k, w in weight.items()}
    pooled["der"] = float(np.average([r["der"] for r in rows], weights=[r["duration"] for r in rows]))
    return pooled


if __name__ == "__main__":
    main()
