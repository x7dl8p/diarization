"""Sweep DiariZen VBx clustering hyper-parameters on a dev split.

Segmentations and embeddings are computed once per file and cached; only clustering +
reconstruction is re-run per configuration, scored by DER (no collar).

Usage:
    python -m diarai.tune_clustering --manifest data/synth/manifest.jsonl --dev 10 --diar-step 0.1
"""

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter

from diarai import metrics
from diarai.pipeline import SAMPLE_RATE, DiarAI, load_audio


def cache_features(pipe, items, cache_dir: Path):
    p = pipe.diar
    feats = {}
    for item in items:
        path = cache_dir / f"{item['uri']}.pkl"
        if not path.exists():
            wav = load_audio(item["audio"])
            f = {"waveform": torch.from_numpy(wav)[None], "sample_rate": SAMPLE_RATE}
            with torch.inference_mode():
                seg = p.get_segmentations(f, soft=False)
                seg.data = median_filter(seg.data, size=(1, 11, 1), mode="reflect")
                count = p.speaker_count(seg, p._segmentation.model._receptive_field, warm_up=(0.0, 0.0))
                emb = p.get_embeddings(f, seg, exclude_overlap=True)
            path.write_bytes(pickle.dumps((seg, count, emb)))
            print("cached", item["uri"], flush=True)
        feats[item["uri"]] = pickle.loads(path.read_bytes())
    return feats


def run(p, seg, count, emb, params):
    from pyannote.audio.utils.signal import Binarize

    c = p.clustering
    c.ahc_threshold, c.Fa, c.Fb = params["ahc_threshold"], params["Fa"], params["Fb"]
    hard, _, _ = c(embeddings=emb, segmentations=seg)
    cnt = count.copy() if hasattr(count, "copy") else count
    hard[np.sum(seg.data, axis=1) == 0] = -2
    discrete, _ = p.reconstruct(seg, hard, cnt)
    ann = Binarize(onset=0.5, offset=0.5, min_duration_on=0.0, min_duration_off=0.0)(discrete)
    return [{"start": s.start, "end": s.end, "speaker": l} for s, _, l in ann.itertracks(yield_label=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--dev", type=int, default=10, help="use the first N files as dev set")
    ap.add_argument("--diar-step", type=float, default=0.1)
    ap.add_argument("--cache", type=Path, default=Path("results/tune_cache"))
    args = ap.parse_args()

    items = [json.loads(l) for l in args.manifest.read_text().splitlines()][: args.dev]
    cache = args.cache / f"step{args.diar_step}"
    cache.mkdir(parents=True, exist_ok=True)
    pipe = DiarAI(diar_step=args.diar_step, asr_model=None)
    feats = cache_features(pipe, items, cache)

    grid = {"ahc_threshold": [0.5, 0.6, 0.7, 0.8], "Fa": [0.07, 0.2, 0.4], "Fb": [0.8, 3.0, 8.0]}
    results = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        ders, conf, nspk = [], [], []
        for item in items:
            seg, count, emb = feats[item["uri"]]
            turns = run(pipe.diar, seg, count, emb, params)
            d = metrics.der(item["rttm"], turns, item["uri"])
            ders.append(d["diarization error rate"])
            conf.append(d["confusion"] / d["total"])
            nspk.append(len({t["speaker"] for t in turns}) == item["num_speakers"])
        results.append({**params, "der": float(np.mean(ders)), "der_max": float(np.max(ders)),
                        "confusion": float(np.mean(conf)), "spk_exact": float(np.mean(nspk))})
        print(json.dumps(results[-1]), flush=True)
    results.sort(key=lambda r: r["der"])
    (args.cache / f"sweep_step{args.diar_step}.json").write_text(json.dumps(results, indent=1))
    print("BEST:", results[:5])


if __name__ == "__main__":
    main()
