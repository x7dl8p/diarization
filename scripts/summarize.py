"""Pretty-print one or more results/*/summary.json files."""
import json
import sys

for path in sys.argv[1:]:
    s = json.load(open(f"{path}/summary.json"))
    print(f"=== {path}: {s['files']} files, {s['audio_hours']:.2f} h audio")
    print("  pooled: " + ", ".join(f"{k}={v:.4f}" for k, v in s["pooled"].items()))
    for k, v in s["per_file_accuracy"].items():
        print(f"  {k:45s} mean={v['mean']:.4f} median={v['median']:.4f} p95={v['p95']:.4f} "
              f"min={v['min']:.4f} files>=95%={v['frac_ge_0.95']:.0%}")
    sp = s["speed"]
    print(f"  speed: {sp['x_realtime']:.1f}x realtime, RTF p95={sp['rtf']['p95']:.3f}")
