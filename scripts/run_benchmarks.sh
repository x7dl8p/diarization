#!/usr/bin/env bash
# Full benchmark: synthetic conversations, then AMI test (Mix-Headset).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1
python -m diarai.evaluate --manifest data/synth/manifest.jsonl --out results/synth
python -m diarai.evaluate --manifest data/ami/manifest.jsonl --out results/ami
