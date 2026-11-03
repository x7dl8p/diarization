#!/usr/bin/env bash
# Download / build every model diarai uses. Idempotent: finished steps are skipped.
#   - Systran/faster-whisper-large-v3            default ASR + language ID   (~3.1 GB)
#   - BUT-FIT/diarizen-wavlm-large-s80-md        diarization (CC BY-NC 4.0)  (~0.3 GB)
#   - pyannote/wespeaker-voxceleb-resnet34-LM    speaker embeddings
#   - IndicWhisper large Hindi (AI4Bharat Vistaar, MIT) -> models/indicwhisper-large-hi-ct2 (CTranslate2 int8)
#     The 4.3 GB archive is downloaded and converted in a throw-away venv, then deleted.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
UV=${UV:-.bootstrap/bin/uv}

echo "[1/2] Hugging Face models"
HF_HUB_DISABLE_XET=1 "$PY" - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ["Systran/faster-whisper-large-v3", "BUT-FIT/diarizen-wavlm-large-s80-md",
             "pyannote/wespeaker-voxceleb-resnet34-LM"]:
    print(" ", repo, "->", snapshot_download(repo))
EOF

OUT=models/indicwhisper-large-hi-ct2
if [ -f "$OUT/model.bin" ]; then
  echo "[2/2] $OUT already built"
  exit 0
fi
echo "[2/2] IndicWhisper large (Hindi) -> $OUT"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/indicwhisper.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
for i in $(seq 1 30); do
  curl -fL -C - --retry 5 -o "$TMP/hindi_models.zip" https://indicwhisper.objectstore.e2enetworks.net/hindi_models.zip && break
  sleep 5
done
unzip -q "$TMP/hindi_models.zip" 'hindi_models/whisper-large-hi-noldcil/*' -x '*/rng_state_*' -d "$TMP"
rm "$TMP/hindi_models.zip"
M="$TMP/hindi_models/whisper-large-hi-noldcil"

"$UV" venv "$TMP/venv" --python 3.10 -q
"$UV" pip install --python "$TMP/venv/bin/python" -q "torch==2.1.1" --index-url https://download.pytorch.org/whl/cu121
"$UV" pip install --python "$TMP/venv/bin/python" -q "transformers==4.46.3" "ctranslate2==4.4.0" "numpy<2"
# ctranslate2's bundled .so requests an executable stack, which newer glibc refuses
for f in "$TMP"/venv/lib/python3.10/site-packages/ctranslate2.libs/*.so*; do .venv/bin/patchelf --clear-execstack "$f"; done
# the checkpoint ships a slow tokenizer only; faster-whisper needs tokenizer.json
"$TMP/venv/bin/python" -c "from transformers import WhisperTokenizerFast as T; T.from_pretrained('$M').save_pretrained('$M')"
mkdir -p models
"$TMP/venv/bin/ct2-transformers-converter" --model "$M" --output_dir "$OUT" \
  --copy_files tokenizer.json preprocessor_config.json --quantization int8_float16
echo "done: $OUT"
