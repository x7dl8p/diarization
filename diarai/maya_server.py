"""maya-transcribe ASR service (maya-research/maya-transcribe, a Qwen3-ASR 0.6B fine-tune for Hindi/Hinglish).

Runs in its own virtualenv (Python 3.12, torch 2.14, transformers 5.20), separate from diarai's (torch 2.1 for
DiariZen), and only on localhost. diarai does the speaker separation and sends this service the single-speaker
pieces (<= 25 s; maya takes at most 30 s per clip); this service only turns each piece into text.

    POST /transcribe?style=native     body: the clips' float32 16 kHz mono samples, concatenated
                                      header X-Lengths: samples per clip, comma separated
        -> {"texts": [...], "seconds": <GPU time>}
    GET  /health

Standalone on purpose (imports nothing from diarai). Run:  python diarai/maya_server.py  (env MAYA_*).
"""

import logging
import os
import threading
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL = os.environ.get("MAYA_MODEL", os.path.join(os.path.dirname(__file__), "..", "models", "maya-transcribe"))
DEVICE = os.environ.get("MAYA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = os.environ.get("MAYA_DTYPE", "auto")  # auto | float16 | bfloat16 | float32
BATCH = int(os.environ.get("MAYA_BATCH", "8"))
SR = 16000
MAX_CLIP_S = 30.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s maya: %(message)s")
log = logging.getLogger("maya")


def _dtype():
    if DTYPE != "auto":
        return getattr(torch, DTYPE)
    if DEVICE.startswith("cuda"):
        # Turing/Pascal have no bf16: fp16 is the fast path there
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


t0 = time.perf_counter()
processor = AutoProcessor.from_pretrained(MODEL, local_files_only=os.path.isdir(MODEL), trust_remote_code=False)
model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=_dtype(), local_files_only=os.path.isdir(MODEL),
                                                 trust_remote_code=False).to(DEVICE).eval()
log.info("loaded %s on %s (%s) in %.1fs", MODEL, DEVICE, model.dtype, time.perf_counter() - t0)
_lock = threading.Lock()
_batched = None  # whether apply_transcription_request takes a list of clips; found out on first use


def _generate(clips, style, max_new_tokens):
    inputs = processor.apply_transcription_request(audio=clips if len(clips) > 1 else clips[0],
                                                   prompt=f"style: {style}", language="Hindi")
    inputs = inputs.to(model.device, model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")


def transcribe(clips, style="native"):
    global _batched
    texts = []
    for i in range(0, len(clips), BATCH):
        group = clips[i:i + BATCH]
        longest = max(len(c) for c in group) / SR
        # bound the output to what the audio can hold (fast Hindi stays well under 24 tokens/s)
        max_new = int(min(512, 32 + 24 * longest))
        if _batched is not False and len(group) > 1:
            try:
                got = _generate(group, style, max_new)
                if len(got) == len(group):
                    _batched = True
                    texts += got
                    continue
            except Exception as e:  # noqa: BLE001 - processor without list support: fall back per clip
                if _batched:
                    raise
                log.info("batched requests unsupported (%s); decoding clip by clip", e)
            _batched = False
        for c in group:
            texts += _generate([c], style, int(min(512, 32 + 24 * len(c) / SR)))
    return [t.strip() for t in texts]


app = FastAPI(title="maya-transcribe", docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(os.path.normpath(MODEL)), "device": DEVICE,
            "dtype": str(model.dtype), "batched": _batched}


@app.post("/transcribe")
async def post_transcribe(request: Request, style: str = "native"):
    if style not in ("native", "mixed"):
        raise HTTPException(400, "style must be native or mixed")
    try:
        lengths = [int(x) for x in request.headers.get("x-lengths", "").split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "bad X-Lengths")
    body = await request.body()
    samples = np.frombuffer(body, dtype="<f4")
    if not lengths or sum(lengths) != len(samples) or min(lengths) <= 0:
        raise HTTPException(400, "X-Lengths must list the samples of each clip and add up to the body")
    if max(lengths) > MAX_CLIP_S * SR:
        raise HTTPException(400, f"each clip must be at most {MAX_CLIP_S:.0f} s")
    clips, o = [], 0
    for n in lengths:
        clips.append(np.ascontiguousarray(samples[o:o + n], dtype=np.float32))
        o += n
    t = time.perf_counter()
    with _lock:  # one generate at a time on the GPU
        try:
            texts = transcribe(clips, style)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(503, "maya: GPU out of memory")
    return {"texts": texts, "seconds": round(time.perf_counter() - t, 3)}


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("MAYA_HOST", "127.0.0.1"), port=int(os.environ.get("MAYA_PORT", "8086")),
                log_level="warning")
