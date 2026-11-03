#!/usr/bin/env python3
"""Stream an audio file to the diarai WebSocket API, the way a live client (microphone) would.

    .venv/bin/python scripts/stream_client.py realaudio/clip.mp3 [--language hi-IN] [--partials] [--fast]

The key is read from $DIARAI_API_KEY or secrets/api_key.txt. Audio is sent as Sarvam-style JSON messages
{"audio": {"data": <base64 16-bit WAV>, "sample_rate": "16000", "encoding": "audio/wav"}} every 100 ms
(in real time unless --fast); at the end {"type": "end"} asks for the final whole-session transcript.
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def wav_bytes(x: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--url", default="ws://127.0.0.1:8000/speech-to-text/ws")
    ap.add_argument("--language", default="unknown", help="language-code: unknown (auto) / hi-IN / en-IN ...")
    ap.add_argument("--partials", action="store_true", help="request live partial transcripts")
    ap.add_argument("--no-final", action="store_true", help="skip the final whole-session pass")
    ap.add_argument("--fast", action="store_true", help="send as fast as possible instead of in real time")
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--json", action="store_true", help="print raw server messages")
    args = ap.parse_args()

    key = os.environ.get("DIARAI_API_KEY") or (ROOT / "secrets/api_key.txt").read_text().strip()
    from diarai.audio import load_audio

    x = load_audio(args.audio)
    params = (f"language-code={args.language}&vad_signals=true&partial_results={str(args.partials).lower()}"
              f"&final_transcript={str(not args.no_final).lower()}")
    t0 = time.perf_counter()
    async with websockets.connect(f"{args.url}?{params}", additional_headers={"Api-Subscription-Key": key},
                                  max_size=16 * 1024 * 1024) as ws:

        async def sender():
            step = 16 * args.chunk_ms
            for i in range(0, len(x), step):
                msg = {"audio": {"data": base64.b64encode(wav_bytes(x[i:i + step])).decode(),
                                 "sample_rate": "16000", "encoding": "audio/wav"}}
                await ws.send(json.dumps(msg))
                if not args.fast:
                    await asyncio.sleep(max(0.0, t0 + (i + step) / 16000 - time.perf_counter()))
            await ws.send(json.dumps({"type": "end"}))

        send_task = asyncio.create_task(sender())
        async for raw in ws:
            m = json.loads(raw)
            now = time.perf_counter() - t0
            if args.json:
                print(json.dumps(m, ensure_ascii=False))
                continue
            d = m["data"]
            if m["type"] == "data":
                print(f"[{now:6.1f}s] spk {d['speaker_id']} {d['start_time_seconds']:6.1f}-{d['end_time_seconds']:6.1f} "
                      f"({d['language_code']}, +{d['metrics']['processing_latency']:.1f}s): {d['transcript']}")
            elif m["type"] == "partial":
                print(f"[{now:6.1f}s]   ... {d['transcript']}")
            elif m["type"] == "events":
                if d.get("signal_type") == "END_OF_STREAM":
                    break
            elif m["type"] == "final":
                print(f"\n[{now:6.1f}s] FINAL (whole-session diarization, {d['language_code']}):")
                for e in d["diarized_transcript"]["entries"]:
                    print(f"  spk {e['speaker_id']} {e['start_time_seconds']:6.1f}-{e['end_time_seconds']:6.1f}: {e['transcript']}")
            elif m["type"] == "error":
                print(f"[{now:6.1f}s] ERROR {d}")
        await send_task


if __name__ == "__main__":
    asyncio.run(main())
