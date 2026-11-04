"""HTTP + WebSocket API (Sarvam-AI-compatible shapes, see diarai/formats.py).

    GET  /                          web UI: paste key, pick a file, it streams live (no auth to load the page)
    GET  /health                    liveness + loaded models (no auth)
    POST /speech-to-text            whole file -> transcript + diarization (multipart upload)
    WS   /speech-to-text/ws         live: audio chunks in, transcripts with speakers out

Run:  python -m diarai serve --host 127.0.0.1 --port 8000
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from diarai import config
from diarai.audio import decode_chunk, load_audio
from diarai.formats import SpeakerIds, entry, lang_code, parse_lang, word_timestamps
from diarai.models import BATCH, get_models
from diarai.server.auth import Authenticator, extract_key, extract_ws_key

log = logging.getLogger("diarai.server")
STATIC = os.path.join(os.path.dirname(__file__), "static")

MAX_STREAMS = int(os.environ.get("DIARAI_MAX_STREAMS", "4"))
IDLE_TIMEOUT_S = float(os.environ.get("DIARAI_IDLE_TIMEOUT_S", "120"))


class State:
    models = None
    auth: Authenticator = None
    streams = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    State.auth = Authenticator(config.API_KEY_HASHES)
    if not State.auth.enabled:
        raise RuntimeError("no API key configured: run `python -m diarai keygen` first (DIARAI_API_KEY_HASH)")
    t = time.perf_counter()
    State.models = await asyncio.to_thread(get_models)
    log.info("models ready in %.1fs (languages with dedicated models: %s)", time.perf_counter() - t,
             sorted(State.models.lang_models))
    yield


app = FastAPI(title="diarai", version="1.0", lifespan=lifespan,
              description="Local GPU speech-to-text with speaker diarization (DiariZen + Whisper/IndicWhisper)")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

_CSP = ("default-src 'self'; connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; "
        "style-src 'self'; script-src 'self'; worker-src 'self'; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'")


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path == "/" or request.url.path.startswith("/static"):
        resp.headers["Content-Security-Policy"] = _CSP
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers.setdefault("Cache-Control", "no-cache")  # revalidate (ETag) instead of a stale CDN copy
    return resp


def _index_html() -> str:
    """index.html with the static asset URLs tagged by a content hash, so CDN caches (Cloudflare keeps .js/.css
    for hours) never serve an old app.js to a new page."""
    import hashlib

    h = hashlib.sha256()
    for name in sorted(os.listdir(STATIC)):
        with open(os.path.join(STATIC, name), "rb") as f:
            h.update(name.encode() + f.read())
    v = h.hexdigest()[:10]
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
        html = f.read()
    for asset in ("app.js", "style.css"):
        html = html.replace(f"/static/{asset}\"", f"/static/{asset}?v={v}\"")
    return html


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def index():
    return HTMLResponse(_index_html(), headers={"Cache-Control": "no-cache"})


def _error(status: int, message: str, code: str):
    return JSONResponse(status_code=status, content={"error": {"message": message, "code": code}})


@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException):
    code = {400: "invalid_request_error", 401: "unauthorized", 403: "forbidden", 413: "payload_too_large",
            429: "rate_limit_exceeded"}.get(exc.status_code, "internal_server_error")
    return _error(exc.status_code, str(exc.detail), code)


def _client(conn) -> str:
    return conn.client.host if conn.client else "?"


async def require_key(request: Request):
    client = _client(request)
    if State.auth.throttled(client):
        raise HTTPException(429, "too many failed authentication attempts")
    ok = await asyncio.to_thread(State.auth.check, extract_key(request.headers), client)
    if not ok:
        raise HTTPException(401, "invalid or missing API key (Api-Subscription-Key header, Bearer or Basic auth)")


@app.get("/health")
async def health():
    m = State.models
    return {"status": "ok" if m else "loading",
            "models": {"default": config.ASR_MODEL, **{k: str(v) for k, v in config.LANG_MODELS.items()},
                       "diarization": config.DIAR_MODEL},
            "streams": State.streams, "max_streams": MAX_STREAMS}


@app.post("/speech-to-text", dependencies=[Depends(require_key)])
async def speech_to_text(file: UploadFile = File(...), language_code: str = Form("unknown"),
                         model: str = Form("diarai"), with_timestamps: bool = Form(True),
                         with_diarization: bool = Form(True), num_speakers: Optional[int] = Form(None)):
    from diarai.pipeline import DiarAI

    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    # delete=False + explicit cleanup: Windows cannot reopen a NamedTemporaryFile that is still open
    tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename or "")[1], delete=False)
    try:
        with tmp:
            size = 0
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"file larger than {config.MAX_UPLOAD_MB} MB")
                tmp.write(chunk)
        try:
            wav = await asyncio.to_thread(load_audio, tmp.name)
        except Exception:
            raise HTTPException(400, "could not decode the audio file")
    finally:
        os.unlink(tmp.name)
    if len(wav) < 1600:
        raise HTTPException(400, "audio is empty or shorter than 0.1 s")
    if num_speakers is not None and not 1 <= num_speakers <= 20:
        raise HTTPException(400, "num_speakers must be between 1 and 20")
    t0 = time.perf_counter()
    result = await asyncio.to_thread(DiarAI(State.models), wav, num_speakers=num_speakers,
                                     language=parse_lang(language_code), priority=BATCH)
    result.timings["total"] = time.perf_counter() - t0
    return result.to_sarvam(uuid.uuid4().hex, with_timestamps, with_diarization)


# ------------------------------------------------------------------- streaming
def _bool(v: Optional[str], default: bool) -> bool:
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@app.websocket("/speech-to-text/ws")
async def speech_to_text_ws(ws: WebSocket):
    from diarai.stream import StreamSession

    q = ws.query_params
    client = _client(ws)
    key, subprotocol = extract_ws_key(ws.headers)

    async def refuse(code: int, reason: str):
        # accept first so the client (a browser in particular) receives the close code and reason
        await ws.accept(subprotocol=subprotocol)
        await ws.close(code=code, reason=reason)

    if State.auth.throttled(client):
        return await refuse(1008, "too many failed authentication attempts")
    if not await asyncio.to_thread(State.auth.check, key, client):
        return await refuse(1008, "invalid or missing API key")
    if State.streams >= MAX_STREAMS:
        return await refuse(1013, "server busy: too many live streams")
    try:
        sample_rate = int(q.get("sample_rate", config.SAMPLE_RATE))
        codec = q.get("input_audio_codec", "pcm_s16le")
        beam = int(q.get("beam_size", 1))
        num_speakers = int(q["num_speakers"]) if q.get("num_speakers") else None
        session = StreamSession(
            State.models, language=parse_lang(q.get("language-code") or q.get("language_code")),
            beam_size=max(1, min(beam, 10)), vad_threshold=0.35 if _bool(q.get("high_vad_sensitivity"), False) else 0.5,
            end_silence=float(q.get("end_silence", 0.6)), max_utterance=float(q.get("max_utterance", 10.0)),
            word_timestamps=_bool(q.get("word_timestamps"), False))
    except (ValueError, TypeError) as e:
        return await refuse(1003, f"bad query parameter: {e}"[:120])
    vad_signals = _bool(q.get("vad_signals"), False)
    partials = _bool(q.get("partial_results"), False)
    final_pass = _bool(q.get("final_transcript"), True)

    await ws.accept(subprotocol=subprotocol)
    State.streams += 1
    rid = uuid.uuid4().hex
    ids = SpeakerIds()
    segments: asyncio.Queue = asyncio.Queue()
    busy = {"n": 0, "lag": 0.0}
    closed = {"v": False}  # client gone: queued segments are dropped  # lag: how late the last finished segment was delivered (s)
    send_lock = asyncio.Lock()

    async def send(kind: str, data: dict):
        async with send_lock:
            await ws.send_text(json.dumps({"type": kind, "data": data}, ensure_ascii=False))

    def data_msg(u, latency: float) -> dict:
        return {"request_id": rid, "transcript": u.text, "timestamps": word_timestamps(u.words) if session.word_timestamps else None,
                "diarized_transcript": {"entries": [entry(u.text, u.start, u.end, ids(u.speaker))]},
                "speaker_id": ids(u.speaker), "start_time_seconds": round(u.start, 2),
                "end_time_seconds": round(u.end, 2), "language_code": lang_code(session.language),
                "language_probability": session.language_probability,
                "metrics": {"audio_duration": round(u.end - u.start, 2), "processing_latency": round(latency, 3)}}

    async def worker():
        while True:
            seg, t_seen = await segments.get()
            if seg is not None and closed["v"]:
                segments.task_done()
                continue
            if seg is None:
                segments.task_done()
                return
            busy["n"] += 1
            try:
                for u in await asyncio.to_thread(session.process, seg):
                    await send("data", data_msg(u, time.perf_counter() - t_seen))
                busy["lag"] = time.perf_counter() - t_seen
            except Exception as e:  # keep the stream alive; report the failed segment
                log.exception("segment failed")
                await send("error", {"error": f"segment {seg.start:.2f}-{seg.end:.2f}s failed: {e}", "code": "processing_error"})
            finally:
                busy["n"] -= 1
                segments.task_done()

    async def partial_loop():
        last = None
        while True:
            await asyncio.sleep(1.5)
            if busy["n"] or not segments.empty() or busy["lag"] > 3.0:
                continue  # finals first: captions only when this stream is idle and keeping up
            p = await asyncio.to_thread(session.partial)
            if p and p != last:
                last = p
                await send("partial", {"request_id": rid, "transcript": p[1], "start_time_seconds": round(p[0], 2),
                                       "language_code": lang_code(session.language)})

    def handle(events):
        for ev in events:
            if ev.kind == "segment":
                segments.put_nowait((ev, time.perf_counter()))
            elif vad_signals:
                now = datetime.now(timezone.utc)
                asyncio.ensure_future(send("events", {
                    "event_type": "vad", "signal_type": "START_SPEECH" if ev.kind == "start" else "END_SPEECH",
                    "timestamp": now.isoformat(), "occured_at": str(now.timestamp()),
                    "audio_time_seconds": round(ev.start, 2)}))

    await send("events", {"event_type": "stream", "signal_type": "READY", "request_id": rid,
                          "timestamp": datetime.now(timezone.utc).isoformat()})
    tasks = [asyncio.create_task(worker())] + ([asyncio.create_task(partial_loop())] if partials else [])
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=IDLE_TIMEOUT_S)
            except asyncio.TimeoutError:
                await send("error", {"error": f"no audio for {IDLE_TIMEOUT_S:.0f}s, closing", "code": "idle_timeout"})
                break
            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(msg.get("code", 1000))
            if msg.get("bytes") is not None:  # binary frame: raw audio in input_audio_codec
                handle(session.feed(decode_chunk(msg["bytes"], sample_rate, codec)))
                continue
            try:
                body = json.loads(msg.get("text") or "{}")
            except json.JSONDecodeError:
                await send("error", {"error": "messages must be JSON or binary audio", "code": "invalid_message"})
                continue
            if "audio" in body:
                a = body["audio"] or {}
                try:
                    chunk = decode_chunk(base64.b64decode(a.get("data", "")),
                                         int(a.get("sample_rate", sample_rate)), codec)
                except (ValueError, TypeError) as e:
                    await send("error", {"error": f"bad audio chunk: {e}", "code": "invalid_audio"})
                    continue
                try:
                    handle(session.feed(chunk))
                except OverflowError:
                    await send("error", {"error": "stream exceeds the maximum duration", "code": "too_long"})
                    break
            elif body.get("type") == "flush":
                handle(session.flush())
            elif body.get("type") in ("end", "close", "finalize"):
                handle(session.flush())
                await segments.join()
                if final_pass and session.audio.seconds > 0.5:
                    t0 = time.perf_counter()
                    result = await asyncio.to_thread(session.finalize, num_speakers)
                    result.timings["total"] = time.perf_counter() - t0
                    out = result.to_sarvam(rid)
                    _align_final_speakers(out, session, ids)
                    await send("final", out)
                await send("events", {"event_type": "stream", "signal_type": "END_OF_STREAM",
                                      "timestamp": datetime.now(timezone.utc).isoformat()})
                break
            else:
                await send("error", {"error": "unknown message", "code": "invalid_message"})
        await ws.close()
    except WebSocketDisconnect:
        pass
    finally:
        State.streams -= 1
        closed["v"] = True
        segments.put_nowait((None, 0.0))
        for t in tasks[1:]:
            t.cancel()


def _align_final_speakers(out: dict, session, ids: SpeakerIds):
    """Renumber the final pass's speakers to the ids already used live (greedy, by time overlap)."""
    entries = (out.get("diarized_transcript") or {}).get("entries", [])
    live = [(ids(u.speaker), u.start, u.end) for u in session.utterances]
    ov = {}
    for e in entries:
        for lid, s, t in live:
            o = min(e["end_time_seconds"], t) - max(e["start_time_seconds"], s)
            if o > 0:
                ov[(e["speaker_id"], lid)] = ov.get((e["speaker_id"], lid), 0.0) + o
    remap, used = {}, set()
    for (fid, lid), _ in sorted(ov.items(), key=lambda kv: -kv[1]):
        if fid not in remap and lid not in used:
            remap[fid], used = lid, used | {lid}
    nxt = len(ids.ids)
    for e in entries:
        if e["speaker_id"] not in remap:
            while str(nxt) in used:
                nxt += 1
            remap[e["speaker_id"]], used = str(nxt), used | {str(nxt)}
        e["speaker_id"] = remap[e["speaker_id"]]
