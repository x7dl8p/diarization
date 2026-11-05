"""Benchmark store: per-call results of the zdial comparison (Sarvam vs diarai transcripts, AI analysis of each,
judge verdicts), uploaded by the benchmark runner and shown in the web UI.

Layout under config.BENCH_DIR:  status.json,  calls/<id>.json,  audio/<id>.<ext>
Every route sits behind the API key (the router is mounted with `require_key`): the data is customer calls.
"""

import json
import os
import re
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from diarai import config

router = APIRouter(prefix="/bench")

_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_AUDIO_EXT = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".opus": "audio/ogg",
              ".flac": "audio/flac", ".m4a": "audio/mp4", ".webm": "audio/webm"}


def _dir(*parts) -> Path:
    d = Path(config.BENCH_DIR, *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_id(call_id: str) -> str:
    if not _ID.match(call_id):
        raise HTTPException(400, "bad call id")
    return call_id


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _audio_path(call_id: str):
    for ext in _AUDIO_EXT:
        p = Path(config.BENCH_DIR, "audio", call_id + ext)
        if p.exists():
            return p
    return None


def _row(d: dict) -> dict:
    """The light list row for one call (no transcripts)."""
    a, b = d.get("analysis_sarvam") or {}, d.get("analysis_diarai") or {}
    return {
        "id": d["id"], "duration": d.get("duration"), "direction": d.get("direction"),
        "created": d.get("created"), "error": d.get("error"),
        "verdict": (d.get("verdict") or {}).get("verdict"),
        "noise": (d.get("noise") or {}).get("verdict"),
        "script": [a.get("scriptKey"), b.get("scriptKey")],
        "score": [a.get("scriptScore"), b.get("scriptScore")],
        "sentiment": [a.get("sentiment"), b.get("sentiment")],
        "steps": d.get("steps"), "steps_noise": d.get("steps_noise"),
        "speakers": d.get("speakers"),
        "diarai_seconds": d.get("diarai_seconds"),
    }


@router.get("/summary")
async def summary():
    rows = []
    for p in sorted(_dir("calls").glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            rows.append(_row(json.loads(p.read_text(encoding="utf-8"))))
        except (ValueError, KeyError):
            continue
    status_file = Path(config.BENCH_DIR, "status.json")
    status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else None
    return {"status": status, "calls": rows}


@router.get("/calls/{call_id}")
async def get_call(call_id: str):
    p = Path(config.BENCH_DIR, "calls", _check_id(call_id) + ".json")
    if not p.exists():
        raise HTTPException(404, "no such call")
    d = json.loads(p.read_text(encoding="utf-8"))
    d["has_audio"] = _audio_path(call_id) is not None
    return d


@router.get("/calls/{call_id}/audio")
async def get_audio(call_id: str):
    p = _audio_path(_check_id(call_id))
    if not p:
        raise HTTPException(404, "no audio for this call")
    return FileResponse(p, media_type=_AUDIO_EXT[p.suffix], headers={"Cache-Control": "private, max-age=3600"})


@router.put("/calls/{call_id}")
async def put_call(call_id: str, meta: str = Form(...), audio: UploadFile = File(None)):
    _check_id(call_id)
    try:
        data = json.loads(meta)
    except ValueError:
        raise HTTPException(400, "meta is not JSON")
    data["id"] = call_id
    data.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if audio is not None:
        ext = os.path.splitext(audio.filename or "")[1].lower()
        if ext not in _AUDIO_EXT:
            raise HTTPException(400, f"audio must be one of {sorted(_AUDIO_EXT)}")
        old = _audio_path(call_id)
        if old:
            old.unlink()
        dst = _dir("audio") / (call_id + ext)
        with open(dst.with_suffix(".part"), "wb") as f:
            while chunk := await audio.read(1 << 20):
                f.write(chunk)
        os.replace(dst.with_suffix(".part"), dst)
    _write_json(_dir("calls") / (call_id + ".json"), data)
    return {"ok": True}


@router.put("/status")
async def put_status(status: dict):
    status["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(_dir() / "status.json", status)
    return {"ok": True}
