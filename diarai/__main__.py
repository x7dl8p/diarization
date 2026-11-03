"""diarai command line.

    python -m diarai transcribe a.mp3 b.wav --out out/      # offline, writes .json (Sarvam format) .rttm .srt .txt
    python -m diarai serve [--host 127.0.0.1] [--port 8000] # HTTP + WebSocket API
    python -m diarai keygen                                 # create an API key (hash -> .env, key -> secrets/)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from diarai import config


def cmd_transcribe(args):
    from diarai.formats import parse_lang
    from diarai.pipeline import DiarAI

    pipe = DiarAI(language=parse_lang(args.language), beam_size=args.beam_size)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.audio:
        res = pipe(path, args.num_speakers, args.min_speakers, args.max_speakers)
        stem = args.out / path.stem
        out = lambda ext: stem.parent / (stem.name + ext)  # with_suffix would eat dots in the name
        out(".json").write_text(json.dumps(res.to_sarvam(), indent=1, ensure_ascii=False))
        out(".rttm").write_text(res.to_rttm(path.stem))
        out(".srt").write_text(res.to_srt())
        out(".txt").write_text(res.to_text())
        print(res.to_text())
        print(f"\n{path.name}: {res.duration:.1f}s audio, language {res.language}, in {res.timings['total']:.1f}s "
              f"({res.duration / res.timings['total']:.2f}x realtime) -> {stem}.{{json,rttm,srt,txt}}", file=sys.stderr)


def cmd_keygen(args):
    from diarai.server.auth import generate_key, hash_key

    key = generate_key()
    h = hash_key(key)
    env = config.ROOT / ".env"
    lines = env.read_text().splitlines() if env.exists() else []
    old = [l for l in lines if l.startswith("DIARAI_API_KEY_HASH=")]
    hashes = ([x for x in old[0].split("=", 1)[1].split(",") if x] if old and args.add else []) + [h]
    lines = [l for l in lines if not l.startswith("DIARAI_API_KEY_HASH=")] + ["DIARAI_API_KEY_HASH=" + ",".join(hashes)]
    env.write_text("\n".join(lines) + "\n")
    os.chmod(env, 0o600)
    sec = config.ROOT / "secrets"
    sec.mkdir(mode=0o700, exist_ok=True)
    keyfile = sec / "api_key.txt"
    keyfile.write_text(key + "\n")
    os.chmod(keyfile, 0o600)
    print(f"API key written to {keyfile} (mode 600); its scrypt hash to {env} "
          f"({len(hashes)} key{'s' if len(hashes) > 1 else ''} active). Restart the server to apply.")
    if args.print:
        print(key)


def cmd_serve(args):
    import uvicorn

    # models are local after scripts/setup_models.sh: don't call the Hugging Face Hub at every start
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not config.API_KEY_HASHES:
        sys.exit("no API key configured: run `python -m diarai keygen` first")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.ssl_certfile:
        print("warning: serving on a non-local interface without TLS; API keys travel in clear text. "
              "Use --ssl-certfile/--ssl-keyfile or a TLS reverse proxy.", file=sys.stderr)
    uvicorn.run("diarai.server.app:app", host=args.host, port=args.port, workers=1, log_level=args.log_level,
                ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile, ws_max_size=16 * 1024 * 1024)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m diarai",
                                 description="Local GPU transcription + speaker diarization (DiariZen + Whisper/IndicWhisper)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="transcribe + diarize files")
    t.add_argument("audio", nargs="+", type=Path)
    t.add_argument("--out", type=Path, default=Path("out"))
    t.add_argument("--language", default=None, help="e.g. hi / hi-IN / en-IN; auto-detected when omitted")
    t.add_argument("--num-speakers", type=int)
    t.add_argument("--min-speakers", type=int)
    t.add_argument("--max-speakers", type=int)
    t.add_argument("--beam-size", type=int, default=5)
    t.set_defaults(fn=cmd_transcribe)

    s = sub.add_parser("serve", help="run the HTTP + WebSocket API")
    s.add_argument("--host", default=os.environ.get("DIARAI_HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("DIARAI_PORT", "8000")))
    s.add_argument("--ssl-certfile")
    s.add_argument("--ssl-keyfile")
    s.add_argument("--log-level", default="info")
    s.set_defaults(fn=cmd_serve)

    k = sub.add_parser("keygen", help="create an API key")
    k.add_argument("--add", action="store_true", help="keep existing keys valid (default: replace them)")
    k.add_argument("--print", action="store_true", help="also print the key to stdout")
    k.set_defaults(fn=cmd_keygen)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
