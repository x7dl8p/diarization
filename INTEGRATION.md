# Using diarai from other services

diarai is a self-hosted speech-to-text service with speaker diarization (who said what). Its API follows Sarvam AI's speech-to-text response shapes. A service that already parses Sarvam output can usually switch by changing the URL and the key.

Base URL: `https://sarvam.androcoders.net`. In the office LAN it is `http://192.168.0.100:8085`.

## Authentication

Every request needs the API key. Send it as one of these:

| Method | Example |
|---|---|
| Sarvam-style header (preferred) | `Api-Subscription-Key: <key>` |
| Bearer token | `Authorization: Bearer <key>` |
| HTTP Basic | `-u any-user:<key>`; the username is ignored |
| WebSocket from a browser | subprotocols `["diarai", "key.<key>"]`; browsers can't set headers |

Keep these rules:
* Never put the key in a URL. URLs end up in logs.
* Store the key in your secret manager. For zdial that's Infisical, `/backend` → `DIARAI_API_KEY`.
* An IP is throttled for 5 minutes after 20 wrong keys.

Keys are created on the diarai host with `python -m diarai keygen` (`--add` keeps the existing keys working). The key is written to `secrets/api_key.txt` there.

## 1. Transcribe a recording: `POST /speech-to-text`

Use this for a finished recording: post-call analysis, voicemail, uploaded files.

```bash
curl -H "Api-Subscription-Key: $DIARAI_API_KEY" \
     -F file=@call.wav -F language_code=hi-IN \
     https://sarvam.androcoders.net/speech-to-text
```

Form fields:

| Field | Default | Meaning |
|---|---|---|
| `file` | | Any audio or video format, including 8 kHz telephony WAV. |
| `language_code` | `unknown` | `hi-IN`, `en-IN`, or `unknown` to auto-detect between Hindi and English. Pass `hi-IN` for Hindi call centres: calls often open with an English "hello". |
| `num_speakers` | | Set it when you know the count, e.g. `2` for a phone call. It improves speaker separation. |
| `with_timestamps` | `true` | Include word timestamps. |
| `with_diarization` | `true` | Include `diarized_transcript`. |

The response is the same as Sarvam's batch output:

```json
{
  "request_id": "6f1c…",
  "transcript": "हेलो सर … ",
  "diarized_transcript": {"entries": [
    {"transcript": "हेलो सर, मैं टाटा मोटर्स से बोल रहा हूँ", "start_time_seconds": 0.4,
     "end_time_seconds": 3.1, "speaker_id": "0"},
    {"transcript": "हाँ जी बोलिए", "start_time_seconds": 3.3, "end_time_seconds": 4.2, "speaker_id": "1"}
  ]},
  "timestamps": {"words": ["हेलो", "सर"], "start_time_seconds": [0.4, 0.8], "end_time_seconds": [0.8, 1.1]},
  "language_code": "hi-IN",
  "language_probability": null,
  "metrics": {"audio_duration": 99.3, "processing_latency": 41.2}
}
```

* **Speaker ids:** `"0"`, `"1"`, ... in order of first appearance. diarai does not know which speaker is the agent.
* **Synchronous:** there is no job and no polling; the call returns when the transcript is done. Budget about 0.5–1.5× the audio length on the current GPU, and set a client timeout of at least 10 minutes.

**Errors:** `{"error": {"message", "code"}}`

| HTTP status | Meaning |
|---|---|
| 400 | Undecodable or empty audio |
| 401 | Bad key |
| 413 | File over 1 GB |
| 429 | Throttled |
| 502 / 503 | The service or its GPU host is down. Fall back (see below). |

## 2. Live transcription: `WS /speech-to-text/ws`

Use this for audio arriving in real time.

```
wss://sarvam.androcoders.net/speech-to-text/ws?language-code=hi-IN&input_audio_codec=pcm_s16le&sample_rate=8000
Api-Subscription-Key: <key>
```

1. Wait for `{"type":"events","data":{"signal_type":"READY"}}`.
2. Send audio, either as binary frames of raw PCM in the declared codec and rate, or as Sarvam JSON messages: `{"audio":{"data":"<base64 wav>","sample_rate":"16000","encoding":"audio/wav"}}`.
3. Receive `{"type":"data","data":{transcript, speaker_id, start_time_seconds, end_time_seconds, language_code, metrics}}`, one message per finished utterance of one speaker.
4. Optional: `partial_results=true` adds `{"type":"partial"}` live captions.
5. Send `{"type":"end"}`. The server flushes, sends a `final` message (the whole session re-diarized offline, with the same shape as the REST response), then `END_OF_STREAM`, then closes.

Live mode is mainly for mixed audio, where you need diarization. A call that is already split per leg (agent and customer on separate channels) doesn't need diarization. The server allows 4 concurrent streams (`DIARAI_MAX_STREAMS`), and every stream shares one GPU.

## 3. Health: `GET /health`

No key needed. Use it for load-balancer and uptime checks.

```json
{"status": "ok", "models": {"default": "large-v3", "hi": ".../indicwhisper-large-hi-ct2",
 "diarization": "BUT-FIT/diarizen-wavlm-large-s80-md"}, "streams": 0, "max_streams": 4}
```

## Always keep a fallback

diarai runs on a single GPU machine. Treat it as preferred, not guaranteed:
* Try diarai first.
* On a network error, a 5xx, a timeout or an empty transcript, use your previous provider (Sarvam).
* Log which provider served each request.

zdial does exactly this. See below.

## zdial integration (reference implementation)

zdial uses diarai for **post-call transcription**:

| File | Role |
|---|---|
| `backend/src/modules/post-call/diarai-stt.service.ts` | REST client |
| `backend/src/modules/post-call/post-call.service.ts` (`executeTranscription`) | diarai first, Sarvam fallback |
| `backend/src/modules/post-call/diarized-transcript.ts` | Shared formatter. Both providers produce identical `[mm:ss] [Speaker N]: text` lines, so the Gemini analysis prompt doesn't change. |

Configuration, in the backend env (Infisical `staging` / `prod`, folder `/backend`):

```
DIARAI_URL=http://192.168.0.100:8085   # LAN: no Cloudflare 100 s limit
DIARAI_API_KEY=<key from the diarai host>
```

If either is empty, zdial uses Sarvam only. That's the safe default, so merging the code changes nothing until the key is set.

Live call transcription in zdial stays on Sarvam realtime. It streams the agent and customer legs separately, which already identifies the speakers, and two extra GPU streams per call would saturate the current GPU at about two concurrent calls.

### Effect on zdial's AI conclusions (benchmark)

See `docs/zdial-benchmark.md` for 5 real zdial calls. For each, it compares the summary, sentiment, script choice and script score Gemini produces from diarai's transcript with the ones from the Sarvam live transcript.

## Deployment (current)

```
client ─HTTPS─> Cloudflare (tunnel, *.androcoders.net) ─> andro 192.168.0.200: caddy container
  └─ sarvam.androcoders.net → 192.168.0.100:8085 / 192.168.1.24:8085
       └─ desktop-514csmm: pm2 app "diarai" (python -m diarai serve, port 8085, RTX 2060)
```

* **Host:** desktop-514csmm runs diarai from `C:\Users\ali\diarai`, under pm2, using `deploy/diarai.ecosystem.config.js`. Its API key is in `C:\Users\ali\diarai\secrets\api_key.txt`.
* **Measured speed** on the RTX 2060: a 365 s call took 213 s, about 1.7× faster than real time.
* **Port 8085:** the firewall already allows it (`ocr-http-8085`). The OCR service used the same port, so OCR and diarai can't run at the same time:
  * Back to OCR: `pm2 stop diarai && pm2 start ocr-api ocr-llama`.
  * Back to diarai: `pm2 stop ocr-api ocr-llama && pm2 start diarai`.
  * sts (`s2s-*`) is stopped too, to free the GPU memory.
* **Caddy block:** in `/home/andro/deployments/caddy/Caddyfile-template` on andro.
* **No `basic_auth` in Caddy:** diarai checks its own key, and a second login prompt would break API clients.
* **Requests that run long:** Cloudflare closes a request that doesn't respond within 100 s (HTTP 524). For longer recordings, call the LAN URL from inside the office network. zdial's backend does this when `DIARAI_URL=http://192.168.0.100:8085`.
