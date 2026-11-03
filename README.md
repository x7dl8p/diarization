# diarai: local GPU speech-to-text with speaker diarization (offline and live)

Who said what, on your own GPU, with a Sarvam-AI-compatible API.

* **Diarization:** [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) (WavLM-Large EEND + VBx) with turn-level re-clustering.
* **Hindi:** **IndicWhisper large** (AI4Bharat Vistaar) transcribes Hindi. On our clips it had 17.6% speaker-attributed WER against Sarvam, versus 24.6% for Whisper large-v3.
* **English and everything else:** **Whisper large-v3** transcribes, and also does language ID.
* **Batch:** `POST /speech-to-text` takes a file and returns the transcript with speakers.
* **Live:** `WS /speech-to-text/ws` takes audio chunks and returns transcript lines with speaker ids while the audio streams, plus an optional whole-session final pass.
* **Auth:** API key, stored only as an scrypt hash. Send it as `Api-Subscription-Key`, Bearer or HTTP Basic.

## Quick start

```bash
scripts/setup_models.sh                    # download + build models (once)
python -m diarai keygen                    # API key -> secrets/api_key.txt (mode 600), hash -> .env
python -m diarai serve                     # http://127.0.0.1:8000  (docs at /docs)

KEY=$(cat secrets/api_key.txt)
curl -H "Api-Subscription-Key: $KEY" -F file=@call.mp3 -F language_code=unknown \
     http://127.0.0.1:8000/speech-to-text
.venv/bin/python scripts/stream_client.py call.mp3 --partials      # live demo client
python -m diarai transcribe call.mp3 --out out/                     # no server: out/call.{json,rttm,srt,txt}
```

## REST: `POST /speech-to-text`

Multipart form fields:

| Field | Default | Meaning |
|---|---|---|
| `file` | | Any audio or video file that PyAV/ffmpeg can decode. |
| `language_code` | `unknown` | `hi-IN`, `en-IN`, ... or `unknown` to auto-detect. Detection is limited to `DIARAI_LANGUAGES`, default `en,hi`. |
| `with_timestamps` | `true` | Include word timestamps. |
| `with_diarization` | `true` | Include the speaker-attributed entries. |
| `num_speakers` | | Number of speakers, if known. |

The response uses the same keys as Sarvam's speech-to-text:

```json
{"request_id": "...", "transcript": "full text",
 "timestamps": {"words": ["..."], "start_time_seconds": [0.0], "end_time_seconds": [0.2]},
 "diarized_transcript": {"entries": [{"transcript": "...", "start_time_seconds": 0.0,
                                      "end_time_seconds": 6.16, "speaker_id": "0"}]},
 "language_code": "hi-IN", "language_probability": 0.9992,
 "metrics": {"audio_duration": 45.19, "processing_latency": 35.4}}
```

Speaker ids are `"0"`, `"1"`, ... in order of first appearance. `timestamps` is word-level, where Sarvam returns phrase chunks under the same keys.

Errors come back as `{"error": {"message", "code"}}`:

| HTTP status | Meaning |
|---|---|
| 400 | Bad input |
| 401 | Missing or wrong key |
| 413 | File too large (`DIARAI_MAX_UPLOAD_MB`, default 1024) |
| 429 | Too many failed authentication attempts from one IP |

## Live: `WS /speech-to-text/ws`

Authenticate with a header on the WebSocket handshake. The connection is refused with close code 1008 if the key is wrong.

### Query parameters

These follow Sarvam's names:

| Parameter | Default | Meaning |
|---|---|---|
| `language-code` | `unknown` | `hi-IN`, `en-IN`, or `unknown` to detect from the first speech and then lock. |
| `sample_rate` | 16000 | Sample rate of raw PCM chunks. WAV chunks carry their own rate. |
| `input_audio_codec` | `pcm_s16le` | For raw chunks: `pcm_s16le` or `pcm_f32le`. |
| `vad_signals` | false | Send `START_SPEECH` / `END_SPEECH` events. |
| `high_vad_sensitivity` | false | Lower VAD threshold. |
| `partial_results` | false | Extension: live captions of the sentence still being spoken. |
| `final_transcript` | true | Extension: on `end`, re-process the whole session offline and send the most accurate transcript. |
| `num_speakers` | | Extension: speaker count for the final pass. |
| `beam_size` | 5 | Extension: decoding beam width. |
| `end_silence` | 0.6 | Extension: seconds of silence that end a segment. |
| `max_utterance` | 10 | Extension: longest segment, in seconds. |

### Client to server

| Message | Meaning |
|---|---|
| `{"audio": {"data": "<base64>", "sample_rate": "16000", "encoding": "audio/wav"}}` | Audio chunk, Sarvam format. The data is a 16-bit WAV, or raw PCM. |
| Binary frame | Raw PCM in `input_audio_codec`. Less overhead than base64. |
| `{"type": "flush"}` | Finish the sentence in progress now. |
| `{"type": "end"}` | Flush, wait for everything, send `final`, then close. |

### Server to client

Every message is `{"type", "data"}`.

**`data`**: one finished utterance of one speaker.

```json
{"request_id", "transcript", "speaker_id", "start_time_seconds", "end_time_seconds",
 "diarized_transcript": {"entries": [...]}, "timestamps": {...}, "language_code", "language_probability",
 "metrics": {"audio_duration", "processing_latency"}}
```

**`partial`**: `{"transcript", "start_time_seconds"}` for the sentence still being spoken. Only sent when `partial_results=true`.

**`events`**: `{"signal_type": "READY" | "START_SPEECH" | "END_SPEECH" | "END_OF_STREAM", ...}`. `READY` is sent once the key is accepted; start sending audio after it.

**`final`**: same shape as the REST response, for the whole session. Speaker ids are aligned to the ids already used live.

**`error`**: `{"error", "code"}`.

### How a live segment is processed

1. **Endpointing:** a streaming Silero VAD ends a segment after 0.6 s of silence. Continuous speech, or speech over music, is cut at its quietest point every 10 s or less.
2. **Diarization:** DiariZen's EEND runs on the last 16 s, which covers the segment plus already-labelled context. It finds the speaker changes and overlaps inside the segment.
   * Local speakers are linked to session speakers first by overlap with the labelled context, otherwise by WeSpeaker embedding against running centroids.
   * A new speaker needs at least 2 s of clean speech.
3. **Transcription:** each speaker's part is transcribed on its own, with the language's model. The language is detected from the first speech, then locked.

## Accuracy and speed

Measured on an RTX 2050 laptop GPU (4 GB). Against Sarvam saaras:v4 on 4 real clips (3.6 min; Sarvam is a strong reference, not ground truth). "Closeness" means 1 − cpWER, the right words under the right speaker.

| Clip | Offline closeness | Live closeness | Offline latency | Live: text after segment end, p50 |
|---|---|---|---|---|
| podcast_hi (61 s) | 83.1% | 82.2% | 56 s | 5.4 s |
| comedy_hi (45 s) | 81.2% | 82.0% | 36 s | 3.0 s |
| agent_en (70 s) | 94.8% | 94.3% | 22 s | 1.4 s |
| meta_en (42 s)* | 86.9% | 83.4% | 14 s | 1.8 s |
| **All, pooled** | **87.0%** | **86.0%** | | |

\* Sarvam hears a third speaker in meta_en that we judge spurious.

Diarization error vs Sarvam is 1.8% offline and 3.9% live. Hindi character-level closeness is 91%. The remaining Hindi gap is mostly spelling variants and digits written out as words, both of which Sarvam is also inconsistent about.

**Speed limits on this GPU:**
* Hindi decoding runs at about 0.8× realtime of compute. Devanagari needs many byte-level tokens, so live Hindi lags 3–7 s behind, versus 1–3 s for English. A bigger GPU shortens both roughly in proportion.
* Partials share the GPU with finished segments. They only run when a stream has no queued segments.

Run the evaluations with `python -m diarai.eval.real_eval --mode offline|stream --auto-language --out results/real/X`. The reference transcripts come from `scripts/sarvam_reference.py`.

## Architecture

```
diarai/
  config.py        paths, defaults, env vars (.env is loaded)
  models.py        one process-wide registry: large-v3 + IndicWhisper (one resident on the GPU, the other parked
                   in CPU RAM, ~1 s swap), DiariZen, WeSpeaker; GPU access through a priority lock
                   (live 0 < partial 1 < batch 2) so live streams overtake batch jobs between units
  pipeline.py      offline: DiariZen → turn refinement → language ID → per-speaker-unit ASR → utterances
  stream.py        live: streaming VAD, endpointer, windowed EEND + online speaker linking, session
  formats.py       Sarvam-compatible output
  audio.py         file / chunk decoding (WAV, PCM s16le/f32le, resampling)
  server/app.py    FastAPI REST + WebSocket;  server/auth.py  scrypt key hashes, throttling
  eval/            metrics, Sarvam comparison (offline + simulated live), synthetic/AMI benchmarks
scripts/           setup_models.sh, stream_client.py, sarvam_reference.py, run_benchmarks.sh
```

### What we added to stock DiariZen + Whisper

Each change was kept only if it helped on dev data:

| Change | Effect |
|---|---|
| Turn-level re-clustering: re-embed every speaker turn, then reassign it to the nearest centroid | Synthetic dev speaker confusion 3.6% → 0.7% |
| AHC threshold 0.6 → 0.8 | Exact speaker count 90% → 100% |
| Diarize first, then decode each speaker's turns separately | No words cross speaker changes; Hindi cpWER 26.7% → 22.7% |
| IndicWhisper large for Hindi, decoded without timestamp tokens and stopped after the first segment | Otherwise faster-whisper hallucinates endings in the padding; cpWER 24.6% → 17.6% |
| Hindi style prompt (loanwords and numbers in Devanagari), cross-talk dedupe, ≤ 6 words/s filter | |

Tried and dropped:
* Meta MMS-1b-all: 40% cpWER, no language model.
* IndicWhisper medium: no better than large-v3.
* Concurrent long-form Whisper with word-to-speaker merging.
* large-v3 decoded without timestamps: runs out of memory.

## Web UI

Open `http://<host>:8000/`, paste the key, then drop an audio file:
* **Streaming:** the file plays and its audio streams to `/speech-to-text/ws` in real time.
* **Live lines:** transcript lines with speakers appear as they're recognised, plus live captions for the sentence in progress.
* **Progress:** the card's border fills as the audio plays.
* **Final pass:** at the end, the whole recording is re-processed and the final transcript replaces the live one. It can be copied or downloaded as .txt or .json.

The page has no dependencies and a strict CSP. The key is sent as a WebSocket subprotocol (`["diarai", "key.<KEY>"]`), never in the URL, and is stored in `localStorage` only if "remember" is ticked.

## Deploy

**Using it from another service** (REST, WebSocket, keys, fallback, the zdial integration): see
[INTEGRATION.md](INTEGRATION.md).

Production layout:
```
https://sarvam.androcoders.net ─ Cloudflare tunnel ─ andro (Caddy, Caddyfile-template)
  └─ 192.168.0.100:8090 (desktop-514csmm, port forward) ─ Tailscale ─ GPU host :8090
```

On the GPU host, run diarai as a service, adjusting `--host`/`--port` in the unit:
```bash
cp deploy/diarai.service /etc/systemd/system/ && systemctl daemon-reload && systemctl enable --now diarai
journalctl -u diarai -f
```

The unit binds to `127.0.0.1:8000` by default. Expose it only through a TLS front, or on a private network (LAN or Tailscale) behind one.

The web page also works over plain HTTP, falling back to ScriptProcessor capture, but keys would then cross the network unencrypted.

## Configuration (env or `.env`)

| Variable | Default | |
|---|---|---|
| `DIARAI_API_KEY_HASH` | (set by `keygen`) | Comma-separated scrypt hashes. `keygen --add` keeps the old keys. |
| `DIARAI_LANGUAGES` | `en,hi` | Language-ID candidates (`all` for any). |
| `DIARAI_ASR_MODEL` | `large-v3` | Default Whisper. |
| `DIARAI_HI_MODEL` | `models/indicwhisper-large-hi-ct2` | Hindi model. |
| `DIARAI_DEVICE` / `DIARAI_COMPUTE_TYPE` | `cuda` / `int8_float16` | |
| `DIARAI_MAX_STREAMS` | 4 | Concurrent WebSocket streams. |
| `DIARAI_IDLE_TIMEOUT_S` | 120 | |
| `DIARAI_MAX_UPLOAD_MB` | 1024 | |
| `DIARAI_HOST` / `DIARAI_PORT` | `127.0.0.1` / 8000 | |

**Security:**
* Keys are 384-bit random. Only salted scrypt hashes are stored, comparison is constant-time, and an IP is throttled after 20 failures in 5 minutes.
* Keys travel in headers. Serve over TLS (`--ssl-certfile/--ssl-keyfile`, or a reverse proxy) whenever the server is not bound to localhost.
* Keys are not accepted in URLs, to keep them out of logs. Browser clients therefore need a small proxy that adds the header.

## Install

```bash
uv venv -p 3.10 .venv && source .venv/bin/activate
git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen third_party/DiariZen
uv pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt -c third_party/DiariZen/constraints.txt
uv pip install -e third_party/DiariZen -e third_party/DiariZen/pyannote-audio -c third_party/DiariZen/constraints.txt
patchelf --clear-execstack .venv/lib/python3.10/site-packages/ctranslate2.libs/libctranslate2-*.so*  # glibc >= 2.41
scripts/setup_models.sh
pytest -q tests
```

* **Version pins:** DiariZen needs torch 2.1.1 (cuDNN 8), so ASR uses CTranslate2 4.4.0, the last release built against cuDNN 8.
* **GPU memory:** one Whisper large (int8) plus DiariZen fits in 4 GB.
* **Runs on startup:** the server runs as one process (`workers=1`) and keeps its models resident. Run it under systemd or similar for restarts.

## Licences

| Component | Licence |
|---|---|
| DiariZen weights (`BUT-FIT/diarizen-wavlm-large-s80-md`) | **CC BY-NC 4.0, non-commercial** |
| WeSpeaker ResNet34 | CC BY 4.0 |
| Whisper | MIT |
| IndicWhisper / Vistaar | MIT |

The DiariZen weights are non-commercial. For commercial use, replace the diarization model.
