"use strict";
// diarai web client: plays the chosen file, captures its PCM while it plays and streams it over the
// WebSocket in real time; transcript lines with speakers stream back. The card's border shows playback.

const $ = (s) => document.querySelector(s);
const els = {
  key: $("#key"), remember: $("#remember"), card: $("#card"), file: $("#file"), pick: $("#pick"),
  browse: $("#browse"), chosen: $("#chosen"), fname: $("#fname"), fdur: $("#fdur"), lang: $("#lang"),
  partials: $("#partials"), final: $("#final"), change: $("#change"), go: $("#go"), status: $("#status"),
  lines: $("#lines"), ttitle: $("#ttitle"), tools: $("#tools"), copy: $("#copy"), dltxt: $("#dltxt"),
  dljson: $("#dljson"), player: $("#player"),
  ringBase: document.querySelector(".ring-base"), ringFill: document.querySelector(".ring-fill"),
};

const KEY_STORE = "diarai.key";
let file = null, url = null;
let run = null; // {ws, ctx, src, node, pending, chunk, state, live:[], final:null}

// ------------------------------------------------------------------ key
try {
  const k = localStorage.getItem(KEY_STORE);
  if (k) { els.key.value = k; els.remember.checked = true; }
} catch (_) { /* storage unavailable */ }
function saveKey() {
  try {
    if (els.remember.checked) localStorage.setItem(KEY_STORE, els.key.value.trim());
    else localStorage.removeItem(KEY_STORE);
  } catch (_) {}
}
els.remember.addEventListener("change", saveKey);

// ------------------------------------------------------------- status/ring
function status(text, err = false) {
  els.status.textContent = text;
  els.status.classList.toggle("err", err);
}
let perimeter = 0;
function layoutRing() {
  const r = els.card.getBoundingClientRect();
  for (const rect of [els.ringBase, els.ringFill]) {
    rect.setAttribute("x", 1); rect.setAttribute("y", 1);
    rect.setAttribute("width", Math.max(0, r.width - 2)); rect.setAttribute("height", Math.max(0, r.height - 2));
    rect.setAttribute("rx", 13);
    rect.removeAttribute("pathLength");
  }
  perimeter = els.ringFill.getTotalLength ? els.ringFill.getTotalLength() : 2 * (r.width + r.height);
  els.ringFill.style.strokeDasharray = `${perimeter} ${perimeter}`;
  setProgress(lastProgress, false);
}
let lastProgress = 0;
function setProgress(p, animate = true) {
  lastProgress = Math.max(0, Math.min(1, p));
  els.ringFill.style.transition = animate && lastProgress > 0 ? "" : "none";
  els.ringFill.style.visibility = lastProgress > 0 ? "visible" : "hidden";
  els.ringFill.style.strokeDashoffset = String(perimeter * (1 - lastProgress));
}
new ResizeObserver(layoutRing).observe(els.card);

const fmt = (s) => {
  s = Math.max(0, Math.floor(s || 0));
  const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
};

// ------------------------------------------------------------------ file
function setFile(f) {
  if (!f) return;
  if (run && run.state !== "done") return;
  file = f;
  if (url) URL.revokeObjectURL(url);
  url = URL.createObjectURL(f);
  els.player.src = url;
  els.fname.textContent = f.name;
  els.fdur.textContent = "…";
  els.pick.hidden = true;
  els.chosen.hidden = false;
  els.card.classList.add("ready");
  setProgress(0);
  status("ready");
}
els.player.addEventListener("loadedmetadata", () => { els.fdur.textContent = fmt(els.player.duration); });
els.player.addEventListener("error", () => { if (file) status("this browser can't play that file", true); });
els.browse.addEventListener("click", (e) => { e.stopPropagation(); els.file.click(); });
els.pick.addEventListener("click", () => els.file.click());
els.change.addEventListener("click", () => els.file.click());
els.file.addEventListener("change", () => setFile(els.file.files[0]));
els.card.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && !file) els.file.click(); });
for (const ev of ["dragenter", "dragover"]) els.card.addEventListener(ev, (e) => { e.preventDefault(); els.card.classList.add("drag"); });
for (const ev of ["dragleave", "drop"]) els.card.addEventListener(ev, (e) => { e.preventDefault(); els.card.classList.remove("drag"); });
els.card.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

// ------------------------------------------------------------ transcript
function clearLines() {
  els.lines.replaceChildren();
  els.tools.hidden = true;
  els.ttitle.textContent = "Transcript";
}
function lineEl(spk, t, text, partial = false) {
  const div = document.createElement("div");
  div.className = "line" + (partial ? " partial" : "");
  const meta = document.createElement("div");
  meta.className = "meta";
  if (spk !== null) {
    const chip = document.createElement("span");
    chip.className = `spk s${Number(spk) % 4}`;
    chip.textContent = `Speaker ${Number(spk) + 1}`;
    meta.append(chip);
  }
  const ts = document.createElement("span");
  ts.className = "t";
  ts.textContent = fmt(t);
  meta.append(ts);
  const p = document.createElement("p");
  p.className = "text";
  p.textContent = text;
  div.append(meta, p);
  return div;
}
function partialLine() { return els.lines.querySelector(".line.partial"); }
function showPartial(t, text) {
  let el = partialLine();
  if (!text) { if (el) el.remove(); return; }
  const fresh = lineEl(null, t, text, true);
  if (el) el.replaceWith(fresh); else els.lines.append(fresh);
  follow();
}
function addLine(d) {
  const p = partialLine();
  const el = lineEl(d.speaker_id, d.start_time_seconds, d.transcript);
  // keep time order (segments of different speakers can finish together)
  const after = [...els.lines.querySelectorAll(".line:not(.partial)")].find(
    (x) => Number(x.dataset.t) > d.start_time_seconds);
  el.dataset.t = d.start_time_seconds;
  if (after) els.lines.insertBefore(el, after); else els.lines.insertBefore(el, p || null);
  follow();
}
function follow() {
  const nearBottom = window.innerHeight + window.scrollY > document.body.scrollHeight - 160;
  if (nearBottom) window.scrollTo({ top: document.body.scrollHeight });
}
function showFinal(d) {
  clearLines();
  els.ttitle.textContent = "Final transcript";
  for (const e of d.diarized_transcript.entries) {
    els.lines.append(lineEl(e.speaker_id, e.start_time_seconds, e.transcript));
  }
}
function entries() {
  if (run && run.final) return run.final.diarized_transcript.entries;
  return run ? run.live : [];
}
function asText() {
  return entries().map((e) => `[${fmt(e.start_time_seconds)}] Speaker ${Number(e.speaker_id) + 1}: ${e.transcript}`).join("\n");
}
function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
const base = () => (file ? file.name.replace(/\.[^.]+$/, "") : "transcript");
els.dltxt.addEventListener("click", () => download(base() + ".txt", asText(), "text/plain"));
els.dljson.addEventListener("click", () => download(base() + ".json",
  JSON.stringify(run && run.final ? run.final : { diarized_transcript: { entries: entries() } }, null, 1), "application/json"));
if (!navigator.clipboard) els.copy.hidden = true;
els.copy.addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(asText()); els.copy.textContent = "copied"; }
  catch (_) { els.copy.textContent = "copy failed"; }
  setTimeout(() => (els.copy.textContent = "copy"), 1500);
});

// --------------------------------------------------------------- stream
function setRunning(on) {
  els.go.textContent = on ? "Stop" : "Start";
  els.change.hidden = on;
  for (const el of [els.lang, els.partials, els.final, els.key]) el.disabled = on;
}

async function start() {
  const key = els.key.value.trim();
  if (!key) { status("paste your API key first", true); els.key.focus(); return; }
  if (!/^[A-Za-z0-9_\-]+$/.test(key)) { status("that doesn't look like a diarai key", true); return; }
  if (!file) return;
  saveKey();
  clearLines();
  setProgress(0);

  let ctx;
  try { ctx = new AudioContext({ sampleRate: 16000 }); } catch (_) { ctx = new AudioContext(); }
  const player = new Audio(url); // fresh element: a media element can feed only one AudioContext
  const r = { ws: null, ctx, player, src: null, node: null, sink: null, pending: [], pendingLen: 0,
              chunk: Math.round(ctx.sampleRate / 10), state: "connecting", live: [], final: null };
  run = r;
  setRunning(true);
  status("connecting…");

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const q = new URLSearchParams({
    "language-code": els.lang.value, partial_results: String(els.partials.checked),
    final_transcript: String(els.final.checked), input_audio_codec: "pcm_s16le", sample_rate: String(ctx.sampleRate),
  });
  const ws = new WebSocket(`${proto}://${location.host}/speech-to-text/ws?${q}`, ["diarai", "key." + key]);
  ws.binaryType = "arraybuffer";
  r.ws = ws;

  ws.onmessage = (ev) => onMessage(r, JSON.parse(ev.data));
  ws.onclose = (ev) => {
    if (r.state === "done") return;
    stopAudio(r);
    r.state = "done";
    setRunning(false);
    els.card.classList.remove("busy");
    if (ev.code === 1008) status(ev.reason || "API key rejected", true);
    else if (ev.code === 1013) status("server busy, try again shortly", true);
    else status(`disconnected${ev.reason ? ": " + ev.reason : ` (${ev.code})`}`, true);
    if (r.live.length) els.tools.hidden = false;
  };
  // audio starts only once the server has accepted the key (READY), not on socket open
  r.begin = async () => {
    try {
      r.src = ctx.createMediaElementSource(player);
      if (ctx.audioWorklet && window.isSecureContext) {
        await ctx.audioWorklet.addModule("/static/capture.js");
        r.node = new AudioWorkletNode(ctx, "capture");
        r.node.port.onmessage = (e) => push(r, e.data);
      } else { // plain-http origins: AudioWorklet needs a secure context
        r.node = ctx.createScriptProcessor(4096, 1, 1);
        r.node.onaudioprocess = (e) => push(r, new Float32Array(e.inputBuffer.getChannelData(0)));
      }
      r.sink = ctx.createGain();
      r.sink.gain.value = 0;
      r.src.connect(r.node);
      r.node.connect(r.sink).connect(ctx.destination);
      r.src.connect(ctx.destination); // you hear what is being transcribed
      await ctx.resume();
      player.addEventListener("ended", () => finish(r));
      await player.play();
      r.state = "streaming";
      els.card.classList.remove("busy");
      tick(r);
    } catch (err) {
      if (r.state === "done") return;
      status("audio error: " + err.message, true);
      ws.close();
    }
  };
  ws.onopen = () => status("authorising…");
}

function push(r, f32) {
  if (r.state !== "streaming") return;
  r.pending.push(f32);
  r.pendingLen += f32.length;
  if (r.pendingLen >= r.chunk) sendPending(r);
}
function sendPending(r) {
  if (!r.pendingLen || r.ws.readyState !== WebSocket.OPEN) return;
  const pcm = new Int16Array(r.pendingLen);
  let o = 0;
  for (const b of r.pending) {
    for (let i = 0; i < b.length; i++) {
      const v = Math.max(-1, Math.min(1, b[i]));
      pcm[o++] = v < 0 ? v * 32768 : v * 32767;
    }
  }
  r.pending = []; r.pendingLen = 0;
  r.ws.send(pcm.buffer);
}
function tick(r) {
  if (r.state !== "streaming") return;
  const d = r.player.duration || 0;
  setProgress(d ? r.player.currentTime / d : 0);
  status(`streaming ${fmt(r.player.currentTime)} / ${fmt(d)}`);
  requestAnimationFrame(() => tick(r));
}
function stopAudio(r) {
  r.player.pause();
  try { r.src && r.src.disconnect(); r.node && r.node.disconnect(); } catch (_) {}
  if (r.ctx && r.ctx.state !== "closed") r.ctx.close().catch(() => {});
}
function finish(r) {
  if (r.state !== "streaming") return;
  sendPending(r);
  r.state = "finishing";
  stopAudio(r);
  setProgress(1);
  els.card.classList.add("busy");
  status(els.final.checked ? "finishing: re-processing the whole recording…" : "finishing…");
  if (r.ws.readyState === WebSocket.OPEN) r.ws.send(JSON.stringify({ type: "end" }));
}

function onMessage(r, m) {
  const d = m.data || {};
  if (m.type === "data") {
    r.live.push({ speaker_id: d.speaker_id, start_time_seconds: d.start_time_seconds,
                  end_time_seconds: d.end_time_seconds, transcript: d.transcript });
    if (els.lines.querySelector(".empty")) els.lines.replaceChildren();
    addLine(d);
  } else if (m.type === "partial") {
    if (els.lines.querySelector(".empty")) els.lines.replaceChildren();
    showPartial(d.start_time_seconds, d.transcript);
  } else if (m.type === "final") {
    r.final = d;
    showFinal(d);
  } else if (m.type === "events" && d.signal_type === "READY") {
    if (r.state === "connecting") r.begin();
  } else if (m.type === "events" && d.signal_type === "END_OF_STREAM") {
    r.state = "done";
    showPartial(0, "");
    setRunning(false);
    els.card.classList.remove("busy");
    els.tools.hidden = !entries().length;
    const lang = r.final ? r.final.language_code : "";
    status(`done${lang ? " · " + lang : ""}`);
    r.ws.close();
  } else if (m.type === "error") {
    status(d.error || "server error", true);
  }
}

els.go.addEventListener("click", () => {
  if (run && (run.state === "streaming" || run.state === "connecting")) {
    if (run.state === "connecting") { run.ws.close(); return; }
    finish(run);
  } else if (!run || run.state === "done") {
    start();
  }
});
