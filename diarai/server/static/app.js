"use strict";
// diarai web client. Files are queued and streamed one at a time (one GPU): each file is decoded by a muted
// player whose PCM goes over the WebSocket in real time, while transcript lines with speakers stream back
// into that file's own section. You hear the file (a second, audible player) from its first returned text
// on, so sound and text stay roughly in step. The card's border shows how much of the current file is sent.

const $ = (s) => document.querySelector(s);
const els = {
  key: $("#key"), remember: $("#remember"), card: $("#card"), file: $("#file"), pick: $("#pick"),
  browse: $("#browse"), chosen: $("#chosen"), fname: $("#fname"), fdur: $("#fdur"), queue: $("#queue"),
  lang: $("#lang"), partials: $("#partials"), final: $("#final"), add: $("#add"), go: $("#go"),
  status: $("#status"), jobsEl: $("#jobs"), apiprompt: $("#apiprompt"), player: $("#player"),
  ringBase: document.querySelector(".ring-base"), ringFill: document.querySelector(".ring-fill"),
};

const KEY_STORE = "diarai.key";
const jobs = [];      // {id, file, url, name, dur, state, live, final, el, linesEl, statusEl, toolsEl, removeEl}
let current = null;   // job being streamed
let running = false;  // queue mode: start the next queued job when one finishes
let seq = 0;

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
let perimeter = 0, lastProgress = 0;
function layoutRing() {
  const r = els.card.getBoundingClientRect();
  for (const rect of [els.ringBase, els.ringFill]) {
    rect.setAttribute("x", 1); rect.setAttribute("y", 1);
    rect.setAttribute("width", Math.max(0, r.width - 2)); rect.setAttribute("height", Math.max(0, r.height - 2));
    rect.setAttribute("rx", 13);
  }
  perimeter = els.ringFill.getTotalLength ? els.ringFill.getTotalLength() : 2 * (r.width + r.height);
  els.ringFill.style.strokeDasharray = `${perimeter} ${perimeter}`;
  setProgress(lastProgress, false);
}
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

// --------------------------------------------------------------- clipboard
async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try { await navigator.clipboard.writeText(text); return true; } catch (_) { /* fall through */ }
  }
  const ta = document.createElement("textarea"); // plain-http origins have no async clipboard
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed"; ta.style.opacity = "0";
  document.body.append(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch (_) {}
  ta.remove();
  return ok;
}
function flash(btn, ok, label) {
  btn.textContent = ok ? "copied" : "copy failed";
  setTimeout(() => (btn.textContent = label), 1500);
}

// ------------------------------------------------------------ API prompt
function apiPrompt() {
  const o = location.origin, ws = o.replace(/^http/, "ws");
  return `diarai is a self-hosted speech-to-text API with speaker diarization (who said what) for Hindi and English, with output in Sarvam AI's speech-to-text format. Base URL: ${o}; interactive docs at ${o}/docs and the OpenAPI schema at ${o}/openapi.json; GET ${o}/health needs no key. Every other request needs an API key in the header \`Api-Subscription-Key: <key>\` (or \`Authorization: Bearer <key>\`). The key is not included here: ask the user to paste it, keep it in an environment variable or secret store (e.g. DIARAI_API_KEY), and never put it in code, URLs or logs.

For a recording, POST ${o}/speech-to-text as multipart/form-data with \`file\` (any audio or video, including 8 kHz call audio) and optional \`language_code\` (hi-IN, en-IN, or unknown to auto-detect), \`num_speakers\` (e.g. 2 for a phone call), \`with_timestamps\` and \`with_diarization\` (both default true). The JSON response has \`transcript\`, \`diarized_transcript.entries[]\` (each {transcript, start_time_seconds, end_time_seconds, speaker_id}), \`language_code\` and \`metrics\`. It runs on one GPU at roughly real time, so send files one at a time and allow a timeout of several minutes. For live audio, open ${ws}/speech-to-text/ws?language-code=hi-IN&input_audio_codec=pcm_s16le&sample_rate=16000 with the same header (browsers pass the key as WebSocket subprotocols ["diarai", "key.<key>"]), wait for the READY event, send 16-bit mono PCM as binary frames in real time, then send {"type":"end"}; the server replies with JSON messages of type data (a finished line with its speaker), partial, events (READY, END_OF_STREAM) and final (the whole result in the batch format).`;
}
els.apiprompt.addEventListener("click", async () => flash(els.apiprompt, await copyText(apiPrompt()), "copy API prompt"));

// ------------------------------------------------------------------ files
function addFiles(list) {
  const fresh = [...list].filter((f) => f && f.size);
  if (!fresh.length) return;
  if (!jobs.length) els.jobsEl.replaceChildren();
  for (const f of fresh) {
    const job = { id: ++seq, file: f, url: URL.createObjectURL(f), name: f.name, dur: 0, state: "queued",
                  live: [], final: null };
    buildSection(job);
    jobs.push(job);
    const probe = new Audio();
    probe.preload = "metadata";
    probe.addEventListener("loadedmetadata", () => { job.dur = probe.duration; jobStatus(job); renderCard(); });
    probe.addEventListener("error", () => { if (job.state === "queued") jobStatus(job, "this browser can't play that file", true); });
    probe.src = job.url;
  }
  renderCard();
  if (!running) status(`${queued().length} file${queued().length === 1 ? "" : "s"} ready`);
}
const queued = () => jobs.filter((j) => j.state === "queued");

function renderCard() {
  const show = current || queued()[0];
  els.pick.hidden = !!show;
  els.chosen.hidden = !show;
  els.card.classList.toggle("ready", !!show);
  if (!show) { setProgress(0); return; }
  els.fname.textContent = show.name;
  els.fdur.textContent = show.dur ? fmt(show.dur) : "…";
  const rest = queued().filter((j) => j !== show).length;
  els.queue.hidden = !rest;
  els.queue.textContent = rest ? `+ ${rest} more queued, transcribed one after another` : "";
}

els.browse.addEventListener("click", (e) => { e.stopPropagation(); els.file.click(); });
els.pick.addEventListener("click", () => els.file.click());
els.add.addEventListener("click", () => els.file.click());
els.file.addEventListener("change", () => { addFiles(els.file.files); els.file.value = ""; });
els.card.addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && e.target === els.card) { e.preventDefault(); els.file.click(); }
});
for (const ev of ["dragenter", "dragover"]) els.card.addEventListener(ev, (e) => { e.preventDefault(); els.card.classList.add("drag"); });
for (const ev of ["dragleave", "drop"]) els.card.addEventListener(ev, (e) => { e.preventDefault(); els.card.classList.remove("drag"); });
els.card.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

// --------------------------------------------------------- job sections
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function buildSection(job) {
  const sec = el("section", "job");
  const head = el("div", "thead");
  const titles = el("div", "jhead");
  titles.append(el("p", "jname", job.name), (job.statusEl = el("span", "jstatus", "queued")));
  const tools = el("div", "tools");
  const copy = el("button", "link", "copy"), txt = el("button", "link", ".txt"), json = el("button", "link", ".json");
  const remove = el("button", "link", "remove");
  for (const b of [copy, txt, json, remove]) b.type = "button";
  job.toolsEls = [copy, txt, json];
  for (const b of job.toolsEls) b.hidden = true;
  tools.append(copy, txt, json, remove);
  head.append(titles, tools);
  job.linesEl = el("div", "lines");
  job.linesEl.append(el("p", "empty muted", "Waiting for its turn."));
  sec.append(head, job.linesEl);
  job.el = sec;
  job.removeEl = remove;

  const base = job.name.replace(/\.[^.]+$/, "") || "transcript";
  copy.addEventListener("click", async () => flash(copy, await copyText(asText(job)), "copy"));
  txt.addEventListener("click", () => download(base + ".txt", asText(job), "text/plain"));
  json.addEventListener("click", () => download(base + ".json",
    JSON.stringify(job.final || { diarized_transcript: { entries: job.live } }, null, 1), "application/json"));
  remove.addEventListener("click", () => {
    if (job === current) return;
    jobs.splice(jobs.indexOf(job), 1);
    URL.revokeObjectURL(job.url);
    sec.remove();
    if (!jobs.length) els.jobsEl.append(el("p", "empty muted", "Each file's text appears here."));
    renderCard();
  });
  els.jobsEl.append(sec);
}
function jobStatus(job, text, err = false) {
  if (text === undefined) text = job.state === "queued" ? `queued${job.dur ? " · " + fmt(job.dur) : ""}` : job.statusEl.textContent;
  job.statusEl.textContent = text;
  job.statusEl.classList.toggle("err", err);
}
function showTools(job) {
  const has = !!entries(job).length;
  for (const b of job.toolsEls) b.hidden = !has;
  job.removeEl.hidden = job === current;
}

// ------------------------------------------------------------ transcript
function lineEl(spk, t, text, partial = false) {
  const div = el("div", "line" + (partial ? " partial" : ""));
  const meta = el("div", "meta");
  if (spk !== null) meta.append(el("span", `spk s${Number(spk) % 4}`, `Speaker ${Number(spk) + 1}`));
  meta.append(el("span", "t", fmt(t)));
  div.append(meta, el("p", "text", text));
  return div;
}
function clearEmpty(job) { const e = job.linesEl.querySelector(".empty"); if (e) e.remove(); }
function showPartial(job, t, text) {
  const old = job.linesEl.querySelector(".line.partial");
  if (!text) { if (old) old.remove(); return; }
  clearEmpty(job);
  const fresh = lineEl(null, t, text, true);
  if (old) old.replaceWith(fresh); else job.linesEl.append(fresh);
  follow();
}
function addLine(job, d) {
  clearEmpty(job);
  const p = job.linesEl.querySelector(".line.partial");
  const line = lineEl(d.speaker_id, d.start_time_seconds, d.transcript);
  line.dataset.t = d.start_time_seconds;
  // keep time order (segments of different speakers can finish together)
  const after = [...job.linesEl.querySelectorAll(".line:not(.partial)")].find((x) => Number(x.dataset.t) > d.start_time_seconds);
  job.linesEl.insertBefore(line, after || p || null);
  follow();
}
function follow() {
  const nearBottom = window.innerHeight + window.scrollY > document.body.scrollHeight - 160;
  if (nearBottom) window.scrollTo({ top: document.body.scrollHeight });
}
function showFinal(job) {
  job.linesEl.replaceChildren();
  for (const e of job.final.diarized_transcript.entries) job.linesEl.append(lineEl(e.speaker_id, e.start_time_seconds, e.transcript));
  if (!job.final.diarized_transcript.entries.length) job.linesEl.append(el("p", "empty muted", "No speech found."));
}
const entries = (job) => (job.final ? job.final.diarized_transcript.entries : job.live);
const asText = (job) => entries(job).map((e) => `[${fmt(e.start_time_seconds)}] Speaker ${Number(e.speaker_id) + 1}: ${e.transcript}`).join("\n");
function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// --------------------------------------------------------------- queue
function setRunning(on) {
  running = on;
  els.go.textContent = on ? "Stop" : "Start";
  for (const e of [els.lang, els.partials, els.final, els.key]) e.disabled = on;
}
function startQueue() {
  const key = els.key.value.trim();
  if (!key) { status("paste your API key first", true); els.key.focus(); return; }
  if (!/^[A-Za-z0-9_\-]+$/.test(key)) { status("that doesn't look like a diarai key", true); return; }
  if (!queued().length) return;
  saveKey();
  if (current) { setRunning(true); return; } // a stopped file is still finishing; the queue resumes after it
  // unlock delayed playback while we still hold the click's user activation
  const p = els.player;
  p.src = queued()[0].url;
  p.muted = true;
  p.play().then(() => { p.pause(); p.muted = false; }).catch(() => { p.muted = false; });
  setRunning(true);
  next();
}
function next() {
  const job = running ? queued()[0] : null;
  const prev = current;
  current = null;
  if (prev) showTools(prev);
  if (!job) {
    setRunning(false);
    renderCard();
    const done = jobs.filter((j) => j.state === "done").length;
    if (!queued().length && done) status(`done · ${done} file${done === 1 ? "" : "s"}`);
    return;
  }
  startJob(job, els.key.value.trim());
}

async function startJob(job, key) {
  current = job;
  job.state = "connecting";
  job.live = []; job.final = null; job.heard = false;
  job.linesEl.replaceChildren(el("p", "empty muted", "Listening… the audio plays once the first text comes back."));
  for (const j of jobs) j.el.classList.toggle("active", j === job);
  showTools(job);
  renderCard();
  setProgress(0);
  els.card.classList.add("busy");
  jobStatus(job, "connecting…");
  status(`connecting · ${job.name}`);

  let ctx;
  try { ctx = new AudioContext({ sampleRate: 16000 }); } catch (_) { ctx = new AudioContext(); }
  const capture = new Audio(job.url); // muted source: only feeds the stream, never the speakers
  const r = { job, ws: null, ctx, capture, src: null, node: null, pending: [], pendingLen: 0,
              chunk: Math.round(ctx.sampleRate / 10) };
  job.run = r;

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const q = new URLSearchParams({
    "language-code": els.lang.value, partial_results: String(els.partials.checked),
    final_transcript: String(els.final.checked), input_audio_codec: "pcm_s16le", sample_rate: String(ctx.sampleRate),
  });
  const ws = new WebSocket(`${proto}://${location.host}/speech-to-text/ws?${q}`, ["diarai", "key." + key]);
  ws.binaryType = "arraybuffer";
  r.ws = ws;
  ws.onopen = () => jobStatus(job, "authorising…");
  ws.onmessage = (ev) => onMessage(r, JSON.parse(ev.data));
  ws.onclose = (ev) => {
    if (job.state === "done" || job.state === "failed") return;
    stopCapture(r);
    job.state = "failed";
    els.card.classList.remove("busy");
    let msg = `disconnected${ev.reason ? ": " + ev.reason : ` (${ev.code})`}`;
    const retry = (ev.code === 1008 || ev.code === 1013) && !job.live.length;
    if (ev.code === 1008) msg = ev.reason || "API key rejected";
    else if (ev.code === 1013) msg = "server busy, try again shortly";
    status(msg, true);
    job.el.classList.remove("active");
    if (retry) { // nothing was transcribed: keep it queued and stop, so Start retries it
      job.state = "queued";
      jobStatus(job, `${msg} · still queued`, true);
      job.linesEl.replaceChildren(el("p", "empty muted", "Waiting for its turn."));
      setRunning(false);
    } else {
      job.state = job.live.length ? "done" : "failed";
      jobStatus(job, msg, true);
      if (!job.live.length) job.linesEl.replaceChildren(el("p", "empty muted", "Not transcribed."));
    }
    showTools(job);
    next();
  };
  // capture starts only once the server has accepted the key (READY), not on socket open
  r.begin = async () => {
    try {
      r.src = ctx.createMediaElementSource(capture);
      if (ctx.audioWorklet && window.isSecureContext) {
        await ctx.audioWorklet.addModule("/static/capture.js");
        r.node = new AudioWorkletNode(ctx, "capture");
        r.node.port.onmessage = (e) => push(r, e.data);
      } else { // plain-http origins: AudioWorklet needs a secure context
        r.node = ctx.createScriptProcessor(4096, 1, 1);
        r.node.onaudioprocess = (e) => push(r, new Float32Array(e.inputBuffer.getChannelData(0)));
      }
      const sink = ctx.createGain();
      sink.gain.value = 0; // keeps the graph pulling audio without anything reaching the speakers
      r.src.connect(r.node);
      r.node.connect(sink).connect(ctx.destination);
      await ctx.resume();
      capture.addEventListener("ended", () => finish(r));
      await capture.play();
      job.state = "streaming";
      els.card.classList.remove("busy");
      tick(r);
    } catch (err) {
      if (job.state === "done") return;
      jobStatus(job, "audio error: " + err.message, true);
      ws.close();
    }
  };
}

function push(r, f32) {
  if (r.job.state !== "streaming") return;
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
  if (r.job.state !== "streaming") return;
  const d = r.capture.duration || 0;
  setProgress(d ? r.capture.currentTime / d : 0);
  const t = `sending ${fmt(r.capture.currentTime)} / ${fmt(d)}${r.job.heard ? "" : " · waiting for first text"}`;
  jobStatus(r.job, t);
  status(`${t} · ${r.job.name}`);
  requestAnimationFrame(() => tick(r));
}
function stopCapture(r) {
  r.capture.pause();
  try { r.src && r.src.disconnect(); r.node && r.node.disconnect(); } catch (_) {}
  if (r.ctx && r.ctx.state !== "closed") r.ctx.close().catch(() => {});
}
function finish(r) {
  if (r.job.state !== "streaming") return;
  sendPending(r);
  r.job.state = "finishing";
  stopCapture(r);
  setProgress(1);
  els.card.classList.add("busy");
  const t = els.final.checked ? "finishing: re-processing the whole recording…" : "finishing…";
  jobStatus(r.job, t);
  status(`${t} · ${r.job.name}`);
  if (r.ws.readyState === WebSocket.OPEN) r.ws.send(JSON.stringify({ type: "end" }));
}
// audible playback of this job, started by its first transcript text
function hear(job) {
  if (job.heard) return;
  job.heard = true;
  const p = els.player;
  p.src = job.url;
  p.currentTime = 0;
  p.play().catch(() => jobStatus(job, "text is streaming; your browser blocked playback"));
}

function onMessage(r, m) {
  const job = r.job, d = m.data || {};
  if (m.type === "data") {
    job.live.push({ speaker_id: d.speaker_id, start_time_seconds: d.start_time_seconds,
                    end_time_seconds: d.end_time_seconds, transcript: d.transcript });
    addLine(job, d);
    showTools(job);
    hear(job);
  } else if (m.type === "partial") {
    showPartial(job, d.start_time_seconds, d.transcript);
    if (d.transcript) hear(job);
  } else if (m.type === "final") {
    job.final = d;
    showFinal(job);
  } else if (m.type === "events" && d.signal_type === "READY") {
    if (job.state === "connecting") r.begin();
  } else if (m.type === "events" && d.signal_type === "END_OF_STREAM") {
    job.state = "done";
    showPartial(job, 0, "");
    els.card.classList.remove("busy");
    job.el.classList.remove("active");
    showTools(job);
    const lang = job.final ? job.final.language_code : "";
    jobStatus(job, `done${lang ? " · " + lang : ""}${job.dur ? " · " + fmt(job.dur) : ""}`);
    r.ws.close();
    next();
  } else if (m.type === "error") {
    jobStatus(job, d.error || "server error", true);
  }
}

els.go.addEventListener("click", () => {
  if (!running) { startQueue(); return; }
  // Stop: end the current file (its text so far is kept) and leave the rest queued
  setRunning(false);
  els.player.pause();
  const r = current && current.run;
  if (!r) return;
  if (current.state === "connecting") { r.ws.close(); return; }
  finish(r);
});
