"use strict";
// zdial benchmark view (door button). Reads /bench/* with the API key from the main page; polls while the run is live.
(() => {
  const $ = (s) => document.querySelector(s);
  const home = $("#home"), view = $("#bench"), door = $("#door"), back = $("#benchback");
  const statsEl = $("#bstats"), listEl = $("#blist"), filterEl = $("#bfilter"), needKey = $("#bneedkey");
  let rows = [], status = null, runs = {}, filter = "all", open = null, timer = null, audioUrl = null;

  const key = () => $("#key").value.trim();
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  };
  const fmt = (s) => {
    s = Math.max(0, Math.round(Number(s) || 0));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };
  const pct = (a, b) => (b ? Math.round((100 * a) / b) + "%" : "–");
  const num = (x) => typeof x === "number" && isFinite(x);
  const secs = (x) => (num(x) ? (x < 90 ? `${Math.round(x * 10) / 10} s` : fmt(x)) : "–");
  const median = (a) => {
    const b = a.filter(num).sort((x, y) => x - y);
    return b.length ? (b.length % 2 ? b[(b.length - 1) / 2] : (b[b.length / 2 - 1] + b[b.length / 2]) / 2) : null;
  };
  // hang-up -> analysis ready. Sarvam: zdial's own timestamps (live transcripts are ready at hang-up, so this is
  // the analysis). diarai: its transcription time + the same analysis time (estimate, marked with ≈).
  const diaraiReady = (r) => (num(r.diarai_seconds) ? r.diarai_seconds + (num(r.sarvam_ready_seconds) && r.sarvam_mode === "live" ? r.sarvam_ready_seconds : 0) : null);

  async function api(path, asBlob = false) {
    const r = await fetch(path, { headers: { "Api-Subscription-Key": key() }, cache: "no-store" });
    if (!r.ok) throw new Error(r.status === 401 ? "API key rejected" : `HTTP ${r.status}`);
    return asBlob ? r.blob() : r.json();
  }

  // ------------------------------------------------------------ open / close
  function show(on) {
    home.hidden = on;
    view.hidden = !on;
    door.setAttribute("aria-pressed", String(on));
    clearTimeout(timer);
    if (on) { window.scrollTo({ top: 0 }); refresh(); }
    else if (audioUrl) { URL.revokeObjectURL(audioUrl); audioUrl = null; }
    try { history.replaceState(null, "", on ? "#bench" : location.pathname); } catch (_) {}
  }
  door.addEventListener("click", () => show(view.hidden));
  back.addEventListener("click", () => show(false));
  if (location.hash === "#bench") setTimeout(() => show(true), 0);

  async function refresh() {
    clearTimeout(timer);
    if (view.hidden) return;
    needKey.hidden = !!key();
    if (!key()) return;
    try {
      const d = await api("/bench/summary");
      rows = d.calls || [];
      status = d.status;
      runs = d.runs || {};
      render();
    } catch (e) {
      statsEl.replaceChildren(el("p", "muted small", "Could not load the benchmark: " + e.message));
    }
    if (Object.values(runs).some((x) => x && (x.state === "running" || x.state === "starting"))) timer = setTimeout(refresh, 10000);
  }

  // ---------------------------------------------------------------- stats
  // second engine: diarai's diarization + maya-transcribe for the text, on the same recordings
  function mayaStats(ok) {
    const run = runs.maya;
    const m = ok.filter((r) => r.maya && r.maya.verdict);
    if (!m.length) {
      return [stat("–", "maya-transcribe vs Sarvam", run ? `${run.state} · ${run.done || 0}/${run.total || "–"}` : "not run yet")];
    }
    const vm = tally(m.map((r) => ({ v: r.maya.verdict })), "v");
    const vd = tally(m, "verdict"); // diarai on the same subset, for a fair side-by-side
    const sc = m.filter((r) => r.script[0] || r.maya.script);
    const sameSc = sc.filter((r) => r.script[0] === r.maya.script).length;
    const st = m.reduce((a, r) => { const x = r.maya.steps; if (x) { a[0] += x.agree; a[1] += x.total; } return a; }, [0, 0]);
    const audio = m.reduce((a, r) => a + (r.duration || 0), 0), took = m.reduce((a, r) => a + (r.maya.seconds || 0), 0);
    const ready = median(m.map((r) => (num(r.maya.seconds) ? r.maya.seconds + (r.sarvam_mode === "live" && num(r.sarvam_ready_seconds) ? r.sarvam_ready_seconds : 0) : null)));
    const state = run ? (run.state === "running" ? `running · ${run.done}/${run.total}` : run.state) : "";
    return [
      stat(pct(vm.same + vm.minor, m.length), "same conclusion (maya vs Sarvam)",
        `same ${vm.same} · minor ${vm.minor} · different ${vm.different} · ${m.length} calls${state ? " · " + state : ""}`, verdictBar(vm)),
      stat(`${pct(vd.same + vd.minor, m.length)} → ${pct(vm.same + vm.minor, m.length)}`, "same conclusion: diarai → maya",
        `different: diarai ${vd.different} · maya ${vm.different} (same ${m.length} calls)`),
      stat(pct(sameSc, sc.length), "same script picked (maya)", `${sameSc} of ${sc.length} graded · steps ${pct(st[0], st[1])}`),
      stat(audio ? `${(took / audio).toFixed(2)}×` : "–", "maya transcription time / call length",
        `≈${secs(ready)} hang-up to analysis (median)`),
    ];
  }

  function speedText(list) {
    const a = list.filter((r) => num(r.diarai_seconds) && num(r.duration) && r.duration > 0);
    const audio = a.reduce((x, r) => x + r.duration, 0), took = a.reduce((x, r) => x + r.diarai_seconds, 0);
    return audio ? `${(took / audio).toFixed(2)}×` : "–";
  }
  function totalText(list) {
    const a = list.filter((r) => num(r.diarai_seconds) && num(r.duration));
    const m = (x) => Math.round(x / 60);
    return a.length ? `${m(a.reduce((x, r) => x + r.diarai_seconds, 0))} min for ${m(a.reduce((x, r) => x + r.duration, 0))} min of audio` : "";
  }
  function tally(list, field) {
    const t = { same: 0, minor: 0, different: 0 };
    for (const r of list) if (r[field] in t) t[r[field]]++;
    return t;
  }
  function verdictBar(t) {
    const n = t.same + t.minor + t.different;
    const bar = el("div", "bar");
    for (const k of ["same", "minor", "different"]) {
      const i = el("i", k);
      i.style.width = n ? `${(100 * t[k]) / n}%` : "0";
      i.title = `${k}: ${t[k]}`;
      bar.append(i);
    }
    return bar;
  }
  function stat(value, label, note, extra) {
    const s = el("div", "stat");
    s.append(el("div", "v", value), el("div", "l", label));
    if (note) s.append(el("div", "n", note));
    if (extra) s.append(extra);
    return s;
  }
  function render() {
    const ok = rows.filter((r) => !r.error && r.verdict);
    const failed = rows.filter((r) => r.error);
    const total = status ? status.total : rows.length;
    const v = tally(ok, "verdict"), nz = tally(ok, "noise");
    const scriptRows = ok.filter((r) => r.script[0] || r.script[1]);
    const sameScript = scriptRows.filter((r) => r.script[0] === r.script[1]).length;
    const deltas = ok.filter((r) => num(r.score[0]) && num(r.score[1])).map((r) => Math.abs(r.score[0] - r.score[1]));
    const sum = (k) => ok.reduce((a, r) => { const s = r[k]; if (s) { a[0] += s.agree; a[1] += s.total; } return a; }, [0, 0]);
    const st = sum("steps"), stn = sum("steps_noise");
    const state = status ? status.state : "no run yet";
    const live = state === "running" ? `running · now ${String(status.current || "").slice(0, 8)}` : state;

    statsEl.replaceChildren(
      stat(`${ok.length} / ${total || "–"}`, "calls compared", `${live}${failed.length ? ` · ${failed.length} failed` : ""}`),
      stat(pct(v.same + v.minor, ok.length), "same conclusion (diarai vs Sarvam)",
        `same ${v.same} · minor ${v.minor} · different ${v.different}`, verdictBar(v)),
      stat(pct(nz.same + nz.minor, ok.length), "noise: Sarvam vs Sarvam again",
        `same ${nz.same} · minor ${nz.minor} · different ${nz.different}`, verdictBar(nz)),
      stat(pct(sameScript, scriptRows.length), "same script picked", `${sameScript} of ${scriptRows.length} graded calls`),
      stat(deltas.length ? (deltas.reduce((a, b) => a + b, 0) / deltas.length).toFixed(1) : "–", "mean |score difference| /10",
        `${deltas.length} calls with both scores`),
      stat(pct(st[0], st[1]), "script steps graded the same", `${st[0]}/${st[1]} · noise ${pct(stn[0], stn[1])}`),
      stat(`${secs(median(ok.map((r) => r.sarvam_ready_seconds)))} → ≈${secs(median(ok.map(diaraiReady)))}`,
        "hang-up to analysis ready, median: Sarvam → diarai",
        `Sarvam transcribed live in ${ok.filter((r) => r.sarvam_mode === "live").length}/${ok.length} calls`),
      stat(speedText(ok), "diarai transcription time / call length", totalText(ok)),
      ...mayaStats(ok),
    );

    filterEl.replaceChildren();
    const counts = { all: rows.length, different: v.different, minor: v.minor, same: v.same, failed: failed.length,
                     "maya different": ok.filter((r) => r.maya && r.maya.verdict === "different").length };
    for (const k of Object.keys(counts)) {
      const b = el("button", "chip", `${k} ${counts[k]}`);
      b.type = "button";
      b.setAttribute("aria-pressed", String(filter === k));
      b.addEventListener("click", () => { filter = k; render(); });
      filterEl.append(b);
    }

    const order = { different: 0, minor: 1, same: 2 };
    const shown = rows
      .filter((r) => filter === "all" || (filter === "failed" ? r.error
        : filter === "maya different" ? r.maya && r.maya.verdict === "different" : r.verdict === filter))
      .sort((a, b) => (a.error ? 3 : order[a.verdict] ?? 3) - (b.error ? 3 : order[b.verdict] ?? 3));
    listEl.replaceChildren();
    if (!shown.length) listEl.append(el("p", "muted small", rows.length ? "No calls in this filter." : "No results yet."));
    for (const r of shown) {
      const row = el("button", "brow");
      row.type = "button";
      const verdict = r.error ? "failed" : r.verdict || "pending";
      const mid = el("div");
      mid.append(el("div", "id", r.id.slice(0, 8)),
        el("div", "sub", r.error ? r.error : `${r.direction || ""} · ${fmt(r.duration)} · ${r.speakers ?? "?"} speakers · noise ${r.noise || "–"}`),
        r.error ? "" : el("div", "sub", `ready after hang-up: Sarvam ${secs(r.sarvam_ready_seconds)}${r.sarvam_mode === "live" ? " (live)" : ""} · diarai ≈${secs(diaraiReady(r))} (transcribing ${secs(r.diarai_seconds)})`));
      const sc = r.script[0] || r.script[1]
        ? `${r.script[0] || "none"} ${num(r.score[0]) ? r.score[0] : ""} → ${r.script[1] || "none"} ${num(r.score[1]) ? r.score[1] : ""}`
        : `${(r.sentiment[0] || "").toLowerCase()} → ${(r.sentiment[1] || "").toLowerCase()}`;
      const chips = el("div", "chips");
      chips.append(el("span", `verdict ${verdict}`, verdict));
      if (r.maya) {
        const mv = r.maya.error ? "failed" : r.maya.verdict || "pending";
        const c = el("span", `verdict small ${mv}`, `maya ${mv}`);
        c.title = "maya-transcribe vs Sarvam";
        chips.append(c);
      }
      row.append(chips, mid, el("div", "right", sc));
      row.setAttribute("aria-expanded", String(open === r.id));
      row.addEventListener("click", () => toggle(r.id));
      listEl.append(row);
      if (open === r.id) listEl.append(detailHolder(r.id));
    }
  }

  // --------------------------------------------------------------- detail
  const details = new Map();
  let openBox = null; // the open call's panel survives re-renders, so polling never restarts its audio
  function detailHolder(id) {
    if (openBox && openBox.dataset.id === id) return openBox;
    const box = el("div", "bdetail");
    box.dataset.id = id;
    const d = details.get(id);
    if (d) fillDetail(box, d); else box.append(el("p", "muted small", "Loading…"));
    openBox = box;
    return box;
  }
  async function toggle(id) {
    open = open === id ? null : id;
    openBox = null;
    if (audioUrl) { URL.revokeObjectURL(audioUrl); audioUrl = null; }
    render();
    if (open && !details.has(id)) {
      try { details.set(id, await api(`/bench/calls/${encodeURIComponent(id)}`)); }
      catch (e) { details.set(id, { error: e.message }); }
      if (open === id && openBox && openBox.dataset.id === id) fillDetail(openBox, details.get(id));
    }
  }
  function analysisCard(title, a) {
    const col = el("div", "col");
    col.append(el("h3", null, title));
    const card = el("div", "card2");
    if (!a) { card.append(el("p", "muted", "No analysis.")); col.append(card); return col; }
    card.append(el("p", null, a.summary || ""));
    const kv = el("div", "kv");
    kv.append(el("span", null, a.sentiment || ""), el("span", null, a.scriptKey ? `${a.scriptKey} ${a.scriptScore ?? ""}/10` : "no script"));
    card.append(kv);
    if (a.scriptStepResults && a.scriptStepResults.length) {
      const ul = el("ul", "steps");
      for (const s of a.scriptStepResults) {
        const li = el("li");
        li.append(el("span", "st", s.status), el("span", null, s.key));
        ul.append(li);
      }
      card.append(ul);
    }
    col.append(card);
    return col;
  }
  function fillDetail(box, d) {
    box.replaceChildren();
    if (d.error) { box.append(el("p", "muted small", d.error)); return; }
    const head = el("p", "small");
    head.append(el("b", null, `Verdict: ${d.verdict ? d.verdict.verdict : "–"}`),
      document.createTextNode(` · noise check: ${d.noise ? d.noise.verdict : "–"} · ${fmt(d.duration)} call · ${d.speakers ?? "?"} speakers`));
    box.append(head);
    const timing = el("p", "small muted");
    const ratio = num(d.diarai_seconds) && d.duration ? ` (${(d.diarai_seconds / d.duration).toFixed(2)}× the call)` : "";
    timing.textContent = (d.sarvam_mode === "live"
      ? `Sarvam: transcribed live during the call, so the transcript was ready at hang-up; zdial's analysis was stored ${secs(d.sarvam_ready_seconds)} after.`
      : `Sarvam: transcribed after the call; zdial's analysis was stored ${secs(d.sarvam_ready_seconds)} after hang-up.`)
      + ` diarai: transcribed the recording in ${secs(d.diarai_seconds)}${ratio}, so with the same analysis the result would be ready about ${secs(diaraiReady(d))} after hang-up.`;
    box.append(timing);
    if (d.verdict_maya) {
      const m = el("p", "small");
      const r = d.duration ? ` (${((d.maya_seconds || 0) / d.duration).toFixed(2)}× the call)` : "";
      m.append(el("b", null, `maya verdict: ${d.verdict_maya.verdict}`),
        document.createTextNode(` · maya vs diarai: ${d.verdict_maya_vs_diarai ? d.verdict_maya_vs_diarai.verdict : "–"} · maya transcribed in ${secs(d.maya_seconds)}${r}`));
      box.append(m);
    } else if (d.maya_error) {
      box.append(el("p", "small muted", "maya: " + d.maya_error));
    }
    const diffs = (d.verdict && d.verdict.differences) || [];
    if (diffs.length) {
      const ul = el("ul", "diffs");
      for (const x of diffs) ul.append(el("li", null, x));
      box.append(ul);
    }
    if (d.has_audio) {
      const audio = el("audio");
      audio.controls = true;
      audio.preload = "none";
      box.append(audio);
      api(`/bench/calls/${encodeURIComponent(d.id)}/audio`, true).then((b) => {
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        audioUrl = URL.createObjectURL(b);
        audio.src = audioUrl;
      }).catch(() => audio.replaceWith(el("p", "muted small", "Audio unavailable.")));
    }
    const cols = el("div", "cols");
    cols.append(analysisCard("AI on Sarvam transcript", d.analysis_sarvam), analysisCard("AI on diarai transcript", d.analysis_diarai));
    if (d.analysis_maya) cols.append(analysisCard("AI on maya transcript", d.analysis_maya));
    box.append(cols);
    const tx = el("div", "cols");
    const txs = [["Sarvam transcript (stored by zdial)", d.transcript_sarvam], ["diarai transcript", d.transcript_diarai]];
    if (d.transcript_maya !== undefined) txs.push(["maya transcript", d.transcript_maya]);
    if (d.verdict_maya && (d.verdict_maya.differences || []).length) {
      box.append(el("p", "small muted", "maya vs Sarvam differences:"));
      const ul = el("ul", "diffs");
      for (const x of d.verdict_maya.differences) ul.append(el("li", null, x));
      box.append(ul);
    }
    for (const [title, text] of txs) {
      const c = el("div", "col");
      c.append(el("h3", null, title), el("pre", "tx", text || "(empty)"));
      tx.append(c);
    }
    box.append(tx);
  }
})();
