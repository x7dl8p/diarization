"use strict";
// zdial benchmark view (door button). Reads /bench/* with the API key from the main page; polls while the run is live.
(() => {
  const $ = (s) => document.querySelector(s);
  const home = $("#home"), view = $("#bench"), door = $("#door"), back = $("#benchback");
  const statsEl = $("#bstats"), listEl = $("#blist"), filterEl = $("#bfilter"), needKey = $("#bneedkey");
  let rows = [], status = null, filter = "all", open = null, timer = null, audioUrl = null;

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
      render();
    } catch (e) {
      statsEl.replaceChildren(el("p", "muted small", "Could not load the benchmark: " + e.message));
    }
    if (status && (status.state === "running" || status.state === "starting")) timer = setTimeout(refresh, 10000);
  }

  // ---------------------------------------------------------------- stats
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
    );

    filterEl.replaceChildren();
    const counts = { all: rows.length, different: v.different, minor: v.minor, same: v.same, failed: failed.length };
    for (const k of Object.keys(counts)) {
      const b = el("button", "chip", `${k} ${counts[k]}`);
      b.type = "button";
      b.setAttribute("aria-pressed", String(filter === k));
      b.addEventListener("click", () => { filter = k; render(); });
      filterEl.append(b);
    }

    const order = { different: 0, minor: 1, same: 2 };
    const shown = rows
      .filter((r) => filter === "all" || (filter === "failed" ? r.error : r.verdict === filter))
      .sort((a, b) => (a.error ? 3 : order[a.verdict] ?? 3) - (b.error ? 3 : order[b.verdict] ?? 3));
    listEl.replaceChildren();
    if (!shown.length) listEl.append(el("p", "muted small", rows.length ? "No calls in this filter." : "No results yet."));
    for (const r of shown) {
      const row = el("button", "brow");
      row.type = "button";
      const verdict = r.error ? "failed" : r.verdict || "pending";
      const mid = el("div");
      mid.append(el("div", "id", r.id.slice(0, 8)),
        el("div", "sub", r.error ? r.error : `${r.direction || ""} · ${fmt(r.duration)} · ${r.speakers ?? "?"} speakers · noise ${r.noise || "–"}`));
      const sc = r.script[0] || r.script[1]
        ? `${r.script[0] || "none"} ${num(r.score[0]) ? r.score[0] : ""} → ${r.script[1] || "none"} ${num(r.score[1]) ? r.score[1] : ""}`
        : `${(r.sentiment[0] || "").toLowerCase()} → ${(r.sentiment[1] || "").toLowerCase()}`;
      row.append(el("span", `verdict ${verdict}`, verdict), mid, el("div", "right", sc));
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
      document.createTextNode(` · noise check: ${d.noise ? d.noise.verdict : "–"} · ${fmt(d.duration)} call, diarai took ${d.diarai_seconds ?? "?"} s · ${d.speakers ?? "?"} speakers`));
    box.append(head);
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
    box.append(cols);
    const tx = el("div", "cols");
    for (const [title, text] of [["Sarvam transcript (stored by zdial)", d.transcript_sarvam], ["diarai transcript", d.transcript_diarai]]) {
      const c = el("div", "col");
      c.append(el("h3", null, title), el("pre", "tx", text || "(empty)"));
      tx.append(c);
    }
    box.append(tx);
  }
})();
