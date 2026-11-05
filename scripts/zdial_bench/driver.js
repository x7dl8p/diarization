// zdial benchmark runner: Sarvam transcript vs diarai transcript, judged by what zdial's AI concludes from each.
//
// Runs on the zdial host in a throwaway container made from the zdial-backend image (same env and network,
// so it reaches Postgres, MinIO, diarai and the AI gateway exactly as the backend does). Read-only towards
// zdial: the DB session is read-only and nothing is written back. Results go to diarai's /bench store.
//
// Per call:
//   1. recording from S3/MinIO -> diarai POST /speech-to-text (hi-IN, as zdial's DiaraiSttService sends it)
//   2. zdial's own PostCallService.executeAnalysis (compiled code, current prompt, the call's scripts) on
//        A  the stored Sarvam transcript      A2  the same again (LLM run-to-run noise)      B  the diarai transcript
//   3. judge(A, B) = does diarai change the conclusion?   judge(A, A2) = how much it changes with no transcript change
//   4. PUT /bench/calls/<id> with the audio and everything above; PUT /bench/status for live progress
//
// env: the zdial backend env (DB_*, S3_*, AI_GATEWAY_API_KEY, DIARAI_URL, DIARAI_API_KEY) + BENCH_N, BENCH_INBOUND
const R = require("module").createRequire("/repo/backend/dist/src/main.js");
const { Client } = R("pg");
const { S3Client, GetObjectCommand } = R("@aws-sdk/client-s3");
const { z } = R("zod");
const { PostCallService } = R("./modules/post-call/post-call.service.js");
const { AiService } = R("./common/modules/ai/ai.service.js");

const N = Number(process.env.BENCH_N || 100);
const N_IN = Number(process.env.BENCH_INBOUND || 30);
const DIARAI = String(process.env.DIARAI_URL || "").replace(/\/+$/, "");
const KEY = process.env.DIARAI_API_KEY || "";
if (!DIARAI || !KEY) throw new Error("DIARAI_URL / DIARAI_API_KEY missing from the env");
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

// ------------------------------------------------------------------ zdial's analysis, as compiled
const ai = new AiService({ get: (k) => (k === "aiGateway.apiKey" ? process.env.AI_GATEWAY_API_KEY : undefined) });
const postCall = Object.create(PostCallService.prototype);
postCall.logger = { log() {}, warn() {}, error() {}, debug() {}, verbose() {} };
postCall.aiService = ai;

async function analyse(call, transcript, scripts, assigned) {
  for (let i = 0; ; i++) {
    try { return await postCall.executeAnalysis(call, transcript, scripts, assigned); }
    catch (e) { if (i >= 2) throw e; await new Promise((r) => setTimeout(r, 5000 * (i + 1))); }
  }
}

const verdictSchema = z.object({
  verdict: z.enum(["same", "minor", "different"]),
  differences: z.array(z.string()).default([]),
});
async function judge(a, b) {
  const view = (x) => JSON.stringify({ summary: x.summary, sentiment: x.sentiment, keyTopics: x.keyTopics,
    actionItems: x.actionItems, scriptKey: x.scriptKey, scriptScore: x.scriptScore }, null, 1);
  const prompt = `Two automated analyses were written for the SAME customer phone call, each from a different
transcript of it. A supervisor reads them to decide what happened and what to do next.

Analysis A:
${view(a)}

Analysis B:
${view(b)}

Compare what a supervisor would conclude from each: what the call was about, what the customer
wanted or complained about, how it ended, and what follow-up is needed. Ignore wording, ordering
and small score differences (within 1.5 points). Report:
- verdict: "same" if a supervisor would take the same conclusion and actions from both,
  "minor" if the conclusion is the same but a fact, detail or follow-up differs,
  "different" if the conclusion or the required action differs.
- differences: each concrete factual difference, briefly (empty when none).`;
  for (let i = 0; ; i++) {
    try { return await ai.generateObject({ prompt, schema: verdictSchema }); }
    catch (e) { if (i >= 2) throw e; await new Promise((r) => setTimeout(r, 5000 * (i + 1))); }
  }
}

function stepAgreement(a, b) {
  if (!a || !b || !a.scriptKey || a.scriptKey !== b.scriptKey) return null;
  const sa = Object.fromEntries((a.scriptStepResults || []).map((s) => [s.key, s.status]));
  const sb = Object.fromEntries((b.scriptStepResults || []).map((s) => [s.key, s.status]));
  const keys = Object.keys(sa).filter((k) => k in sb);
  return { agree: keys.filter((k) => sa[k] === sb[k]).length, total: keys.length };
}

// same lines as zdial's formatDiarizedEntries: "[mm:ss] [Speaker N]: text", ids by first appearance
function formatEntries(entries) {
  const ids = new Map();
  const out = [];
  for (const e of entries || []) {
    const t = String(e.transcript || "").trim();
    if (!t) continue;
    const key = String(e.speaker_id);
    if (!ids.has(key)) ids.set(key, `Speaker ${ids.size + 1}`);
    const s = Math.floor(Number(e.start_time_seconds) || 0);
    out.push(`[${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}] [${ids.get(key)}]: ${t}`);
  }
  return { text: out.join("\n"), speakers: ids.size };
}

// ------------------------------------------------------------------ diarai
async function diarai(path, init = {}) {
  const res = await fetch(DIARAI + path, { ...init, headers: { "Api-Subscription-Key": KEY, ...(init.headers || {}) } });
  if (!res.ok) throw new Error(`diarai ${path} HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  return res.json();
}
async function transcribe(audio, name) {
  const form = new FormData();
  form.append("file", new Blob([audio]), name);
  form.append("language_code", "hi-IN");
  form.append("with_timestamps", "false");
  form.append("with_diarization", "true");
  // Not fetch(): Node's built-in fetch gives up when response headers take over 300 s (undici headersTimeout),
  // and a long recording legitimately takes diarai longer. Plain http with our own 20-minute limit instead.
  const req = new Request("http://x/", { method: "POST", body: form });
  const body = Buffer.from(await req.arrayBuffer());
  const url = new URL(DIARAI + "/speech-to-text");
  const lib = url.protocol === "https:" ? R("https") : R("http");
  return new Promise((resolve, reject) => {
    const r = lib.request(url, { method: "POST", timeout: 20 * 60_000, headers: {
      "Api-Subscription-Key": KEY, "Content-Type": req.headers.get("content-type"), "Content-Length": body.length } }, (res) => {
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => {
        const text = Buffer.concat(chunks).toString("utf8");
        if (res.statusCode >= 400) return reject(new Error(`diarai /speech-to-text HTTP ${res.statusCode}: ${text.slice(0, 200)}`));
        try { resolve(JSON.parse(text)); } catch (e) { reject(e); }
      });
    });
    r.on("timeout", () => r.destroy(new Error("diarai took over 20 minutes")));
    r.on("error", reject);
    r.end(body);
  });
}
async function upload(id, meta, audio, name) {
  const form = new FormData();
  form.append("meta", JSON.stringify(meta));
  if (audio) form.append("audio", new Blob([audio]), name);
  return diarai(`/bench/calls/${id}`, { method: "PUT", body: form });
}
const status = { state: "starting", total: 0, done: 0, failed: 0, ids: [], errors: [], started: new Date().toISOString() };
const putStatus = () => diarai("/bench/status", { method: "PUT", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(status) }).catch((e) => log("status upload failed:", e.message));

// ------------------------------------------------------------------ main
async function main() {
  const db = new Client({
    host: process.env.DB_HOST, port: Number(process.env.DB_PORT || 5432), user: process.env.DB_USER,
    password: process.env.DB_PASS, database: process.env.DB_NAME,
    ssl: /^(1|true)$/i.test(process.env.DB_SSL || "") ? { rejectUnauthorized: false } : undefined,
    options: "-c default_transaction_read_only=on",
  });
  await db.connect();
  const s3 = new S3Client({
    endpoint: process.env.S3_ENDPOINT_URL || undefined, region: process.env.S3_REGION || "us-east-1",
    forcePathStyle: /^(1|true)$/i.test(process.env.S3_FORCE_PATH_STYLE || "true"),
    credentials: { accessKeyId: process.env.S3_ACCESS_KEY_ID, secretAccessKey: process.env.S3_SECRET_KEY },
  });

  // the sample is fixed once (kept in status.json), so a restart resumes the same calls
  const prev = await diarai("/bench/summary").catch(() => ({ status: null, calls: [] }));
  const finished = new Set(prev.calls.filter((c) => !c.error).map((c) => c.id));
  let ids = prev.status && prev.status.ids && prev.status.ids.length ? prev.status.ids : null;
  if (!ids) {
    const pick = async (dir, n) => (await db.query(
      `select p.call_id from post_call_analysis p join call_sessions c on c."callId"::text = p.call_id
        where p.status = 'COMPLETED' and c."recordingKey" is not null and length(p.transcript) > 200
          and c."durationSeconds" between 30 and 600 and c.direction = $1
        order by md5(p.call_id || 'diarai-bench-1') limit $2`, [dir, n])).rows.map((r) => r.call_id);
    const inbound = await pick("inbound", N_IN);
    ids = [...inbound, ...(await pick("outbound", N - inbound.length))];
  }
  Object.assign(status, { state: "running", total: ids.length, ids, done: ids.filter((i) => finished.has(i)).length });
  await putStatus();
  log(`benchmark: ${ids.length} calls, ${status.done} already done`);

  const inbound = (await db.query(
    `select key, name, description, steps from call_scripts
      where "inboundEligible" and "isActive" and deleted_at is null order by name`)).rows;

  let pending = Promise.resolve();
  for (const id of ids) {
    if (finished.has(id)) continue;
    status.current = id;
    await putStatus();
    try {
      const { rows } = await db.query(
        `select c."callId" as "callId", c."durationSeconds" as "durationSeconds", c."destinationNumber" as "destinationNumber",
                c.status, c.direction, c."callScriptId" as "callScriptId", c."recordingKey" as "recordingKey",
                p.transcript, p.summary, p.sentiment, p.script_key, p.script_score, p.updated_at,
                extract(epoch from p.updated_at - p.created_at)::float as "sarvamReady",
                (p.transcript ~ '^\[\d\d:\d\d\] \[Speaker') as "sarvamBatch"
           from post_call_analysis p join call_sessions c on c."callId"::text = p.call_id where p.call_id = $1`, [id]);
      const call = rows[0];
      if (!call) throw new Error("call not found");
      let scripts = [], assigned = false;
      if (call.callScriptId) {
        scripts = (await db.query(`select key, name, description, steps from call_scripts where id = $1`, [call.callScriptId])).rows;
        assigned = scripts.length > 0;
      } else if (call.direction === "inbound") {
        scripts = inbound;
      }
      const obj = await s3.send(new GetObjectCommand({ Bucket: process.env.S3_BUCKET_NAME, Key: call.recordingKey }));
      const audio = Buffer.from(await obj.Body.transformToByteArray());
      const name = call.recordingKey.split("/").pop() || `${id}.wav`;
      const t0 = Date.now();
      const tr = await transcribe(audio, name);
      const wall = (Date.now() - t0) / 1000;
      const fmt = formatEntries((tr.diarized_transcript || {}).entries);
      log(`${id.slice(0, 8)} ${call.direction} ${call.durationSeconds}s -> diarai ${wall.toFixed(0)}s, ${fmt.speakers} speakers`);

      // the AI half runs while the next recording is being transcribed
      await pending;
      pending = (async () => {
        try {
          const callCtx = { callId: call.callId, durationSeconds: call.durationSeconds,
                            destinationNumber: call.destinationNumber, status: call.status };
          const [A, A2, B] = await Promise.all([
            analyse(callCtx, call.transcript, scripts, assigned),
            analyse(callCtx, call.transcript, scripts, assigned),
            analyse(callCtx, fmt.text || (tr.transcript || ""), scripts, assigned),
          ]);
          const [verdict, noise] = await Promise.all([judge(A, B), judge(A, A2)]);
          await upload(id, {
            duration: call.durationSeconds, direction: call.direction, call_status: call.status,
            scripts_mode: assigned ? "assigned" : scripts.length ? "choose" : "none",
            analysed_at: call.updated_at,
            stored: { summary: call.summary, sentiment: call.sentiment, scriptKey: call.script_key, scriptScore: call.script_score },
            transcript_sarvam: call.transcript, transcript_diarai: fmt.text, speakers: fmt.speakers,
            diarai_language: tr.language_code, diarai_seconds: Math.round(wall),
            // hang-up -> zdial's stored analysis; for live transcripts that is the analysis alone
            sarvam_mode: call.sarvamBatch ? "batch" : "live", sarvam_ready_seconds: Math.round(call.sarvamReady * 10) / 10,
            audio_seconds: (tr.metrics || {}).audio_duration,
            analysis_sarvam: A, analysis_sarvam_repeat: A2, analysis_diarai: B,
            verdict, noise, steps: stepAgreement(A, B), steps_noise: stepAgreement(A, A2),
          }, audio, name);
          status.done += 1;
          log(`${id.slice(0, 8)} verdict ${verdict.verdict} (noise ${noise.verdict}) [${status.done}/${status.total}]`);
        } catch (e) {
          await fail(id, e);
        }
        await putStatus();
      })();
    } catch (e) {
      await fail(id, e);
      await putStatus();
    }
  }
  await pending;
  status.state = "finished";
  status.current = null;
  await putStatus();
  await db.end();
  log(`finished: ${status.done} done, ${status.failed} failed`);
}

async function fail(id, e) {
  status.failed += 1;
  status.errors.push({ id, error: String(e && e.message || e).slice(0, 300) });
  log(`${id.slice(0, 8)} FAILED: ${e && e.message}`);
  await upload(id, { error: String(e && e.message || e).slice(0, 300) }).catch(() => {});
}

main().catch(async (e) => {
  log("fatal:", e && e.stack || e);
  status.state = "crashed";
  status.errors.push({ id: null, error: String(e && e.message || e).slice(0, 300) });
  await putStatus();
  process.exit(1);
});
