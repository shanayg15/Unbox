// Unbox UI. Vanilla JS. Everything on screen is derived from the event stream (contracts/events.py)
// plus /api/report for the diagnostic report, so a live run, a replay and a reload render the same way.
"use strict";

const $ = (id) => document.getElementById(id);
const GLOW_MS = 3000;
const PLAY_FPS_MAX = 10;                 // clips play at most this fast
const HOLD_MS = 1800;                    // hold the last frame (with its label) before looping
const HIDE_FROM_MENU = new Set([]);      // flaws not reliably found go here (they stay in Results)
const SHOWN_ENVS = new Set(["CartPole", "MountainCar"]);   // robots this demo covers
const HIDDEN_POLICIES = new Set(["bb-02"]);                  // CartPole DQN: copies are not reliable enough to demo

const ROBOTS = {
  CartPole: "Balances a pole on a moving cart",
  MountainCar: "Drives an underpowered car up a hill",
  LunarLander: "Lands a craft between two flags",
  Acrobot: "Swings a two-link arm above a line",
};

const S = {};
window.__unbox = S;
let POLICIES = [];
let RESULT_POLICIES = [];

function resetRun() {
  Object.assign(S, {
    finished: false, cached: false, banner: null, lastT: 0, log: [],
    budget: 5000, queries: 0, observed: 0, metrics: [], best: null, shownCode: null,
    ledger: [], highlight: new Set(), snaps: [],
    fix: null, revealed: null, report: null, finishedData: null, fixing: false, fixStep: -1, fixError: null,
    clip: null, shield: { on: false, bb: null, bb_end: null }, viewFixed: false, fixClip: null,
  });
}
Object.assign(S, { runId: null, kind: null, config: {} });
resetRun();

// ---------------------------------------------------------------- formatting
const fmtInt = (n) => (n == null ? "—" : Math.round(n).toLocaleString("en-US"));
const fmtPct = (x, d = 1) => (x == null ? "—" : `${(100 * x).toFixed(d)}%`);
const fmtK = (n) => (n >= 1000 ? `${Math.round(n / 1000)}K` : `${n}`);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const words = (s, n = 12) => { const w = String(s || "").split(/\s+/).filter(Boolean); return w.length > n ? w.slice(0, n).join(" ") + "…" : w.join(" "); };
const envFamily = (envId) => String(envId || "").split("-")[0];
function policyOf(bb) { return RESULT_POLICIES.find((p) => p.bb === bb) || { bb, label: bb, env_id: "" }; }
function robotName(p) { const [env, algo] = String(p.label || p.bb).split("-"); return algo ? `${env}, ${algo}` : env; }
const flawNum = (tid) => (tid ? (String(tid).match(/(\d+)$/) || [, "?"])[1] : null);
const matchNow = () => (S.best ? S.best.agreement : null);
const hasFlaw = () => !!S.config.hasFlaw;
const scenarioName = () => (S.config.twist_id ? `Planted flaw ${flawNum(S.config.twist_id)}` : "Normal operation");

function pill(id, text, kind = "") {
  const el = $(id);
  el.textContent = text;
  el.className = `pill ${kind}`;
}

// ---------------------------------------------------------------- event dispatch
function apply(ev) {
  const d = ev.data || {};
  if (typeof ev.t === "number") S.lastT = Math.max(S.lastT, ev.t);
  switch (ev.type) {
    case "run_started": S.budget = d.budget || S.budget; logLine("Run started", ev.t); break;
    case "status":
      if (d.banner) { softReset(); showBanner(d.text); S.cached = true; logLine(d.text, ev.t, "warn"); break; }
      logLine(d.text, ev.t);
      break;
    case "error": logLine(d.message, ev.t, "error"); break;
    case "query_count":
      S.queries = Math.max(S.queries, d.n || 0);
      if (d.observed) S.observed = Math.max(S.observed, d.observed);
      break;
    case "metric":
      S.metrics.push({ q: d.queries, a: d.agreement });
      S.queries = Math.max(S.queries, d.queries || 0);
      renderCurve();
      break;
    case "ledger": S.ledger = d.entries || []; renderLedger(); break;
    case "snap": S.snaps.push(d); renderSnap(); break;
    case "program": onProgram(d); break;
    case "frames": onFrames(d); break;
    case "reveal": onReveal(d); break;
    case "fix": onFix(d); break;
    case "shield": onShield(d); break;
    case "run_finished":
      S.finished = true; S.cached = S.cached || !!d.cached; S.finishedData = d;
      if (d.total_queries != null) S.queries = Math.max(S.queries, d.total_queries);
      logLine(`Analysis complete. The recovered policy matches the network on ${fmtPct(matchNow())} of unseen situations.`, ev.t, "ok");
      renderCurve();
      loadReport();
      break;
    default: break;
  }
  renderKpis();
  renderState();
}

// ---------------------------------------------------------------- activity + KPIs + state pills
function logLine(text, t, kind = "") {
  if (!text) return;
  const last = S.log[S.log.length - 1];
  if (last && last.text === text) return;
  S.log.push({ text, t, kind });
  const rows = S.log.slice(-7).reverse();
  $("feed").innerHTML = rows.map((r, i) => `<li class="${r.kind}${i === 0 ? " latest" : ""}"><span class="t mono">${r.t != null ? `${(+r.t).toFixed(0)} s` : ""}</span><span>${esc(r.text)}</span></li>`).join("");
}

function renderKpis() {
  $("kMatch").textContent = fmtPct(matchNow());
  const snapRows = S.snaps.length ? S.snaps[S.snaps.length - 1].n_rows : 0;
  $("kDecisions").textContent = fmtInt(Math.max(S.observed + S.queries, snapRows || 0));
  $("kVersion").textContent = S.best ? `v${S.best.version}` : "—";
  $("kTime").textContent = S.runId ? `${Math.round(S.lastT)} s` : "—";
}

function renderState() {
  const running = S.runId && !S.finished;
  $("btnRun").disabled = !!running;
  pill("pillSetup", !S.runId ? "Ready" : running ? "Running" : "Complete", running ? "run" : S.runId ? "ok" : "");
  pill("pillAnalysis", !S.runId ? "Waiting" : running ? "Running" : S.cached ? "Complete · earlier run" : "Complete", running ? "run" : S.runId ? "ok" : "");
  if (!S.finished) pill("pillFix", "Waiting");
  else if (!hasFlaw()) pill("pillFix", "Not needed", "muted");
  else if (S.fixing) pill("pillFix", "In progress", "run");
  else if (S.shield.on) pill("pillFix", "Applied", "ok");
  else pill("pillFix", "Action required", "warn");
  pill("pillReport", S.report ? "Ready" : S.finished ? "Generating" : "Waiting", S.report ? "ok" : "");
  renderBadge();
  renderDiag();
}

function showBanner(text) {
  S.banner = text;
  $("banner").textContent = text;
  $("banner").hidden = false;
}
function renderBadge() {
  const b = $("runBadge");
  if (!S.runId) { b.textContent = ""; return; }
  b.textContent = S.kind === "replay" ? "Earlier run" : S.cached ? "Earlier run (automatic)" : "Live run";
  b.title = S.runId;
}

// ---------------------------------------------------------------- 2 Analysis
function renderCurve() {
  const svg = $("curve");
  const W = 900, H = 220, L = 54, R = 24, T = 18, B = 40;
  const best = [];
  let run = -1;
  S.metrics.forEach((p) => { run = Math.max(run, p.a); best.push({ q: p.q, a: run }); });
  const qMax = Math.max(1, S.queries, ...best.map((p) => p.q));
  const step = qMax > 4000 ? 1000 : qMax > 2000 ? 500 : qMax > 800 ? 250 : 100;
  const maxQ = Math.ceil((qMax * 1.08) / step) * step;
  const x = (q) => L + (W - L - R) * (q / maxQ);
  const y = (a) => T + (H - T - B) * (1 - a);
  let g = "";
  [0, 0.5, 1].forEach((a) => {
    g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(a)}" y2="${y(a)}"/>`;
    g += `<text class="lbl" x="${L - 10}" y="${y(a) + 4}" text-anchor="end">${a * 100}%</text>`;
  });
  for (let q = 0; q <= maxQ + 1e-9; q += step * Math.max(1, Math.round(maxQ / step / 5))) {
    g += `<text class="lbl" x="${x(q)}" y="${H - B + 18}" text-anchor="middle">${q >= 1000 ? (q / 1000).toFixed(q % 1000 ? 1 : 0) + "k" : q}</text>`;
  }
  g += `<line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>`;
  g += `<text class="lbl" x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">experiments run on the network (queries)</text>`;
  if (best.length) {
    let path = `M${x(0)},${y(best[0].a)} `;
    best.forEach((p, i) => { if (i > 0) path += `L${x(p.q)},${y(best[i - 1].a)} `; path += `L${x(p.q)},${y(p.a)} `; });
    g += `<path class="line" d="${path}"/>`;
    best.forEach((p) => (g += `<circle class="pt" cx="${x(p.q)}" cy="${y(p.a)}" r="3.5"/>`));
    const last = best[best.length - 1];
    const lx = x(last.q), ly = y(last.a);
    if (S.finished) g += `<circle class="final" cx="${lx}" cy="${ly}" r="7"/>`;
    const anchor = lx > W - 140 ? "end" : "start";
    g += `<text class="val" x="${anchor === "end" ? lx - 12 : lx + 12}" y="${ly > T + 30 ? ly - 14 : ly + 26}" text-anchor="${anchor}">${S.finished ? "final " : ""}${fmtPct(last.a)}</text>`;
  } else {
    g += `<text class="lbl" x="${(L + W - R) / 2}" y="${(T + H - B) / 2}" text-anchor="middle">best match so far appears here</text>`;
  }
  svg.innerHTML = g;
}

function renderSnap() {
  const s = S.snaps[S.snaps.length - 1];
  if (!s) return;
  const an = s.anomaly_feature || {};
  const tokens = s.input_tokens ? ` (${fmtK(s.input_tokens)} tokens)` : "";
  const cause = an.choice && an.choice !== "none" ? `most likely source of disagreement: <b>${esc(an.choice)}</b>` : "no single sensor stands out yet";
  $("subqLine").innerHTML = `Read all ${fmtInt(s.n_rows)} raw rows${tokens} · ${cause}`;
  const items = Object.entries(an.probabilities || {}).sort((a, b) => b[1] - a[1]).slice(0, 5);
  $("subqBars").innerHTML = items.map(([k, p]) => `<div class="bar${k === an.choice ? " top" : ""}">
      <span class="name">${esc(k)}</span><span class="track"><span class="fill" style="width:${Math.max(1, 100 * p)}%"></span></span><span class="p">${(+p).toFixed(2)}</span></div>`).join("");
}

const ruleNum = (rid) => parseInt(String(rid).replace(/\D/g, ""), 10) || 0;
const STATUS_WORD = { observed: "seen in data", inferred: "hypothesis", probing: "testing now" };
function ruleRow(e, label) {
  return `<div class="lrow${S.highlight.has(e.rule_id) ? " highlight" : ""}" title="${esc(e.text)}"><span class="rid">${esc(e.rule_id)}</span><span class="text">${esc(words(e.text))}</span>${label ? `<span class="st">${esc(label)}</span>` : "<span></span>"}</div>`;
}
function renderLedger() {
  const box = $("ledger");
  if (!S.ledger.length) { box.innerHTML = '<div class="muted">No rules yet.</div>'; return; }
  const sorted = [...S.ledger].sort((a, b) => ruleNum(a.rule_id) - ruleNum(b.rule_id));
  const conf = sorted.filter((e) => e.status === "confirmed");
  const test = sorted.filter((e) => ["observed", "inferred", "probing"].includes(e.status));
  const out = sorted.filter((e) => e.status === "refuted");
  const wasOpen = box.querySelector("details.ruled")?.open;
  let h = `<div class="lgroup"><div class="lhead">Confirmed <span class="count">${conf.length}</span></div>${conf.map((e) => ruleRow(e)).join("") || '<div class="muted small">None yet.</div>'}</div>`;
  if (test.length) h += `<div class="lgroup"><div class="lhead">Under test <span class="count">${test.length}</span></div>${test.map((e) => ruleRow(e, STATUS_WORD[e.status])).join("")}</div>`;
  if (out.length) h += `<details class="lgroup ruled"${wasOpen ? " open" : ""}><summary class="lhead">Rejected (${out.length})</summary>${out.map((e) => ruleRow(e)).join("")}</details>`;
  box.innerHTML = h;
}

const KW = new Set(["def", "if", "elif", "else", "return", "and", "or", "not", "True", "False", "None", "in", "is"]);
const TOKEN_RE = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d+\.?\d*(?:e-?\d+)?\b)|(\b[A-Za-z_]\w*\b)|([\s\S])/gm;
function highlightLine(line) {
  let out = "";
  line.replace(TOKEN_RE, (m, com, str, num, word, other) => {
    if (com) out += `<span class="tok-com">${esc(com)}</span>`;
    else if (str) out += `<span class="tok-str">${esc(str)}</span>`;
    else if (num) out += `<span class="tok-num">${esc(num)}</span>`;
    else if (word) out += KW.has(word) ? `<span class="tok-kw">${word}</span>` : word === "P" ? `<span class="tok-p">P</span>` : esc(word);
    else out += esc(other);
    return m;
  });
  return out || " ";
}
function lineDiff(a, b) {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const ops = []; let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { ops.push({ op: "eq", line: b[j] }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) ops.push({ op: "del", line: a[i++] });
    else ops.push({ op: "add", line: b[j++] });
  }
  while (i < n) ops.push({ op: "del", line: a[i++] });
  while (j < m) ops.push({ op: "add", line: b[j++] });
  return ops;
}

function onProgram(d) {
  if (S.best && (d.agreement ?? 0) < (S.best.agreement ?? 0)) {
    logLine(`Version ${d.version} matched ${fmtPct(d.agreement)}, below v${S.best.version}; keeping v${S.best.version}`, null);
    return;
  }
  S.best = d;
  const lines = d.code.replace(/\s+$/, "").split("\n");
  const changed = new Set();
  if (S.shownCode != null) {
    let ln = 0;
    lineDiff(S.shownCode.split("\n"), lines).forEach((o) => { if (o.op !== "del") { ln++; if (o.op === "add") changed.add(ln); } });
  } else (d.changed_lines || []).forEach((n) => changed.add(n));
  S.shownCode = lines.join("\n");
  $("codeTitle").innerHTML = `Recovered policy <span class="meta">v${d.version} · ${lines.length} lines · ${fmtPct(d.agreement)} match</span>`;
  $("code").innerHTML = lines.map((l, i) =>
    `<div class="cl${changed.has(i + 1) ? " glow" : ""}"><span class="ln">${i + 1}</span><span class="src">${highlightLine(l)}</span></div>`).join("");
  $("params").innerHTML = Object.entries(d.params || {}).map(([k, v]) => `<span>${esc(k)} = ${(+v).toFixed(3)}</span>`).join("");
  clearTimeout(S.glowTimer);
  S.glowTimer = setTimeout(() => $("code").querySelectorAll(".cl.glow").forEach((el) => el.classList.remove("glow")), GLOW_MS);
  pill("clonePill", `v${d.version} · ${fmtPct(d.agreement)} match`, "ok");
}

// ---------------------------------------------------------------- 3 Remediation
function suspect() {
  const r = S.report;
  if (r && r.failing) return r.failing.sentence || r.failing.text;
  const conf = S.ledger.filter((e) => e.status === "confirmed");
  if (!conf.length) return null;
  const snap = S.snaps[S.snaps.length - 1];
  const f = snap && snap.anomaly_feature && snap.anomaly_feature.choice;
  const hit = f && f !== "none" ? conf.filter((e) => e.text.toLowerCase().includes(f.toLowerCase())) : [];
  return (hit.length ? hit[hit.length - 1] : conf[conf.length - 1]).text;
}

function renderDiag() {
  const box = $("diag");
  $("fixRow").hidden = true;
  if (!S.runId || !S.finished) { box.innerHTML = '<div class="muted">Available when the analysis completes.</div>'; return; }
  if (!hasFlaw()) {
    box.innerHTML = `<div class="callout ok"><div class="ck">No failure detected</div><div>The recovered policy matches the network on <b>${fmtPct(matchNow())}</b> of unseen situations. No remediation is needed.</div></div>`;
    return;
  }
  $("fixRow").hidden = false;
  $("btnFix").hidden = !!S.shield.on;
  $("btnFix").disabled = S.fixing;
  $("btnFix").textContent = S.fixing ? "Applying fix…" : "Apply fix";
  $("viewToggle").hidden = !S.shield.on;
  const r = S.report;
  if (!S.revealed) {
    const t = suspect();
    box.innerHTML = `<div class="callout warn"><div class="ck">Suspected failure</div><div class="big">${t ? esc(t) : "No confirmed rule yet."}</div>
      <div class="muted small">Identified by Unbox before the issue is reported. Apply fix to compare it with the reported issue, correct it and verify the result.</div></div>`;
    return;
  }
  const found = r && r.failing ? (r.failing.sentence || r.failing.text) : null;
  const badge = !r ? '<span class="badge wait">Checking…</span>' : r.found === "match" ? '<span class="badge ok">Diagnosis confirmed</span>' : '<span class="badge no">No match</span>';
  box.innerHTML = `<div class="two">
      <div class="callout"><div class="ck">Reported issue</div><div class="big">${esc(S.revealed.label)}</div></div>
      <div class="callout"><div class="ck">Unbox diagnosis</div><div class="big">${found ? esc(found) : '<span class="muted">No matching confirmed rule</span>'}</div>
      ${r && r.failing && r.failing.line ? `<div class="muted small">Line ${r.failing.line} of the recovered policy</div>` : ""}</div>
    </div><div class="badgerow">${badge}</div>`;
}

const FIX_STEPS = ["Compare with the reported issue", "Muse edits the recovered policy", "Verify the fix on the deployed policy"];
function renderFixSteps() {
  const box = $("fixSteps");
  if (S.fixStep < 0) { box.hidden = true; return; }
  box.hidden = false;
  box.innerHTML = FIX_STEPS.map((t, i) => {
    const st = S.fixError && i === S.fixStep ? "err" : i < S.fixStep ? "done" : i === S.fixStep && S.fixing ? "run" : i === S.fixStep ? "done" : "todo";
    const word = { done: "Done", run: "Working", todo: "Pending", err: "Failed" }[st];
    return `<li class="${st}"><span class="dot"></span><span>${t}</span><span class="w">${word}</span></li>`;
  }).join("");
}

async function onFixClick() {
  if (S.fixing) return;
  S.fixing = true; S.fixError = null; S.fixStep = 0;
  renderFixSteps(); renderState();
  try {
    if (!S.revealed) await post("/api/reveal", { run_id: S.runId });
    S.fixStep = 1; renderFixSteps();
    if (!S.fix) await post("/api/fix", { run_id: S.runId });
    S.fixStep = 2; renderFixSteps();
    await post("/api/shield", { run_id: S.runId, on: true });
    S.fixStep = 3;
  } catch (e) {
    S.fixError = e.message;
    logLine(`Fix failed: ${e.message}`, S.lastT, "error");
  } finally {
    S.fixing = false;
    renderFixSteps(); renderState();
  }
}

function onReveal(d) {
  S.revealed = d;
  S.highlight = new Set((d.highlight_rule_ids || []).slice(0, 1));
  renderLedger();
  loadReport();
}

function onFix(d) {
  S.fix = d;
  const rows = String(d.diff || "").split("\n").filter((l) => l && !l.startsWith("---") && !l.startsWith("+++"));
  $("diff").innerHTML = rows.map((l) => {
    const c = l.startsWith("@@") ? "hunk" : l[0] === "+" ? "add" : l[0] === "-" ? "del" : "ctx";
    return `<div class="dl ${c}">${esc(l)}</div>`;
  }).join("") || '<div class="dl ctx">(no change)</div>';
  $("diff").hidden = false;
  const pIdx = rows.findIndex((l) => l.startsWith("@@") && l.includes("params"));
  const codeRows = pIdx >= 0 ? rows.slice(0, pIdx) : rows;
  const nAdd = codeRows.filter((l) => l[0] === "+").length, nDel = codeRows.filter((l) => l[0] === "-").length;
  const reg = S.report && S.report.region ? S.report.region.text : "the flaw zone";
  $("fixExplain").innerHTML = nAdd + nDel
    ? `<div class="ck">Change made by Muse: ${nDel} line${nDel === 1 ? "" : "s"} removed, ${nAdd} added</div>${esc(d.explanation || "")}`
    : `<div class="ck">No code change needed</div>The recovered policy has no separate branch for the failing rule: its general law already selects the correct action in ${esc(reg)}. The runtime guard applies it there. ${esc(d.explanation || "")}`;
  $("fixExplain").hidden = false;
  $("diff").hidden = !(nAdd + nDel);
  S.fixClip = d.clone && d.clone.length ? { clone: d.clone, clone_end: d.clone_end } : null;
  logLine("Fix written to the recovered policy", S.lastT, "ok");
  loadReport();
}

function onShield(d) {
  if (!d.on) return;
  S.shield = { on: true, bb: d.bb, bb_end: d.bb_end };
  setView(true);
  logLine("Fix applied to the deployed policy", S.lastT, "ok");
  loadReport();
}

function setView(fixed) {
  S.viewFixed = !!(fixed && S.shield.on);
  $("viewFixed").classList.toggle("on", S.viewFixed);
  $("viewOrig").classList.toggle("on", !S.viewFixed);
  $("bbPanel").classList.toggle("guarded", S.viewFixed);
  $("bbTitle").textContent = S.viewFixed ? "Deployed policy · fix applied" : "Deployed policy";
  $("cloneTitle").textContent = S.viewFixed && S.fixClip ? "Recovered policy · fixed" : "Recovered policy";
  $("clonePanel").classList.toggle("guarded", S.viewFixed && !!S.fixClip);
  pill("bbPill", S.viewFixed ? "With fix" : S.shield.on ? "Original" : S.clip ? "Running" : "Idle", S.viewFixed ? "ok" : "");
  if (S.clip) startPlayer();
}

function renderChecks() {
  const r = S.report, box = $("checks");
  const g = r && r.guard_check;
  const c = r && r.cost && !r.cost.pending && !r.cost.error && r.cost.robot_reward != null ? r.cost : null;
  if (!S.shield.on || (!g && !c && !(r && r.cost && r.cost.pending))) { box.hidden = true; return; }
  box.hidden = false;
  let h = "";
  if (g) h += `<div class="check"><span class="ck">No side effects</span><span>${esc(g.text)}</span></div>`;
  if (c) h += `<div class="check"><span class="ck">Outcome</span><span>Reward on the same ${c.n} test starts: ${(+c.robot_reward).toFixed(0)} without the fix, ${(+c.guarded_reward).toFixed(0)} with it` +
    (c.robot_success != null && c.guarded_success != null ? ` · success ${c.robot_success}/${c.n} → ${c.guarded_success}/${c.n}` : "") + `</span></div>`;
  else if (r && r.cost && r.cost.pending) h += `<div class="check"><span class="ck">Outcome</span><span class="muted">Measuring on 5 test starts…</span></div>`;
  box.innerHTML = h;
}

// ---------------------------------------------------------------- 4 Diagnostic report
let reportTimer = null;
function loadReport() {
  if (!S.runId || !S.finished) return;
  clearTimeout(reportTimer);
  reportTimer = setTimeout(fetchReport, 150);
}
async function fetchReport() {
  const rid = S.runId;
  try {
    const r = await fetch(`/api/report/${encodeURIComponent(rid)}`);
    if (!r.ok) throw new Error(`${r.status}`);
    const j = await r.json();
    if (rid !== S.runId) return;
    S.report = j;
    if (typeof j.has_flaw === "boolean") S.config.hasFlaw = j.has_flaw;
    if (S.revealed && j.failing && j.failing.rule_id) { S.highlight = new Set([j.failing.rule_id]); renderLedger(); }
    renderReport(); renderChecks(); renderState();
    if (j.cost && j.cost.pending) { clearTimeout(reportTimer); reportTimer = setTimeout(fetchReport, 1500); }
  } catch (e) {
    if (!S.report) $("reportBody").innerHTML = `<div class="muted">Report not available yet (${esc(e.message)}).</div>`;
  }
}

function dl(href, text) { return `<a class="btn" href="${href}" download>${esc(text)}</a>`; }

function renderReport() {
  const r = S.report, rid = encodeURIComponent(S.runId);
  if (!r) return;
  const p = policyOf(S.config.bb);
  const prog = r.program || S.best || {};
  const nLines = prog.lines || (prog.code ? prog.code.trim().split("\n").length : null);
  const head = `<div class="doc-meta"><span>Policy <b>${esc(robotName(p))}</b></span><span>Scenario <b>${esc(scenarioName())}</b></span><span>Run <span class="mono">${esc(S.runId)}</span></span>${S.cached || S.kind === "replay" ? "<span>Source <b>earlier run</b></span>" : ""}</div>`;
  const sec = (n, title, body) => `<div class="doc-sec"><div class="doc-h"><span class="doc-n">${n}</span>${title}</div><div class="doc-b">${body}</div></div>`;
  const reg = r.regression || [];
  const regList = reg.length ? `<ol class="reglist">${reg.map((x) => `<li title="${esc(x.text)}">${esc(words(x.text, 16))}</li>`).join("")}</ol>` : '<div class="muted">No other confirmed rules.</div>';
  const downloads = (csv) => `<div class="doc-dl"><span class="muted">Download</span>${dl(`/api/report/${rid}.md`, "Full report (.md)")}${csv ? dl(`/api/report/${rid}/training.csv`, `Training set (.csv, ${fmtInt(csv)} rows)`) : ""}${dl(`/api/report/${rid}/tests.json`, "Regression tests (.json)")}</div>`;

  if (!r.has_flaw) {
    $("reportBody").innerHTML = head +
      sec(1, "Summary", `No failure mode was identified. The recovered policy${nLines ? ` (${nLines} lines)` : ""} matches the deployed network on <b>${fmtPct(r.match)}</b> of unseen situations.`) +
      sec(2, "Regression suite", `Behaviours that must remain unchanged after any retraining:${regList}`) +
      downloads(null);
    return;
  }
  const f = r.failing, rg = r.region, ev = r.evidence, act = r.action, gc = r.guard_check;
  const cost = r.cost && !r.cost.pending && !r.cost.error && r.cost.robot_reward != null ? r.cost : null;
  const pending = '<span class="muted">Pending remediation (step 3).</span>';
  let summary = `The recovered policy${nLines ? ` (${nLines} lines)` : ""} matches the deployed network on <b>${fmtPct(r.match)}</b> of unseen situations.`;
  if (f) summary += ` One failure mode was identified: ${esc((f.sentence || f.text).replace(/\.$/, ""))}.`;
  if (rg) summary += ` It is active on ${fmtPct(rg.fire_pct)} of operating steps.`;
  let h = head;
  h += sec(1, "Summary", summary);
  h += sec(2, "Failure mode", f ? `${esc(f.sentence || f.text)}${f.line ? `<div class="muted small">Location: line ${f.line} of the recovered policy · ledger rule ${esc(f.rule_id)}</div>` : ""}` : '<span class="muted">No confirmed rule matches a failure mode.</span>');
  h += sec(3, "Trigger region and frequency", rg ? `<span class="mono">${esc(rg.text)}</span> · active on ${fmtPct(rg.fire_pct)} of the network's steps · present in ${rg.episodes_hit} of ${rg.episodes_total} observed episodes` : "—");
  h += sec(4, "Operational impact", cost ? `Reward on the same ${cost.n} test starts: ${(+cost.robot_reward).toFixed(0)} without the fix, ${(+cost.guarded_reward).toFixed(0)} with it` + (cost.robot_success != null && cost.guarded_success != null ? `; task success ${cost.robot_success}/${cost.n} → ${cost.guarded_success}/${cost.n}` : "") + "." : r.cost && r.cost.pending ? '<span class="muted">Measuring…</span>' : pending);
  h += sec(5, "Evidence", ev ? esc(ev.text) : "—");
  h += sec(6, "Recommended retraining", act ? esc(act.text) : pending);
  h += sec(7, "Regression suite", `Behaviours that must remain unchanged after retraining:${regList}`);
  h += sec(8, "Runtime guard verification", gc ? esc(gc.text) + "." : pending);
  h += downloads(act ? act.n_training : null);
  $("reportBody").innerHTML = h;
}

// ---------------------------------------------------------------- simulations: loop in sync, hold the last frame
const player = { timer: null, hold: null, i: 0 };

function onFrames(d) {
  const clip = { bb: d.bb || [], clone: d.clone || [], disagree: d.disagree || [], fps: d.fps || 10,
                 version: d.version || 0, seed: d.seed, bb_end: d.bb_end || null, clone_end: d.clone_end || null };
  const first = !S.clip;
  S.clip = clip;                          // a newer version replaces the loop at once, both panels from frame 1
  if (first) pill("bbPill", "Running");
  startPlayer();
}

function bbFrames() { return S.viewFixed && S.shield.bb ? S.shield.bb : S.clip.bb; }
function bbEndInfo() { return S.viewFixed && S.shield.bb ? S.shield.bb_end : S.clip.bb_end; }
function cloneClip() {
  if (S.viewFixed && S.fixClip) return { frames: S.fixClip.clone, end: S.fixClip.clone_end, disagree: [] };
  return { frames: S.clip.clone, end: S.clip.clone_end, disagree: S.clip.disagree };
}

function startPlayer() {
  if (!S.clip) return;
  player.i = 0;
  clearInterval(player.timer); clearTimeout(player.hold);
  player.timer = setInterval(tick, 1000 / Math.min(S.clip.fps || 10, PLAY_FPS_MAX));
  $("bbEmpty").hidden = true;
  $("cloneEmpty").hidden = !!S.clip.clone.length;
  $("bbEnd").hidden = true; $("cloneEnd").hidden = true;
  tick();
}

function endLabel(end) {
  const fam = envFamily(policyOf(S.config.bb).env_id);
  if (!end) return "Episode over";
  const timeUp = end.truncated && !end.terminated;
  const at = end.steps ? ` (step ${end.steps})` : "";
  if (fam === "CartPole") return timeUp ? "Balanced for the full episode" : `Failed: the pole fell${at}`;
  if (fam === "MountainCar" || fam === "Acrobot") return timeUp ? "Did not reach the goal in time" : `Reached the goal${at}`;
  if (fam === "LunarLander") return timeUp ? "Time limit reached" : (end.last_reward || 0) > 0 ? `Landed${at}` : `Crashed${at}`;
  return "Episode over";
}

function tick() {
  const c = S.clip;
  if (!c) return;
  const cc = cloneClip();
  const bf = bbFrames(), n = Math.max(bf.length, cc.frames.length);
  const i = player.i;
  if (i >= n) {                           // hold the final frame with its label, then loop
    clearInterval(player.timer);
    if (bf.length) { $("bbEnd").textContent = endLabel(bbEndInfo()); $("bbEnd").hidden = false; }
    if (cc.frames.length) { $("cloneEnd").textContent = endLabel(cc.end); $("cloneEnd").hidden = false; }
    $("clonePanel").classList.remove("disagree");
    player.hold = setTimeout(startPlayer, HOLD_MS);
    return;
  }
  if (i < bf.length) { $("bbImg").src = "data:image/jpeg;base64," + bf[i]; $("bbEnd").hidden = true; }
  else if (bf.length) { $("bbEnd").textContent = endLabel(bbEndInfo()); $("bbEnd").hidden = false; }
  let red = false;
  if (cc.frames.length) {
    if (i < cc.frames.length) {
      $("cloneImg").src = "data:image/jpeg;base64," + cc.frames[i];
      red = !!cc.disagree[i];
      $("cloneEnd").hidden = true;
    } else { $("cloneEnd").textContent = endLabel(cc.end); $("cloneEnd").hidden = false; }
  } else $("cloneImg").removeAttribute("src");
  $("clonePanel").classList.toggle("disagree", red);
  $("frameNote").textContent = `start state #${c.seed ?? "—"} · frame ${i + 1}/${n}`;
  player.i = i + 1;
}

// ---------------------------------------------------------------- run control
async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || `${r.status} ${r.statusText}`);
  return j;
}

function softReset() {
  clearInterval(player.timer); clearTimeout(player.hold); player.timer = null;
  resetRun();
  $("banner").hidden = true;
  $("feed").innerHTML = '<li class="muted">No activity yet.</li>';
  $("ledger").innerHTML = '<div class="muted">No rules yet.</div>';
  $("subqLine").textContent = "Waiting for the first scan.";
  $("subqBars").innerHTML = "";
  $("code").innerHTML = '<span class="muted">The recovered program appears here.</span>';
  $("params").innerHTML = "";
  $("codeTitle").textContent = "Recovered policy";
  $("bbTitle").textContent = "Deployed policy";
  $("cloneTitle").textContent = "Recovered policy";
  $("clonePanel").classList.remove("guarded");
  $("bbPanel").classList.remove("guarded"); $("clonePanel").classList.remove("disagree");
  $("bbImg").removeAttribute("src"); $("cloneImg").removeAttribute("src");
  $("bbEmpty").hidden = false; $("cloneEmpty").hidden = false;
  $("bbEmpty").textContent = S.runId ? "Starting the simulation…" : "Press Run analysis to start the simulation";
  $("bbEnd").hidden = true; $("cloneEnd").hidden = true; $("frameNote").textContent = "";
  pill("bbPill", "Idle"); pill("clonePill", "Waiting");
  $("fixSteps").hidden = true; $("diff").hidden = true; $("fixExplain").hidden = true; $("checks").hidden = true;
  $("viewToggle").hidden = true;
  $("reportBody").innerHTML = '<div class="muted">Generated when the analysis completes.</div>';
  renderCurve(); renderKpis(); renderState();
}

function attach(runId, kind, config) {
  if (S.ws) { S.ws.onclose = null; S.ws.close(); }
  S.runId = runId; S.kind = kind; S.config = config || {};
  softReset();
  const params = new URLSearchParams(location.search);
  params.set("run", runId);
  history.replaceState(null, "", `?${params}`);
  connect(runId);
}

function connect(runId) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/${runId}`);
  S.ws = ws;
  ws.onmessage = (m) => { try { apply(JSON.parse(m.data)); } catch (e) { console.error(e); } };
  ws.onopen = () => softReset();          // the server sends the whole buffer first
  ws.onclose = () => { if (S.runId === runId && !S.finished) setTimeout(() => { if (S.runId === runId) connect(runId); }, 1000); };
}

async function onRun() {
  const bb = $("selPolicy").value;
  const twist = $("selFlaw").value || null;
  $("btnRun").disabled = true;
  try {
    const r = await post("/api/run", { bb, twist_id: twist, brain: "muse", live: true });
    attach(r.run_id, "live", { bb, twist_id: twist, hasFlaw: !!twist });
    logLine("Starting the simulation", 0);
  } catch (e) {
    $("btnRun").disabled = false;
    showBanner(e.message);
  }
}

function parseCacheKey(key) {
  const [bb, tw] = String(key).split("__");
  return { bb, twist_id: tw && tw !== "none" ? tw : null };
}

async function onBackup(key) {
  $("backup").open = false;
  const cfg = parseCacheKey(key);
  try {
    const r = await post("/api/replay", { cache_key: key, speed: 1, max_seconds: 20 });
    attach(r.run_id, "replay", { ...cfg, hasFlaw: !!cfg.twist_id });
    if ($("selPolicy").value !== cfg.bb) { $("selPolicy").value = cfg.bb; await loadTwists(); }
    $("selFlaw").value = cfg.twist_id || "";
    showBanner("Showing an earlier run of this exact configuration.");
  } catch (e) { showBanner(`Could not load the earlier run: ${e.message}`); }
}

// ---------------------------------------------------------------- Results tab: scientific summary
function showTab(tab) {
  const results = tab === "results";
  $("tabLive").classList.toggle("active", !results);
  $("tabResults").classList.toggle("active", results);
  document.querySelectorAll(".view-live").forEach((el) => (el.hidden = results));
  $("results").hidden = !results;
  $("scroll").scrollTop = 0;
  if (results) loadResults();
}

const median = (xs) => { const v = xs.filter((x) => x != null).sort((a, b) => a - b); if (!v.length) return null; const m = v.length >> 1; return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2; };
const mean = (xs) => { const v = xs.filter((x) => x != null); return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null; };
const pretty = (label) => robotName({ label });
const twistName = (t) => (!t || t === "none" ? "Normal" : `Flaw ${flawNum(t)}`);
const fmtR = (v) => (v == null ? "—" : (+v).toFixed(0));

function htable(head, rows, cls = () => "", numCols = []) {
  return `<table class="sci"><thead><tr>${head.map((h, i) => `<th class="${numCols.includes(i) ? "num" : ""}">${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r, ri) =>
    `<tr class="${cls(ri)}">${r.map((c, i) => `<td class="${numCols.includes(i) ? "num" : ""}">${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

async function loadResults() {
  const box = $("results");
  box.innerHTML = '<div class="muted">Loading…</div>';
  let j;
  try { j = await (await fetch("/api/results")).json(); } catch (e) { box.innerHTML = `<div class="muted">Could not load results: ${esc(e.message)}</div>`; return; }
  if (!j.available) { box.innerHTML = '<div class="muted">No batch results yet (python -m scripts.batch --snap 1,0).</div>'; return; }
  const R = j.results;
  const shownBB = new Set(RESULT_POLICIES.map((p) => p.bb));
  const rows = ((R.summary && R.summary.rows) || []).filter((r) => shownBB.has(r.bb));
  const flawRows = rows.filter((r) => r.twist && r.twist !== "none");
  const on = rows.filter((r) => r.snap === true);
  const flawOn = flawRows.filter((r) => r.snap === true), flawOff = flawRows.filter((r) => r.snap === false);
  const foundOn = flawOn.filter((r) => r.twist_found).length, foundOff = flawOff.filter((r) => r.twist_found).length;

  const pairs = {};
  flawRows.forEach((r) => { (pairs[`${r.bb}|${r.twist}`] = pairs[`${r.bb}|${r.twist}`] || {})[r.snap ? "on" : "off"] = r; });
  const pairList = Object.values(pairs).filter((p) => p.on && p.off);
  const both = pairList.filter((p) => p.on.queries_to_twist != null && p.off.queries_to_twist != null);
  const totOn = both.reduce((s, p) => s + p.on.queries_to_twist, 0), totOff = both.reduce((s, p) => s + p.off.queries_to_twist, 0);
  const ratios = both.map((p) => p.off.queries_to_twist / Math.max(1, p.on.queries_to_twist));
  const maxRatio = ratios.length ? Math.max(...ratios) : null;
  const cut = totOff ? 1 - totOn / totOff : null;

  const cm = Object.fromEntries(Object.entries(R.crossmatch || {})
    .filter(([prog]) => shownBB.has(prog.replace(/ program$/, "")))
    .map(([prog, cols]) => [prog, Object.fromEntries(Object.entries(cols || {}).filter(([bb]) => shownBB.has(bb)))]));
  const diag = [], offd = [];
  Object.entries(cm).forEach(([prog, cols]) => { const own = prog.replace(/ program$/, ""); Object.entries(cols).forEach(([pol, v]) => (pol === own ? diag : offd).push(v)); });
  const base = ((R.baselines && R.baselines.rows) || []).filter((r) => shownBB.has(r.bb));
  const medOn = median(on.map((r) => r.agreement)), medBase = median(base.map((r) => r.agreement));
  const medSpeed = median(rows.map((r) => r.speedup));
  const robots = [...new Set(rows.map((r) => pretty(r.label)))];

  let h = `<section class="paper">
    <h2>Summary of results</h2>
    <p class="abstract"><b>Abstract.</b> We evaluated Unbox on ${robots.length} trained control policies (${esc(robots.join(", "))}) in ${rows.length} batch runs, each repeated with and without SubQ Snap evidence scanning.
    The recovered programs agree with the network on a median of <b>${fmtPct(medOn)}</b> of held-out states (SubQ on), against ${fmtPct(medBase)} for a controller written by Muse without querying the network.
    Unbox identified <b>${foundOn} of ${flawOn.length}</b> planted failure modes with SubQ (${foundOff} of ${flawOff.length} without)${cut != null ? `, using <b>${Math.round(100 * cut)}% fewer</b> queries to detection` : ""}.
    Recovered programs execute a median <b>${medSpeed != null ? medSpeed.toFixed(0) : "—"}×</b> faster than the networks.</p>`;
  const t1 = [
    ["Failure-mode detection", `${foundOn} / ${flawOn.length}`, `${foundOff} / ${flawOff.length} without SubQ`, `${flawOn.length} planted flaws`],
    ["Behavioural fidelity (median held-out agreement)", fmtPct(medOn), `${fmtPct(medBase)} zero-query baseline`, `${on.length} runs`],
    ["Queries to detection (total)", fmtInt(totOn), `${fmtInt(totOff)} without SubQ (${cut != null ? "−" + Math.round(100 * cut) + "%" : "—"}; max ${maxRatio != null ? maxRatio.toFixed(1) : "—"}× on one flaw)`, `${both.length} flaws found both ways`],
    ["Policy specificity (agreement, own vs sibling policy)", fmtPct(mean(diag)), `${fmtPct(mean(offd))} on the sibling policy`, `${diag.length} programs`],
    ["Inference speed-up (median)", medSpeed != null ? `${medSpeed.toFixed(0)}×` : "—", "readable program vs network forward pass", `${rows.length} runs`],
  ];
  h += `<div class="tcap"><b>Table 1.</b> Headline measures.</div>` + htable(["Measure", "Result", "Comparison", "n"], t1, () => "", [1]);
  h += `<p class="fn">Agreement: share of states from 8 unseen episodes where the program selects the network's action. Detection: a confirmed ledger rule names the flaw's sensor with a threshold within 20% of the true one. Source: results/summary.json, crossmatch.json, baselines.json.</p></section>`;

  const subqRows = pairList.map((p) => [esc(pretty(p.on.label)), twistName(p.on.twist),
    p.on.queries_to_twist != null ? fmtInt(p.on.queries_to_twist) : "not found", p.off.queries_to_twist != null ? fmtInt(p.off.queries_to_twist) : "not found",
    p.on.turns_to_twist ?? "—", p.off.turns_to_twist ?? "—"]);
  subqRows.push(["Total", `${both.length} flaws found both ways`, fmtInt(totOn), fmtInt(totOff), "", ""]);
  h += `<section class="paper"><div class="tcap"><b>Table 2.</b> Queries and turns to detection, with and without SubQ Snap. Same policy, flaw and brain; only the evidence scan differs.</div>
    ${htable(["Policy", "Scenario", "Queries, SubQ", "Queries, no SubQ", "Turns, SubQ", "Turns, no SubQ"], subqRows, (i) => (i === subqRows.length - 1 ? "total" : ""), [2, 3, 4, 5])}</section>`;

  const sumRows = rows.map((r) => [esc(pretty(r.label)), twistName(r.twist), r.snap ? "on" : "off", fmtPct(r.agreement), `${fmtR(r.clone_reward)} / ${fmtR(r.bb_reward)}`,
    r.lines_of_code ?? "—", fmtInt(r.queries), r.speedup != null ? `${(+r.speedup).toFixed(0)}×` : "—",
    !r.twist || r.twist === "none" ? "—" : r.twist_found ? "detected" : "missed"]);
  h += `<section class="paper"><div class="tcap"><b>Table 3.</b> All runs. Reward is program / network on the same 20 start states. Rows in red are missed detections.</div>
    ${htable(["Policy", "Scenario", "SubQ", "Agreement", "Reward", "Lines", "Queries", "Speed-up", "Flaw"], sumRows,
      (i) => (rows[i].twist && rows[i].twist !== "none" && rows[i].twist_found === false ? "fail" : ""), [3, 4, 5, 6, 7])}</section>`;

  const progs = Object.keys(cm), pols = [...new Set(progs.flatMap((p) => Object.keys(cm[p] || {})))];
  const lab = (bb) => esc(pretty(policyOf(bb).label));
  if (progs.length) h += `<section class="paper"><div class="tcap"><b>Table 4.</b> Cross-policy agreement. Each recovered program evaluated on its own policy (diagonal) and on a sibling policy trained with a different algorithm.</div>
    ${htable(["Program from ↓ / evaluated on →", ...pols.map(lab)], progs.map((p) => [lab(p.replace(/ program$/, "")), ...pols.map((q) => {
      const v = (cm[p] || {})[q]; const own = p.replace(/ program$/, "") === q;
      return v == null ? "—" : own ? `<b>${fmtPct(v)}</b>` : fmtPct(v);
    })]), () => "", pols.map((_, i) => i + 1))}</section>`;

  const ownMatch = (bb) => { const r = rows.find((x) => x.bb === bb && (!x.twist || x.twist === "none") && x.snap); return r ? r.agreement : null; };
  h += `<section class="paper"><div class="tcap"><b>Table 5.</b> Zero-query baseline: Muse given only the task name and sensor list, without querying the network.</div>
    ${htable(["Policy", "Agreement, baseline", "Agreement, Unbox", "Reward baseline / network", "Lines"], base.map((b) => [esc(pretty(b.label)), fmtPct(b.agreement), fmtPct(ownMatch(b.bb)), `${fmtR(b.clone_reward)} / ${fmtR(b.bb_reward)}`, b.lines_of_code ?? "—"]), () => "", [1, 2, 3, 4])}</section>`;
  box.innerHTML = h;
}

// ---------------------------------------------------------------- boot
async function getJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); }

function renderPolicyDesc() {
  const p = policyOf($("selPolicy").value);
  $("policyDesc").textContent = ROBOTS[envFamily(p.env_id)] || "A trained robot controller";
}

async function loadTwists() {
  renderPolicyDesc();
  const bb = $("selPolicy").value;
  const sel = $("selFlaw");
  try {
    const j = await getJSON(`/api/twists?bb=${encodeURIComponent(bb)}`);
    const tw = j.twists.filter((t) => !HIDE_FROM_MENU.has(t.twist_id));
    sel.innerHTML = `<option value="">Normal operation (no planted flaw)</option>` + tw.map((t) =>
      `<option value="${esc(t.twist_id)}">Planted flaw ${flawNum(t.twist_id)}</option>`).join("");
  } catch (e) { sel.innerHTML = '<option value="">Normal operation (no planted flaw)</option>'; }
}

async function loadBackups() {
  try {
    const j = await getJSON("/api/sources");
    const items = (j.cache || []).map((k) => {
      const c = parseCacheKey(k);
      if (c.twist_id && HIDE_FROM_MENU.has(c.twist_id)) return null;
      if (!POLICIES.some((p) => p.bb === c.bb)) return null;
      return `<button class="linkbtn" data-key="${esc(k)}">${esc(robotName(policyOf(c.bb)))} · ${c.twist_id ? `planted flaw ${flawNum(c.twist_id)}` : "normal operation"}</button>`;
    }).filter(Boolean);
    $("backupList").innerHTML = items.join("") || '<div class="muted small">No earlier runs.</div>';
    $("backupList").querySelectorAll("button").forEach((b) => (b.onclick = () => onBackup(b.dataset.key)));
  } catch (e) { $("backupList").innerHTML = '<div class="muted small">Could not list earlier runs.</div>'; }
}

async function boot() {
  renderCurve(); renderKpis(); renderState();
  $("btnRun").onclick = onRun;
  $("btnFix").onclick = onFixClick;
  $("viewOrig").onclick = () => setView(false);
  $("viewFixed").onclick = () => setView(true);
  $("tabLive").onclick = () => showTab("live");
  $("tabResults").onclick = () => showTab("results");
  $("selPolicy").onchange = loadTwists;
  document.querySelectorAll("#secNav a").forEach((a) => (a.onclick = (e) => {
    e.preventDefault();
    const el = document.querySelector(a.getAttribute("href"));
    if (el) $("scroll").scrollTo({ top: el.offsetTop - 8, behavior: "smooth" });
  }));
  try {
    const [cfg, pol] = await Promise.all([getJSON("/api/config"), getJSON("/api/policies")]);
    RESULT_POLICIES = (pol.policies || []).filter((p) => SHOWN_ENVS.has(envFamily(p.env_id)));
    POLICIES = RESULT_POLICIES.filter((p) => !HIDDEN_POLICIES.has(p.bb));
    $("selPolicy").innerHTML = POLICIES.map((p) => `<option value="${esc(p.bb)}">${esc(robotName(p))}</option>`).join("");
    if (POLICIES.some((p) => p.bb === "bb-01")) $("selPolicy").value = "bb-01";
    if (pol.warning) showBanner(pol.warning);
    else if (cfg.use_mocks) showBanner("Mock mode (USE_MOCKS=1).");
  } catch (e) { showBanner(`Could not reach the app server: ${e.message}`); }
  await Promise.all([loadTwists(), loadBackups()]);

  const q = new URLSearchParams(location.search);
  if (q.get("tab") === "results") showTab("results");
  if (q.get("run")) {
    const rid = q.get("run");
    let cfg = {}, kind = "attached";
    try {
      const st = await getJSON(`/api/runs/${encodeURIComponent(rid)}`);
      cfg = { ...(st.config || {}), hasFlaw: !!(st.config && st.config.twist_id) };
      kind = st.kind === "replay" ? "replay" : "live";
    } catch (e) { /* unknown run: the socket reports it */ }
    S.runId = rid; S.kind = kind; S.config = cfg;
    if (cfg.bb && POLICIES.some((p) => p.bb === cfg.bb)) { $("selPolicy").value = cfg.bb; await loadTwists(); $("selFlaw").value = cfg.twist_id || ""; }
    connect(rid);
  }
}

boot();
