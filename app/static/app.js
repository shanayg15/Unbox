// Unbox UI. Vanilla JS. Everything on screen is derived from the event stream (contracts/events.py)
// plus /api/report for the Report card, so a live run, a replay and a reload all render the same way.
"use strict";

const $ = (id) => document.getElementById(id);
const GLOW_MS = 3000;
const PLAY_FPS_MAX = 10;                 // clips play once, at most this fast, then hold the last frame
const HIDE_FROM_MENU = new Set(["t-mc-1"]); // not reliably found (kept in Results as the honest failure)

const ROBOTS = {
  CartPole: "balances a pole on a moving cart",
  LunarLander: "lands a craft between two flags",
  MountainCar: "drives an underpowered car up a hill",
  Acrobot: "swings a two-link arm above a line",
};

const S = {};                   // run state, rebuilt from events
window.__unbox = S;
let POLICIES = [];

function resetRun() {
  Object.assign(S, {
    finished: false, cached: false, banner: null,
    budget: 5000, queries: 0, metrics: [], best: null, shownCode: null,
    ledger: [], highlight: new Set(), snaps: [],
    fix: null, revealed: null, report: null, finishedData: null,
    clip: null, pendingClip: null, shield: { on: false, bb: null, bb_end: null }, lastStatus: "",
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
function policyOf(bb) { return POLICIES.find((p) => p.bb === bb) || { bb, label: bb, env_id: "" }; }
function robotName(p) {
  const [env, algo] = String(p.label || p.bb).split("-");
  return algo ? `${env}, ${algo}` : env;
}
const flawNum = (tid) => (tid ? (String(tid).match(/(\d+)$/) || [, "?"])[1] : null);
const matchNow = () => (S.best ? S.best.agreement : null);

// ---------------------------------------------------------------- event dispatch
function apply(ev) {
  const d = ev.data || {};
  switch (ev.type) {
    case "run_started": S.budget = d.budget || S.budget; break;
    case "status":
      if (d.banner) { softReset(); showBanner(d.text); S.cached = true; }
      setStatus(d.text);
      break;
    case "error": setStatus(d.message, true); break;
    case "query_count": S.queries = Math.max(S.queries, d.n || 0); renderLearning(); break;
    case "metric":
      S.metrics.push({ q: d.queries, a: d.agreement });
      S.queries = Math.max(S.queries, d.queries || 0);
      renderLearning();
      break;
    case "ledger": S.ledger = d.entries || []; renderLedger(); renderFinding(); break;
    case "snap": S.snaps.push(d); renderSnap(); renderFinding(); break;
    case "program": onProgram(d); break;
    case "frames": onFrames(d); break;
    case "reveal": onReveal(d); break;
    case "fix": onFix(d); break;
    case "shield": onShield(d); break;
    case "run_finished":
      S.finished = true; S.cached = S.cached || !!d.cached; S.finishedData = d;
      if (d.total_queries != null) S.queries = Math.max(S.queries, d.total_queries);
      onFinished(d);
      break;
    default: break;
  }
  updateButtons();
}

// ---------------------------------------------------------------- status / banner
function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text || "";
  el.title = text || "";
  el.classList.toggle("error", !!isError);
  S.lastStatus = text;
}
function showBanner(text) {
  S.banner = text;
  $("banner").textContent = text;
  $("banner").hidden = false;
  renderBadge();
}
function renderBadge() {
  const b = $("runBadge");
  if (!S.runId) { b.textContent = ""; return; }
  let tag = S.kind === "replay" ? "Backup replay" : S.cached ? "Earlier run" : S.finished ? "Live run · finished" : "Live run · running";
  b.textContent = tag;
  b.title = S.runId;
  b.classList.toggle("cached", S.kind === "replay" || S.cached);
}
function onFinished(d) {
  const m = matchNow();
  const parts = [`Finished. Our copy matches the robot on ${fmtPct(m)} of new situations`];
  if (d.total_queries != null) parts.push(`${fmtInt(d.total_queries)} queries`);
  if (d.turns != null) parts.push(`${d.turns} turns`);
  if (d.seconds != null && !S.cached) parts.push(`${Math.round(d.seconds)} s`);
  setStatus(parts.join(" · "));
  renderLearning();
  renderFinding();
  loadReport();
}

// ---------------------------------------------------------------- 2 Learning
function renderLearning() {
  const m = matchNow();
  $("learnLine").innerHTML = `Queries asked: <b>${fmtInt(S.queries)}</b> · ` +
    (m == null ? "no copy yet" : `Our copy matches the robot on <b>${fmtPct(m)}</b> of new situations`);
  renderCurve();
}

function renderCurve() {
  const svg = $("curve");
  const W = 900, H = 230, L = 54, R = 24, T = 20, B = 40;
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
  g += `<text class="lbl" x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">queries asked</text>`;
  if (best.length) {
    let path = `M${x(0)},${y(best[0].a)} `;
    best.forEach((p, i) => {
      if (i > 0) path += `L${x(p.q)},${y(best[i - 1].a)} `;
      path += `L${x(p.q)},${y(p.a)} `;
    });
    g += `<path class="line" d="${path}"/>`;
    best.forEach((p) => (g += `<circle class="pt" cx="${x(p.q)}" cy="${y(p.a)}" r="3.5"/>`));
    const last = best[best.length - 1];
    const lx = x(last.q), ly = y(last.a);
    if (S.finished) g += `<circle class="final" cx="${lx}" cy="${ly}" r="7"/>`;
    const label = `${S.finished ? "final " : ""}${fmtPct(last.a)}`;
    const anchor = lx > W - 140 ? "end" : "start";
    const tx = anchor === "end" ? lx - 12 : lx + 12;
    const ty = ly > T + 30 ? ly - 14 : ly + 26;
    g += `<text class="val" x="${tx}" y="${ty}" text-anchor="${anchor}">${label}</text>`;
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
  const cause = an.choice && an.choice !== "none" ? `most likely cause: <b>${esc(an.choice)}</b>` : "no clear cause yet";
  $("subqLine").innerHTML = `SubQ read ${fmtInt(s.n_rows)} raw rows${tokens} · ${cause}`;
  const probs = an.probabilities || {};
  const items = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  $("subqBars").innerHTML = items.map(([k, p]) => `<div class="bar${k === an.choice ? " top" : ""}">
      <span class="name">${esc(k)}</span><span class="track"><span class="fill" style="width:${Math.max(1, 100 * p)}%"></span></span><span class="p">${(+p).toFixed(2)}</span></div>`).join("");
}

// ---------------------------------------------------------------- 3 Finding: ledger, code, finding card
const ruleNum = (rid) => parseInt(String(rid).replace(/\D/g, ""), 10) || 0;
const STATUS_WORD = { observed: "seen in data", inferred: "hypothesis", probing: "testing now" };

function ruleRow(e, label) {
  const cls = ["lrow"];
  if (S.highlight.has(e.rule_id)) cls.push("highlight");
  return `<div class="${cls.join(" ")}" title="${esc(e.text)}"><span class="rid">${esc(e.rule_id)}</span><span class="text">${esc(words(e.text))}</span>${label ? `<span class="st">${esc(label)}</span>` : ""}</div>`;
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
  if (test.length) h += `<div class="lgroup"><div class="lhead">Testing <span class="count">${test.length}</span></div>${test.map((e) => ruleRow(e, STATUS_WORD[e.status])).join("")}</div>`;
  if (out.length) h += `<details class="lgroup ruled"${wasOpen ? " open" : ""}><summary class="lhead">Ruled out (${out.length})</summary>${out.map((e) => ruleRow(e)).join("")}</details>`;
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
    setStatus(`Version ${d.version} matched ${fmtPct(d.agreement)}, below v${S.best.version}; keeping v${S.best.version}`);
    return;
  }
  S.best = d;
  renderCodeCard(true);
  renderLearning();
}

function renderCodeCard(glow) {
  const d = S.best;
  if (!d) return;
  const lines = d.code.replace(/\s+$/, "").split("\n");
  const changed = new Set();
  if (glow && S.shownCode != null) {
    let ln = 0;
    lineDiff(S.shownCode.split("\n"), lines).forEach((o) => { if (o.op !== "del") { ln++; if (o.op === "add") changed.add(ln); } });
  } else if (glow) (d.changed_lines || []).forEach((n) => changed.add(n));
  S.shownCode = lines.join("\n");
  $("codeTitle").textContent = `Our copy · v${d.version} · ${lines.length} lines · ${fmtPct(d.agreement)}`;
  $("code").innerHTML = lines.map((l, i) =>
    `<div class="cl${changed.has(i + 1) ? " glow" : ""}"><span class="ln">${i + 1}</span><span class="src">${highlightLine(l)}</span></div>`).join("");
  $("params").innerHTML = Object.entries(d.params || {}).map(([k, v]) => `<span>${esc(k)} = ${(+v).toFixed(3)}</span>`).join("");
  clearTimeout(S.glowTimer);
  S.glowTimer = setTimeout(() => $("code").querySelectorAll(".cl.glow").forEach((el) => el.classList.remove("glow")), GLOW_MS);
}

function topSuspect() {
  const r = S.report;
  if (r && r.failing) return r.failing.text;
  const conf = S.ledger.filter((e) => e.status === "confirmed");
  if (!conf.length) return null;
  const snap = S.snaps[S.snaps.length - 1];
  const f = snap && snap.anomaly_feature && snap.anomaly_feature.choice;
  const hit = f && f !== "none" ? conf.filter((e) => e.text.toLowerCase().includes(f.toLowerCase())) : [];
  return (hit.length ? hit[hit.length - 1] : conf[conf.length - 1]).text;
}

function renderFinding() {
  const box = $("findingBody");
  const r = S.report;
  if (!S.runId) { box.innerHTML = '<span class="muted">The conclusion appears here once Unbox has confirmed rules.</span>'; return; }
  if (!hasFlaw()) {
    box.innerHTML = S.finished
      ? `<div class="concl">No planted flaw in this run. Our copy matches the robot on <b>${fmtPct(matchNow())}</b> of new situations.</div>`
      : '<div class="concl muted">No planted flaw in this run. Unbox is learning the robot\'s normal behavior.</div>';
    return;
  }
  if (!S.revealed) {
    const t = topSuspect();
    box.innerHTML = t ? `<div class="k">Top suspect</div><div class="concl" title="${esc(t)}">${esc(t)}</div>`
                      : '<div class="concl muted">No confirmed rule yet.</div>';
    return;
  }
  const found = r && r.failing ? (r.failing.sentence || r.failing.text) : null;
  const badge = !r ? '<span class="badge wait">checking…</span>'
    : r.found === "match" ? '<span class="badge ok">Match</span>'
    : '<span class="badge no">Not found</span>';
  box.innerHTML = `<div class="two">
      <div><div class="k">Reported flaw</div><div class="concl">${esc(S.revealed.label)}</div></div>
      <div><div class="k">Unbox found</div><div class="concl">${found ? esc(found) : '<span class="muted">no matching confirmed rule</span>'}</div></div>
    </div><div class="badgerow">${badge}${r && r.failing && r.failing.line ? `<span class="muted small">line ${r.failing.line} of Our copy</span>` : ""}</div>`;
}

function onReveal(d) {
  S.revealed = d;
  S.highlight = new Set((d.highlight_rule_ids || []).slice(0, 1));
  const sel = $("selFlaw");
  const opt = sel.options[sel.selectedIndex];
  if (opt && opt.value) opt.textContent = `Planted flaw ${flawNum(opt.value)}: ${d.label}`;
  renderLedger();
  renderFinding();
  setStatus(`Revealed: "${d.label}"`);
  loadReport();
}

// ---------------------------------------------------------------- 4 Fix
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
  $("fixExplain").innerHTML = `<b>Muse's fix: ${nDel} line${nDel === 1 ? "" : "s"} removed, ${nAdd} added.</b> ${esc(d.explanation || "")}`;
  $("fixExplain").hidden = false;
  setStatus("Our copy is fixed. Turn on the guard to protect the real network.");
  loadReport();
}

function onShield(d) {
  S.shield = { on: !!d.on, bb: d.on ? d.bb : null, bb_end: d.on ? d.bb_end : null };
  $("chkShield").checked = S.shield.on;
  $("bbPanel").classList.toggle("guarded", S.shield.on);
  $("bbTitle").textContent = S.shield.on ? "The robot (real network) + guard" : "The robot (real network)";
  setStatus(S.shield.on ? "Guard on: the robot's own actions, corrected only where the fixed copy disagrees with the flawed copy"
                        : "Guard off: the robot's raw actions");
  if (S.clip) startPlayer();
  renderGuard();
  loadReport();
}

function renderGuard() {
  const r = S.report;
  const g = r && r.guard_check;
  $("guardCheck").hidden = !(S.shield.on && g);
  if (g) $("guardCheck").textContent = g.text;
  const c = r && r.cost && !r.cost.pending && !r.cost.error && r.cost.robot_reward != null ? r.cost : null;
  $("guardCost").hidden = !(S.shield.on && c);
  if (c) $("guardCost").textContent = `Reward on the same ${c.n || (c.seeds || []).length} test starts: robot alone ${(+c.robot_reward).toFixed(0)} → with guard ${(+c.guarded_reward).toFixed(0)}` +
    (c.robot_success != null && c.guarded_success != null ? ` · success ${c.robot_success}/${c.n} → ${c.guarded_success}/${c.n}` : "");
}

// ---------------------------------------------------------------- 5 Report
let reportTimer = null;
function loadReport() {
  if (!S.runId || !S.finished) return;
  clearTimeout(reportTimer);
  reportTimer = setTimeout(fetchReport, 150);
}
async function fetchReport() {
  const rid = S.runId;
  if (!S.report) $("reportBody").innerHTML = '<div class="muted">Building the report…</div>';
  try {
    const r = await fetch(`/api/report/${encodeURIComponent(rid)}`);
    if (!r.ok) throw new Error(`${r.status}`);
    const j = await r.json();
    if (rid !== S.runId) return;
    S.report = j;
    if (typeof j.has_flaw === "boolean") S.config.hasFlaw = j.has_flaw;
    if (S.revealed && j.failing && j.failing.rule_id) { S.highlight = new Set([j.failing.rule_id]); renderLedger(); }
    renderReport(); renderFinding(); renderGuard(); updateButtons();
    if (j.cost && j.cost.pending) { clearTimeout(reportTimer); reportTimer = setTimeout(fetchReport, 1500); }
  } catch (e) {
    if (!S.report) $("reportBody").innerHTML = `<div class="muted">Report not available yet (${esc(e.message)}).</div>`;
  }
}

function dl(href, text) { return `<a class="btn" href="${href}" download>${esc(text)}</a>`; }

function renderReport() {
  const r = S.report, rid = encodeURIComponent(S.runId);
  const box = $("reportBody");
  if (!r) return;
  const md = dl(`/api/report/${rid}.md`, "Download full report (Markdown)");
  const tests = dl(`/api/report/${rid}/tests.json`, "Download regression tests (JSON)");
  const reg = (r.regression || []);
  const regList = reg.length ? `<ul>${reg.map((x) => `<li title="${esc(x.text)}">${esc(words(x.text, 16))}</li>`).join("")}</ul>` : '<div class="muted">No other confirmed rules.</div>';
  if (!r.has_flaw) {
    box.innerHTML = `<div class="rrow"><div class="rk">Result</div><div class="rv">${esc(r.no_flaw_text || `No failing decision found. Readable copy matches the robot on ${fmtPct(r.match)}`)}</div></div>
      <div class="rrow"><div class="rk">Regression tests</div><div class="rv">These behaviors must not change after any retraining:${regList}<div class="btns">${tests}</div></div></div>
      <div class="btns">${md}</div>`;
    return;
  }
  const f = r.failing, reg_ = r.region, ev = r.evidence, act = r.action, gc = r.guard_check;
  const cost = r.cost && !r.cost.pending && !r.cost.error && r.cost.robot_reward != null ? r.cost : null;
  const costNote = r.cost && r.cost.pending ? "Measuring robot vs guarded robot on 5 test starts…"
    : S.kind === "replay" || S.cached ? "Measured on live runs only. Run live and turn on the guard."
    : 'Turn on "Guard the real network" to measure it.';
  const row = (k, v) => `<div class="rrow"><div class="rk">${k}</div><div class="rv">${v}</div></div>`;
  let h = "";
  h += row("Failing decision", f ? `${esc(f.sentence || f.text)}${f.line ? ` <span class="muted">(line ${f.line} in Our copy)</span>` : ""}` : '<span class="muted">No confirmed rule matches a flaw yet.</span>');
  h += row("Where and how often", reg_ ? `${esc(reg_.text)} · fires on ${fmtPct(reg_.fire_pct)} of the robot's steps, in ${reg_.episodes_hit} of ${reg_.episodes_total} of its own runs` : '<span class="muted">—</span>');
  h += row("What it costs", cost ? `Reward on the same ${cost.n || (cost.seeds || []).length} test starts: robot ${(+cost.robot_reward).toFixed(0)} vs guarded robot ${(+cost.guarded_reward).toFixed(0)}` +
      (cost.robot_success != null && cost.guarded_success != null ? ` · success ${cost.robot_success}/${cost.n} → ${cost.guarded_success}/${cost.n}` : "")
    : `<span class="muted">${esc(costNote)}</span>`);
  h += row("Evidence", ev ? esc(ev.text) : "—");
  h += row("Post-training action", act ? `${esc(act.text)}<div class="btns">${dl(`/api/report/${rid}/training.csv`, `Download targeted training set (${fmtInt(act.n_training)} labeled situations, CSV)`)}</div>`
    : '<span class="muted">Press "Fix our copy" to get the corrected action and the training set.</span>');
  h += row("Regression tests", `After retraining, these behaviors must not change:${regList}<div class="btns">${tests}</div>`);
  h += row("Guard check", gc ? esc(gc.text) : '<span class="muted">Available after the fix.</span>');
  h += `<div class="btns">${md}</div>`;
  box.innerHTML = h;
}

// ---------------------------------------------------------------- clips: play once, hold the last frame
const player = { timer: null, i: 0 };

function onFrames(d) {
  const clip = { bb: d.bb || [], clone: d.clone || [], disagree: d.disagree || [], fps: d.fps || 10,
                 version: d.version || 0, seed: d.seed, bb_end: d.bb_end || null, clone_end: d.clone_end || null };
  if (player.timer && S.clip && S.clip.version === 0 && clip.version > 0) { S.clip = clip; startPlayer(); return; }
  if (player.timer) { S.pendingClip = clip; return; }   // never swap mid-clip
  S.clip = clip;
  startPlayer();
}

function startPlayer() {
  if (!S.clip) return;
  player.i = 0;
  clearInterval(player.timer);
  const fps = Math.min(S.clip.fps || 10, PLAY_FPS_MAX);
  player.timer = setInterval(tick, 1000 / fps);
  $("bbEmpty").hidden = true;
  $("cloneEmpty").hidden = !!S.clip.clone.length;
  $("cloneEmpty").textContent = "Waiting for the first copy…";
  $("btnReplayClip").disabled = false;
  tick();
}

function endLabel(end) {
  if (!end) return "Episode ended";
  if (end.truncated && !end.terminated) return "Ran the full episode";
  const fam = envFamily(policyOf(S.config.bb).env_id);
  if (fam === "CartPole") return "Episode ended (crashed)";
  if (fam === "MountainCar" || fam === "Acrobot") return "Episode ended (reached goal)";
  if (fam === "LunarLander") return (end.last_reward || 0) > 0 ? "Episode ended (reached goal)" : "Episode ended (crashed)";
  return "Episode ended";
}

function tick() {
  const c = S.clip;
  if (!c) return;
  const bbFrames = S.shield.on && S.shield.bb ? S.shield.bb : c.bb;
  const bbEnd = S.shield.on && S.shield.bb ? S.shield.bb_end : c.bb_end;
  const n = Math.max(bbFrames.length, c.clone.length);
  const i = player.i;
  if (i >= n) { stopPlayer(); return; }
  if (i < bbFrames.length) { $("bbImg").src = "data:image/jpeg;base64," + bbFrames[i]; $("bbEnd").hidden = true; }
  else { $("bbEnd").textContent = endLabel(bbEnd); $("bbEnd").hidden = false; }
  let red = false;
  if (c.clone.length) {
    if (i < c.clone.length) {
      $("cloneImg").src = "data:image/jpeg;base64," + c.clone[i];
      red = !S.shield.on && !!c.disagree[i];
      $("cloneEnd").hidden = true;
    } else { $("cloneEnd").textContent = endLabel(c.clone_end); $("cloneEnd").hidden = false; }
  } else {
    $("cloneImg").removeAttribute("src"); $("cloneEnd").hidden = true;
  }
  $("clonePanel").classList.toggle("disagree", red);
  $("frameNote").textContent = `frame ${i + 1} / ${n}`;
  $("bbMeta").textContent = `start state #${c.seed ?? "—"}`;
  if (c.clone.length) {
    const nRed = c.disagree.filter(Boolean).length;
    $("cloneMeta").textContent = `v${c.version} · differs on ${nRed} of ${c.clone.length} frames`;
  } else $("cloneMeta").textContent = "";
  player.i = i + 1;
}

function stopPlayer() {
  clearInterval(player.timer);
  player.timer = null;
  const c = S.clip;
  if (c) {
    const bbFrames = S.shield.on && S.shield.bb ? S.shield.bb : c.bb;
    const bbEnd = S.shield.on && S.shield.bb ? S.shield.bb_end : c.bb_end;
    if (bbFrames.length) { $("bbEnd").textContent = endLabel(bbEnd); $("bbEnd").hidden = false; }
    if (c.clone.length) { $("cloneEnd").textContent = endLabel(c.clone_end); $("cloneEnd").hidden = false; }
    $("clonePanel").classList.remove("disagree");
  }
  if (S.pendingClip) { S.clip = S.pendingClip; S.pendingClip = null; setTimeout(startPlayer, 600); }
}

// ---------------------------------------------------------------- buttons
function hasFlaw() { return !!S.config.hasFlaw; }
function updateButtons() {
  const has = !!S.runId;
  $("btnReveal").disabled = !(has && S.finished && hasFlaw() && !S.revealed);
  $("btnReveal").hidden = has && S.finished && !hasFlaw();
  $("btnFix").disabled = !(has && S.revealed && !S.fix);
  $("chkShield").disabled = !(has && S.fix);
  renderBadge();
}

async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || `${r.status} ${r.statusText}`);
  return j;
}

function softReset() {
  clearInterval(player.timer); player.timer = null;
  resetRun();
  $("banner").hidden = true;
  $("ledger").innerHTML = '<div class="muted">No rules yet.</div>';
  $("subqLine").textContent = "SubQ has not read the evidence yet.";
  $("subqBars").innerHTML = "";
  $("code").innerHTML = '<span class="muted">The recovered program appears here.</span>';
  $("params").innerHTML = "";
  $("codeTitle").textContent = "Our copy";
  $("bbTitle").textContent = "The robot (real network)";
  $("bbPanel").classList.remove("guarded"); $("clonePanel").classList.remove("disagree");
  $("bbImg").removeAttribute("src"); $("cloneImg").removeAttribute("src");
  $("bbEmpty").hidden = false; $("cloneEmpty").hidden = false;
  $("bbEmpty").textContent = S.runId ? "Loading the robot's run…" : "Press Run to watch the robot";
  $("bbEnd").hidden = true; $("cloneEnd").hidden = true;
  $("bbMeta").textContent = ""; $("cloneMeta").textContent = ""; $("frameNote").textContent = "";
  $("btnReplayClip").disabled = true;
  $("chkShield").checked = false;
  $("diff").hidden = true; $("fixExplain").hidden = true; $("guardCheck").hidden = true; $("guardCost").hidden = true;
  $("reportBody").innerHTML = '<div class="muted">Appears when a run finishes.</div>';
  renderLearning(); renderFinding();
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
  ws.onopen = () => { softReset(); updateButtons(); };   // the server sends the whole buffer first
  ws.onclose = () => { if (S.runId === runId && !S.finished) setTimeout(() => { if (S.runId === runId) connect(runId); }, 1000); };
}

async function onRun() {
  const bb = $("selPolicy").value;
  const twist = $("selFlaw").value || null;
  try {
    $("btnRun").disabled = true;
    setStatus("Starting run…");
    const r = await post("/api/run", { bb, twist_id: twist, brain: "muse", live: true });
    attach(r.run_id, "live", { bb, twist_id: twist, hasFlaw: !!twist });
    setStatus("Watching the robot run…");
  } catch (e) { setStatus(`Run failed: ${e.message}`, true); }
  finally { $("btnRun").disabled = false; }
}

function parseCacheKey(key) {
  const [bb, tw] = String(key).split("__");
  return { bb, twist_id: tw && tw !== "none" ? tw : null };
}

async function onBackup(key) {
  $("backup").open = false;
  const cfg = parseCacheKey(key);
  try {
    const r = await post("/api/replay", { cache_key: key, speed: 1, max_seconds: 30 });
    attach(r.run_id, "replay", { ...cfg, hasFlaw: !!cfg.twist_id });
    showBanner("Backup: showing our earlier run of this exact configuration.");
  } catch (e) { setStatus(`Backup replay failed: ${e.message}`, true); }
}

async function onRevealClick() {
  try { $("btnReveal").disabled = true; await post("/api/reveal", { run_id: S.runId }); }
  catch (e) { setStatus(`Reveal failed: ${e.message}`, true); updateButtons(); }
}
async function onFixClick() {
  $("btnFix").disabled = true;
  setStatus("Muse is making the smallest change to our copy that removes the reported behavior…");
  try { await post("/api/fix", { run_id: S.runId }); } catch (e) { setStatus(`Fix failed: ${e.message}`, true); updateButtons(); }
}
async function onShieldToggle(ev) {
  const on = ev.target.checked;
  ev.target.checked = !on; // the shield event sets the real state
  if (on) setStatus("Guard: running the same start state with the fixed copy as a guard…");
  try { await post("/api/shield", { run_id: S.runId, on }); } catch (e) { setStatus(`Guard failed: ${e.message}`, true); }
}

// ---------------------------------------------------------------- results tab
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
const twistName = (t) => (!t || t === "none" ? "no planted flaw" : `planted flaw ${flawNum(t)} (${t})`);
const fmtR = (v) => (v == null ? "—" : (+v).toFixed(0));

function htable(head, rows, cls = () => "") {
  return `<table><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr>${rows.map((r, i) => `<tr class="${cls(i)}">${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</table>`;
}

async function loadResults() {
  const box = $("results");
  box.innerHTML = '<div class="muted">Loading…</div>';
  let j;
  try { j = await (await fetch("/api/results")).json(); } catch (e) { box.innerHTML = `<div class="muted">Could not load results: ${esc(e.message)}</div>`; return; }
  if (!j.available) { box.innerHTML = '<div class="muted">No batch results yet (python -m scripts.batch --snap 1,0).</div>'; return; }
  const R = j.results;
  const rows = (R.summary && R.summary.rows) || [];
  const flawRows = rows.filter((r) => r.twist && r.twist !== "none");
  const on = rows.filter((r) => r.snap === true), off = rows.filter((r) => r.snap === false);
  const flawOn = flawRows.filter((r) => r.snap === true), flawOff = flawRows.filter((r) => r.snap === false);

  // SubQ pairs
  const pairs = {};
  flawRows.forEach((r) => { (pairs[`${r.bb}|${r.twist}`] = pairs[`${r.bb}|${r.twist}`] || {})[r.snap ? "on" : "off"] = r; });
  const pairList = Object.entries(pairs).filter(([, p]) => p.on && p.off).map(([k, p]) => ({ k, ...p }));
  const both = pairList.filter((p) => p.on.queries_to_twist != null && p.off.queries_to_twist != null);
  const totOn = both.reduce((s, p) => s + p.on.queries_to_twist, 0), totOff = both.reduce((s, p) => s + p.off.queries_to_twist, 0);
  const ratios = both.map((p) => p.off.queries_to_twist / Math.max(1, p.on.queries_to_twist));
  const maxRatio = ratios.length ? Math.max(...ratios) : null;
  const cut = totOff ? 1 - totOn / totOff : null;

  // cross-match
  const cm = R.crossmatch || {};
  const diag = [], offd = [];
  Object.entries(cm).forEach(([prog, cols]) => {
    const own = prog.replace(/ program$/, "");
    Object.entries(cols || {}).forEach(([pol, v]) => (pol === own ? diag : offd).push(v));
  });
  const base = (R.baselines && R.baselines.rows) || [];

  const keys = [];
  keys.push([`${flawOn.filter((r) => r.twist_found).length} of ${flawOn.length}`, `planted flaws found with SubQ (${flawOff.filter((r) => r.twist_found).length} of ${flawOff.length} without)`]);
  keys.push([fmtPct(median(on.map((r) => r.agreement)), 0), "median match of the readable copies on new situations (SubQ on)"]);
  if (cut != null) keys.push([`${Math.round(100 * cut)}% fewer`, `queries to find the flaw with SubQ (${fmtInt(totOn)} vs ${fmtInt(totOff)}); up to ${maxRatio.toFixed(0)}× on a single case`]);
  keys.push([`${median(rows.map((r) => r.speedup)).toFixed(0)}×`, "faster: the copies run this much faster than the networks (median)"]);
  if (diag.length && offd.length) keys.push([`${fmtPct(mean(diag), 0)} vs ${fmtPct(mean(offd), 0)}`, "each copy matches its own robot better than its sibling trained differently"]);
  if (base.length) keys.push([fmtPct(median(base.map((r) => r.agreement)), 0), "match when Muse writes a controller alone, without asking the robot (median)"]);

  let h = `<section class="sec"><h2>Key results</h2><div class="keys">${keys.map(([v, c]) => `<div class="key"><div class="kv">${esc(v)}</div><div class="kc">${esc(c)}</div></div>`).join("")}</div>
    <div class="muted small">Computed from results/*.json: ${rows.length} batch runs over ${new Set(rows.map((r) => r.bb)).size} robots, each with and without SubQ.</div></section>`;

  // SubQ on vs off
  const subqRows = pairList.map((p) => [esc(pretty(p.on.label)), esc(twistName(p.on.twist)),
    p.on.queries_to_twist != null ? fmtInt(p.on.queries_to_twist) : "not found", p.off.queries_to_twist != null ? fmtInt(p.off.queries_to_twist) : "not found",
    p.on.turns_to_twist ?? "—", p.off.turns_to_twist ?? "—"]);
  subqRows.push(["<b>Total</b>", `<span class="muted">${both.length} cases found both ways</span>`, `<b>${fmtInt(totOn)}</b>`, `<b>${fmtInt(totOff)}</b>`, "", ""]);
  h += `<section class="sec"><h2>SubQ on vs off: queries to find the flaw</h2><div class="cap">Same robot, same planted flaw, same brain; the only difference is whether SubQ Snap scans the raw evidence each turn.</div>
    ${htable(["Robot", "Flaw", "Queries, SubQ on", "Queries, SubQ off", "Turns, on", "Turns, off"], subqRows, (i) => (i === subqRows.length - 1 ? "total" : ""))}</section>`;

  // Summary
  const sumRows = rows.map((r) => [esc(pretty(r.label)), esc(twistName(r.twist)), r.snap ? "on" : "off", fmtPct(r.agreement), `${fmtR(r.clone_reward)} / ${fmtR(r.bb_reward)}`,
    r.lines_of_code ?? "—", fmtInt(r.queries), r.speedup != null ? `${(+r.speedup).toFixed(0)}×` : "—",
    !r.twist || r.twist === "none" ? "—" : r.twist_found ? "found" : "not found"]);
  h += `<section class="sec"><h2>Summary per run</h2><div class="cap">Every batch run. Match is on held-out situations; reward is copy / robot on the same 20 start states. A red row is an honest failure.</div>
    ${htable(["Robot", "Flaw", "SubQ", "Match", "Reward copy / robot", "Lines", "Queries", "Speed-up", "Flaw"], sumRows,
      (i) => (rows[i].twist && rows[i].twist !== "none" && rows[i].twist_found === false ? "fail" : ""))}</section>`;

  // Cross-match
  const progs = Object.keys(cm), pols = [...new Set(progs.flatMap((p) => Object.keys(cm[p] || {})))];
  const lab = (bb) => esc(pretty(policyOf(bb).label));
  h += `<section class="sec"><h2>Cross-match: copy of robot A tested on robot B</h2><div class="cap">A copy should match the robot it was learned from better than a sibling robot trained with a different algorithm.</div>
    ${htable(["Copy of ↓ / tested on →", ...pols.map(lab)], progs.map((p) => [lab(p.replace(/ program$/, "")), ...pols.map((q) => {
      const v = (cm[p] || {})[q]; const own = p.replace(/ program$/, "") === q;
      return v == null ? "—" : own ? `<b>${fmtPct(v)}</b>` : fmtPct(v);
    })]))}</section>`;

  // Baselines
  const ownMatch = (bb) => { const r = rows.find((x) => x.bb === bb && (!x.twist || x.twist === "none") && x.snap); return r ? r.agreement : null; };
  h += `<section class="sec"><h2>Zero-query baseline: Muse writing a controller without asking the robot</h2><div class="cap">Muse gets the task name and sensor list only. It can write a decent controller, but not the robot's controller.</div>
    ${htable(["Robot", "Match, Muse alone", "Match, Unbox copy", "Reward Muse alone / robot", "Lines"], base.map((b) => [esc(pretty(b.label)), fmtPct(b.agreement), fmtPct(ownMatch(b.bb)), `${fmtR(b.clone_reward)} / ${fmtR(b.bb_reward)}`, b.lines_of_code ?? "—"]))}</section>`;
  box.innerHTML = h;
}

// ---------------------------------------------------------------- boot
async function getJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); }

function renderPolicyDesc() {
  const p = policyOf($("selPolicy").value);
  $("policyDesc").textContent = `${robotName(p)}: ${ROBOTS[envFamily(p.env_id)] || "a trained robot controller"}`;
}

async function loadTwists() {
  renderPolicyDesc();
  const bb = $("selPolicy").value;
  const sel = $("selFlaw");
  try {
    const j = await getJSON(`/api/twists?bb=${encodeURIComponent(bb)}`);
    const tw = j.twists.filter((t) => !HIDE_FROM_MENU.has(t.twist_id));
    sel.innerHTML = `<option value="">No planted flaw</option>` + tw.map((t) =>
      `<option value="${esc(t.twist_id)}">Planted flaw ${flawNum(t.twist_id)} (secret until Reveal)</option>`).join("");
  } catch (e) { sel.innerHTML = '<option value="">No planted flaw</option>'; }
}

async function loadBackups() {
  try {
    const j = await getJSON("/api/sources");
    const items = (j.cache || []).map((k) => {
      const c = parseCacheKey(k);
      if (c.twist_id && HIDE_FROM_MENU.has(c.twist_id)) return null;
      return `<button class="linkbtn" data-key="${esc(k)}">${esc(robotName(policyOf(c.bb)))} · ${c.twist_id ? `planted flaw ${flawNum(c.twist_id)}` : "no planted flaw"}</button>`;
    }).filter(Boolean);
    $("backupList").innerHTML = items.join("") || '<div class="muted small">No cached runs.</div>';
    $("backupList").querySelectorAll("button").forEach((b) => (b.onclick = () => onBackup(b.dataset.key)));
  } catch (e) { $("backupList").innerHTML = '<div class="muted small">Could not list cached runs.</div>'; }
}

async function boot() {
  renderLearning();
  $("btnRun").onclick = onRun;
  $("btnReveal").onclick = onRevealClick;
  $("btnFix").onclick = onFixClick;
  $("chkShield").onchange = onShieldToggle;
  $("btnReplayClip").onclick = () => startPlayer();
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
    POLICIES = pol.policies || [];
    $("selPolicy").innerHTML = POLICIES.map((p) => `<option value="${esc(p.bb)}">${esc(robotName(p))}</option>`).join("");
    if (POLICIES.some((p) => p.bb === "bb-01")) $("selPolicy").value = "bb-01";
    if (pol.warning) setStatus(pol.warning, true);
    else if (cfg.use_mocks) setStatus("Mock mode (USE_MOCKS=1).");
  } catch (e) { setStatus(`Could not reach the app server: ${e.message}`, true); }
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
