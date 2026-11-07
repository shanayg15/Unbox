// Lane B, task B6: UI. Vanilla JS. Everything on screen is derived from the event stream
// (contracts/events.py), so a live run, a replay and a reload all render the same way.
"use strict";

const $ = (id) => document.getElementById(id);
const ICONS = { confirmed: "✓", refuted: "✗", inferred: "?", observed: "◎", probing: "🔍" };
const LEDGER_ROWS = 8;
const GLOW_MS = 3000;

const S = {}; // run state, rebuilt from events
window.__unbox = S; // handy in the console and for UI checks

function resetState() {
  Object.assign(S, {
    runId: S.runId || null, kind: S.kind || null, finished: false, cached: false, banner: null,
    budget: 5000, queries: 0, metrics: [], ledger: [], prevLedger: {}, highlight: new Set(),
    snap: null, program: null, prevCode: null, fix: null, revealed: null,
    clip: null, pendingClip: null, shield: { on: false, bb: null }, events: 0,
    lastStatus: "", errors: [],
  });
}
resetState();

// ---------------------------------------------------------------- formatting
const fmtInt = (n) => (n == null ? "—" : Math.round(n).toLocaleString("en-US"));
const fmtPct = (x, d = 1) => (x == null ? "—" : `${(100 * x).toFixed(d)}%`);
const fmtK = (n) => (n >= 1000 ? `${Math.round(n / 1000)}K` : `${n}`);
const fmtP = (p) => (p == null ? "—" : Number(p).toFixed(2));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------------------------------------------------------------- event dispatch
function apply(ev) {
  S.events += 1;
  const d = ev.data || {};
  switch (ev.type) {
    case "run_started":
      S.budget = d.budget || S.budget;
      setStatus(`Run started · ${d.policy_label || ""} · brain ${d.brain || ""}${d.model ? " (" + d.model + ")" : ""}`);
      break;
    case "status":
      if (d.banner) showBanner(d.text);
      setStatus(d.text);
      break;
    case "error":
      S.errors.push(d.message);
      setStatus(d.message, true);
      break;
    case "query_count":
      S.queries = d.n;
      renderStats();
      break;
    case "metric":
      S.metrics.push({ q: d.queries, a: d.agreement });
      if (d.queries > S.queries) S.queries = d.queries;
      renderStats();
      renderCurve();
      break;
    case "ledger":
      onLedger(d.entries || []);
      break;
    case "snap":
      S.snap = d;
      renderSnap();
      renderLedger();
      break;
    case "program":
      onProgram(d);
      break;
    case "frames":
      onFrames(d);
      break;
    case "reveal":
      onReveal(d);
      break;
    case "fix":
      onFix(d);
      break;
    case "shield":
      onShield(d);
      break;
    case "run_finished":
      S.finished = true;
      S.cached = !!d.cached;
      onFinished(d);
      break;
    default:
      break;
  }
  updateButtons();
}

// ---------------------------------------------------------------- status / banner / stats
function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text || "";
  el.title = text || "";
  el.classList.toggle("error", !!isError);
  S.lastStatus = text;
}
function showBanner(text) {
  S.banner = text;
  const b = $("banner");
  b.textContent = "⚠ " + text;
  b.title = text;
  b.hidden = false;
}
function renderStats() {
  $("statQueries").textContent = fmtInt(S.queries);
  const m = S.metrics;
  const last = m[m.length - 1];
  $("statMatch").textContent = last ? fmtPct(last.a) : "—";
  const delta = $("statDelta");
  if (m.length >= 2) {
    const dv = 100 * (last.a - m[m.length - 2].a);
    delta.textContent = `${dv >= 0 ? "▲" : "▼"} ${Math.abs(dv).toFixed(1)}`;
    delta.className = "delta " + (dv >= 0 ? "up" : "down");
  } else {
    delta.textContent = "";
    delta.className = "delta";
  }
}
function onFinished(d) {
  const ev = d.eval || {};
  const parts = [`Finished: match ${fmtPct(ev.open_loop_agreement)} on ${fmtInt(ev.n_holdout_obs)} held-out states`];
  if (ev.clone_mean_reward != null && ev.bb_mean_reward != null)
    parts.push(`reward ${ev.clone_mean_reward.toFixed(0)} vs ${ev.bb_mean_reward.toFixed(0)}`);
  if (ev.us_per_action_bb && ev.us_per_action_clone)
    parts.push(`${Math.round(ev.us_per_action_bb / ev.us_per_action_clone)}× faster`);
  if (d.total_queries != null) parts.push(`${fmtInt(d.total_queries)} queries`);
  if (d.turns != null) parts.push(`${d.turns} turns`);
  setStatus(parts.join(" · "));
  renderBadge();
}
function renderBadge() {
  const b = $("runBadge");
  if (!S.runId) { b.textContent = ""; return; }
  const tags = [];
  if (S.kind === "replay") tags.push("REPLAY");
  if (S.cached || S.banner) tags.push("CACHED");
  if (!S.finished) tags.push("● running");
  b.textContent = `${tags.join(" · ")}  ${S.runId}`;
  b.classList.toggle("cached", tags.includes("CACHED"));
}

// ---------------------------------------------------------------- ledger + SubQ
function onLedger(entries) {
  const prev = S.prevLedger;
  const next = {};
  entries.forEach((e) => (next[e.rule_id] = e));
  S.fresh = new Set(entries.filter((e) => {
    const p = prev[e.rule_id];
    return !p || p.status !== e.status || p.text !== e.text;
  }).map((e) => e.rule_id));
  S.prevLedger = next;
  S.ledger = entries;
  renderLedger();
  if (S.fresh.size) setTimeout(() => { S.fresh = new Set(); renderLedger(); }, 1800);
}

const ruleNum = (rid) => parseInt(String(rid).replace(/\D/g, ""), 10) || 0;

function renderLedger() {
  const box = $("ledgerRows");
  if (!S.ledger.length) { box.innerHTML = '<div class="placeholder">No rules yet.</div>'; return; }
  const rows = [...S.ledger].sort((a, b) => ruleNum(a.rule_id) - ruleNum(b.rule_id)).slice(-LEDGER_ROWS);
  const support = (S.snap && S.snap.rule_support) || {};
  box.innerHTML = rows.map((e) => {
    const cls = ["lrow", e.status];
    if (S.highlight.has(e.rule_id)) cls.push("highlight");
    if (S.fresh && S.fresh.has(e.rule_id)) cls.push("fresh");
    const p = support[e.rule_id];
    const chip = p == null ? "<span></span>"
      : `<span class="chip" title="SubQ Snap: advisory P(rule consistent with all rows). The ✓/✗ verdict comes from Python probes.">SubQ p=${fmtP(p)}</span>`;
    return `<div class="${cls.join(" ")}" data-rule="${esc(e.rule_id)}" title="${esc(e.text)} · evidence ${esc((e.evidence || []).join(", "))}">
      <span class="icon" title="${esc(e.status)} (Python verdict)">${ICONS[e.status] || "·"}</span>
      <span class="rid">${esc(e.rule_id)}</span>
      <span class="text">${esc(e.text)}</span>
      ${chip}
      <span class="conf">${Math.round(100 * (e.confidence || 0))}%</span>
    </div>`;
  }).join("");
}

function renderSnap() {
  const s = S.snap;
  if (!s) return;
  const an = s.anomaly_feature || {};
  const np = s.next_probe || {};
  const anomaly = an.choice && an.choice !== "none"
    ? `anomaly most likely in <b>${esc(an.choice)}</b> (p ${fmtP(an.confidence ?? (an.probabilities || {})[an.choice])})`
    : `no clear anomaly (p ${fmtP(an.confidence)})`;
  const tokens = s.input_tokens ? ` (${fmtK(s.input_tokens)} tokens)` : "";
  $("subqLine").innerHTML =
    `<span class="subq-tag">SubQ Snap</span> read ${fmtInt(s.n_rows)} raw rows${tokens} · ${anomaly}` +
    (np.choice ? ` · probe next: <b>${esc(np.choice)}</b>` : "") +
    (s.turn != null ? ` <span class="meta">· turn ${s.turn}</span>` : "");
  const rel = s.feature_relevance || {};
  const names = Object.keys(rel);
  const top = names.reduce((m, k) => (rel[k] > (rel[m] ?? -1) ? k : m), names[0]);
  const bars = $("subqBars");
  // keep the DOM nodes so widths animate between turns
  if (bars.dataset.names !== names.join(",")) {
    bars.dataset.names = names.join(",");
    bars.innerHTML = names.map((k) => `<div class="bar" data-f="${esc(k)}" title="P(action depends on ${esc(k)})">
      <span class="name">${esc(k)}</span><span class="track"><span class="fill" style="width:0"></span></span><span class="p"></span></div>`).join("");
  }
  bars.querySelectorAll(".bar").forEach((el) => {
    const k = el.dataset.f;
    const p = rel[k] ?? 0;
    el.classList.toggle("top", k === (an.choice && rel[an.choice] != null ? an.choice : top));
    el.querySelector(".fill").style.width = `${Math.max(2, 100 * p)}%`;
    el.querySelector(".p").textContent = fmtP(p);
  });
}

// ---------------------------------------------------------------- code
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

// LCS line diff: [{op: "eq"|"add"|"del", line}]
function lineDiff(a, b) {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const ops = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { ops.push({ op: "eq", line: b[j] }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) ops.push({ op: "del", line: a[i++] });
    else ops.push({ op: "add", line: b[j++] });
  }
  while (i < n) ops.push({ op: "del", line: a[i++] });
  while (j < m) ops.push({ op: "add", line: b[j++] });
  return ops;
}

function renderCode(lines, classes = {}, lineNos = null) {
  const pre = $("code");
  pre.innerHTML = lines.map((l, i) =>
    `<div class="cl ${classes[i] || ""}"><span class="ln">${lineNos ? lineNos[i] : i + 1}</span><span class="src">${highlightLine(l)}</span></div>`
  ).join("");
  const firstIdx = Object.keys(classes).map(Number).sort((x, y) => x - y)[0];
  if (firstIdx != null) {
    const el = pre.children[firstIdx];
    if (el) {
      const top = el.offsetTop - pre.offsetTop;
      if (top < pre.scrollTop || top > pre.scrollTop + pre.clientHeight - 40) pre.scrollTop = Math.max(0, top - 40);
    }
  }
}

function onProgram(d) {
  const lines = d.code.replace(/\s+$/, "").split("\n");
  let changed = Array.isArray(d.changed_lines) && d.changed_lines.length ? d.changed_lines : null;
  if (!changed && S.prevCode != null) {
    changed = [];
    let ln = 0;
    lineDiff(S.prevCode.split("\n"), lines).forEach((o) => { if (o.op !== "del") { ln++; if (o.op === "add") changed.push(ln); } });
  }
  S.prevCode = lines.join("\n");
  S.program = d;
  S.fix = null;
  const classes = {};
  (changed || []).forEach((n) => (classes[n - 1] = "glow"));
  $("codeTitle").textContent = `RECOVERED CODE · v${d.version}`;
  const params = Object.entries(d.params || {}).map(([k, v]) => `${k}=${(+v).toFixed(3)}`).join("  ");
  $("codeMeta").textContent = `match ${fmtPct(d.agreement)} · ${lines.length} lines${params ? " · " + params : ""}`;
  $("codeMeta").title = params;
  renderCode(lines, classes);
  clearTimeout(S.glowTimer);
  S.glowTimer = setTimeout(() => $("code").querySelectorAll(".cl.glow").forEach((el) => el.classList.remove("glow")), GLOW_MS);
}

function onFix(d) {
  S.fix = d;
  const before = (S.prevCode || "").split("\n");
  const after = d.code.replace(/\s+$/, "").split("\n");
  const ops = lineDiff(before, after);
  const lines = [], classes = {}, nos = [];
  let ln = 0;
  ops.forEach((o, i) => {
    lines.push(o.line);
    if (o.op === "del") { classes[i] = "del"; nos.push("−"); }
    else { ln++; nos.push(ln); if (o.op === "add") classes[i] = "add"; }
  });
  const nDel = ops.filter((o) => o.op === "del").length, nAdd = ops.filter((o) => o.op === "add").length;
  $("codeTitle").textContent = "FIXED CODE · minimal diff";
  $("codeMeta").textContent = `−${nDel} +${nAdd} lines · everything else identical`;
  renderCode(lines, classes, nos);
}

// ---------------------------------------------------------------- curve
function renderCurve() {
  const svg = $("curve");
  const W = 420, H = 300, L = 52, R = 18, T = 18, B = 38;
  const pts = S.metrics;
  const maxQ = Math.max(S.budget || 0, ...pts.map((p) => p.q), 1);
  const x = (q) => L + (W - L - R) * (q / maxQ);
  const y = (a) => T + (H - T - B) * (1 - a);
  let g = "";
  [0, 0.25, 0.5, 0.75, 1].forEach((a) => {
    g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(a)}" y2="${y(a)}"/>`;
    g += `<text class="lbl" x="${L - 8}" y="${y(a) + 4}" text-anchor="end">${a * 100}%</text>`;
  });
  [0, 0.5, 1].forEach((f) => {
    const q = maxQ * f;
    g += `<text class="lbl" x="${x(q)}" y="${H - B + 20}" text-anchor="${f === 0 ? "start" : f === 1 ? "end" : "middle"}">${q >= 1000 ? (q / 1000).toFixed(1) + "k" : Math.round(q)}</text>`;
  });
  g += `<line class="axis" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/><line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>`;
  g += `<text class="lbl" x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">queries (budget ${fmtInt(maxQ)})</text>`;
  if (pts.length) {
    const poly = pts.map((p) => `${x(p.q).toFixed(1)},${y(p.a).toFixed(1)}`).join(" ");
    g += `<polyline class="line" points="${poly}"/>`;
    pts.forEach((p, i) => (g += `<circle class="pt ${i === pts.length - 1 ? "last" : ""}" cx="${x(p.q)}" cy="${y(p.a)}" r="5"/>`));
    const last = pts[pts.length - 1];
    const lx = Math.min(x(last.q), W - R - 4);
    g += `<text class="val" x="${lx}" y="${Math.max(T + 14, y(last.a) - 12)}" text-anchor="${lx > W - 90 ? "end" : "middle"}">${fmtPct(last.a)}</text>`;
  }
  svg.innerHTML = g;
  $("curveMeta").textContent = pts.length ? `${pts.length} points` : "";
}

// ---------------------------------------------------------------- clips (one timer drives both)
const player = { timer: null, i: 0, fps: 10 };

function onFrames(d) {
  const clip = { bb: d.bb || [], clone: d.clone || [], disagree: d.disagree || [], fps: d.fps || 10, version: d.version, seed: d.seed };
  if (!S.clip) { S.clip = clip; startPlayer(true); }
  else S.pendingClip = clip; // swapped in at the next loop start, never mid-clip
}

function onShield(d) {
  S.shield = { on: !!d.on, bb: d.on ? d.bb : null };
  $("chkShield").checked = S.shield.on;
  $("bbPanel").classList.toggle("shielded", S.shield.on);
  $("bbTitle").textContent = S.shield.on ? "BLACK BOX + SHIELD (fixed rule)" : "BLACK BOX · the robot's net";
  setStatus(S.shield.on ? "Shield on: the robot's own output, corrected only where the fixed rule disagrees"
                        : "Shield off: the robot's raw output");
  if (S.clip) startPlayer(true); // user action: restart both clips together, still in lockstep
}

function startPlayer(restart) {
  if (!S.clip) return;
  if (restart) player.i = 0;
  player.fps = S.clip.fps || 10;
  clearInterval(player.timer);
  player.timer = setInterval(tick, 1000 / player.fps);
  $("bbEmpty").hidden = true;
  $("cloneEmpty").hidden = true;
  tick();
}

function tick() {
  const c = S.clip;
  if (!c) return;
  const bbFrames = S.shield.on && S.shield.bb ? S.shield.bb : c.bb;
  const n = Math.max(bbFrames.length, c.clone.length);
  if (!n) return;
  if (player.i >= n) {
    player.i = 0;
    if (S.pendingClip) { S.clip = S.pendingClip; S.pendingClip = null; startPlayer(true); return; }
  }
  const i = player.i;
  const bi = Math.min(i, bbFrames.length - 1), ci = Math.min(i, c.clone.length - 1);
  if (bi >= 0) $("bbImg").src = "data:image/jpeg;base64," + bbFrames[bi];
  if (ci >= 0) $("cloneImg").src = "data:image/jpeg;base64," + c.clone[ci];
  $("clonePanel").classList.toggle("disagree", !S.shield.on && !!c.disagree[Math.min(i, c.disagree.length - 1)]);
  $("frameNote").textContent = `frame ${i + 1}/${n}`;
  const red = c.disagree.filter(Boolean).length;
  $("cloneMeta").textContent = `v${c.version} · ${red}/${c.disagree.length} frames disagree`;
  $("cloneTitle").textContent = `CLONE · running recovered code v${c.version}`;
  $("bbMeta").textContent = S.shield.on ? "robot output, corrected only where the fixed rule differs" : `seed ${c.seed} · ${c.fps} fps`;
  $("seedNote").textContent = `same seed (${c.seed}) · synced`;
  player.i = i + 1;
}

// ---------------------------------------------------------------- reveal
function onReveal(d) {
  S.revealed = d;
  S.highlight = new Set(d.highlight_rule_ids || []);
  const sel = $("selFlaw");
  const opt = sel.options[sel.selectedIndex];
  if (opt) opt.textContent = d.label;
  else sel.innerHTML = `<option>${esc(d.label)}</option>`;
  sel.classList.add("revealed");
  sel.title = `feature: ${d.feature}`;
  renderLedger();
  const ids = [...S.highlight];
  setStatus(`Revealed: “${d.label}” (feature ${d.feature})` + (ids.length ? ` · ledger rule ${ids.join(", ")} already describes it` : ""));
}

// ---------------------------------------------------------------- buttons
function updateButtons() {
  const has = !!S.runId;
  $("btnReveal").disabled = !(has && S.program && !S.revealed);
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

function clearScreen() {
  clearInterval(player.timer);
  resetState();
  $("banner").hidden = true;
  $("ledgerRows").innerHTML = '<div class="placeholder">No rules yet.</div>';
  $("subqLine").innerHTML = '<span class="subq-tag">SubQ Snap</span> waiting for the first evidence scan';
  $("subqBars").innerHTML = ""; $("subqBars").dataset.names = "";
  $("code").innerHTML = '<span class="placeholder">The recovered program appears here.</span>';
  $("codeTitle").textContent = "RECOVERED CODE"; $("codeMeta").textContent = "";
  $("cloneTitle").textContent = "CLONE · running recovered code";
  $("bbTitle").textContent = "BLACK BOX · the robot's net";
  $("bbPanel").classList.remove("shielded"); $("clonePanel").classList.remove("disagree");
  $("bbImg").removeAttribute("src"); $("cloneImg").removeAttribute("src");
  $("bbEmpty").hidden = false; $("cloneEmpty").hidden = false;
  $("bbMeta").textContent = "waiting for a clip"; $("cloneMeta").textContent = ""; $("frameNote").textContent = "";
  $("chkShield").checked = false;
  const sel = $("selFlaw");
  sel.classList.remove("revealed");
  [...sel.options].forEach((o) => { if (o.dataset.label) o.textContent = o.dataset.label; });
  renderStats(); renderCurve();
}

function attach(runId, kind) {
  if (S.ws) { S.ws.onclose = null; S.ws.close(); }
  S.runId = runId; S.kind = kind;
  clearScreen();
  S.runId = runId; S.kind = kind;
  const params = new URLSearchParams(location.search);
  params.set("run", runId); params.delete("replay");
  history.replaceState(null, "", `?${params}`);
  connect(runId);
}

function connect(runId) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/${runId}`);
  S.ws = ws;
  let first = true;
  ws.onmessage = (m) => {
    if (first) { first = false; }
    try { apply(JSON.parse(m.data)); } catch (e) { console.error(e); }
  };
  ws.onopen = () => {
    // the server sends the whole buffer first, so rebuild from scratch
    const rid = S.runId, kind = S.kind;
    clearScreen(); S.runId = rid; S.kind = kind;
    updateButtons();
  };
  ws.onclose = () => {
    if (S.runId === runId && !S.finished) setTimeout(() => { if (S.runId === runId) connect(runId); }, 1000);
  };
}

async function onRun() {
  const bb = $("selPolicy").value;
  const twist = $("selFlaw").value || null;
  try {
    setStatus("Starting run…");
    const r = await post("/api/run", { bb, twist_id: twist, brain: $("selBrain").value, live: true });
    attach(r.run_id, "live");
  } catch (e) { setStatus(`Run failed: ${e.message}`, true); }
}

async function onReplay(sourceValue, speed, hold = true) {
  const v = sourceValue || $("selSource").value || "fixture:demo_events";
  const [kind, ...rest] = v.split(":");
  const name = rest.join(":");
  const body = { speed: speed ?? parseFloat($("selSpeed").value), hold_reveal: hold };
  if (kind === "run") body.run_id = name; else if (kind === "cache") body.cache_key = name; else body.fixture = name;
  try {
    const r = await post("/api/replay", body);
    attach(r.run_id, "replay");
  } catch (e) { setStatus(`Replay failed: ${e.message}`, true); }
}

async function onRevealClick() {
  try { await post("/api/reveal", { run_id: S.runId }); } catch (e) { setStatus(`Reveal failed: ${e.message}`, true); }
}
async function onFixClick() {
  $("btnFix").disabled = true;
  setStatus("Asking the brain for the smallest change that removes the reported behavior…");
  try { await post("/api/fix", { run_id: S.runId }); } catch (e) { setStatus(`Fix failed: ${e.message}`, true); updateButtons(); }
}
async function onShieldToggle(ev) {
  const on = ev.target.checked;
  ev.target.checked = !on; // the shield event sets the real state
  if (on) setStatus("Shield: running the demo seed with the fixed rule as a guard…");
  try { await post("/api/shield", { run_id: S.runId, on }); } catch (e) { setStatus(`Shield failed: ${e.message}`, true); }
}

// ---------------------------------------------------------------- results tab
function showTab(tab) {
  const results = tab === "results";
  $("tabLive").classList.toggle("active", !results);
  $("tabResults").classList.toggle("active", results);
  document.querySelectorAll(".view-live").forEach((el) => (el.hidden = results));
  $("results").hidden = !results;
  $("right").classList.toggle("show-results", results);
  if (results) loadResults();
}

const asRows = (x) => (Array.isArray(x) ? x : x && Array.isArray(x.rows) ? x.rows : null);
function cell(k, v) {
  if (v == null) return `<td class="num">—</td>`;
  if (typeof v === "boolean") return `<td>${v ? "yes" : "no"}</td>`;
  if (typeof v === "number") {
    const pct = /agreement|bound|rate/.test(k) && v <= 1;
    return `<td class="num">${pct ? fmtPct(v) : Number.isInteger(v) ? fmtInt(v) : v.toFixed(2)}</td>`;
  }
  return `<td>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</td>`;
}
const HEAD = { clone_reward: "clone R", bb_reward: "robot R", bound_95: "fail ≤95%", lines_of_code: "LOC",
  twist_found: "found", agreement: "match", queries_to_twist: "q→flaw", turns_to_twist: "turns→flaw", seconds: "sec" };
function table(rows, cols, rowClass = () => "") {
  if (!rows || !rows.length) return '<div class="empty-note">No rows.</div>';
  cols = cols || [...new Set(rows.flatMap((r) => Object.keys(r)))];
  return `<table><tr>${cols.map((c) => `<th title="${esc(c)}">${esc(HEAD[c] || c)}</th>`).join("")}</tr>${rows.map((r) =>
    `<tr class="${rowClass(r)}">${cols.map((c) => cell(c, r[c])).join("")}</tr>`).join("")}</table>`;
}
function matrix(obj) {
  const rows = Object.keys(obj), cols = [...new Set(rows.flatMap((r) => Object.keys(obj[r] || {})))];
  return `<table><tr><th>program ↓ / policy →</th>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${rows.map((r) =>
    `<tr><td>${esc(r)}</td>${cols.map((c) => cell("agreement", obj[r][c])).join("")}</tr>`).join("")}</table>`;
}

async function loadResults() {
  const box = $("results");
  box.innerHTML = '<div class="empty-note">Loading…</div>';
  let j;
  const src = new URLSearchParams(location.search).get("results") === "fixture" ? "?source=fixture" : "";
  try { j = await (await fetch("/api/results" + src)).json(); } catch (e) { box.innerHTML = `<div class="empty-note">Could not load results: ${esc(e.message)}</div>`; return; }
  if (!j.available) {
    box.innerHTML = '<div class="empty-note">No batch results yet. They appear here once <span class="mono">python -m scripts.batch --snap 1,0</span> (task A6) writes <span class="mono">results/*.json</span>.</div>';
    return;
  }
  const R = j.results;
  let html = j.source === "fixture"
    ? '<div class="banner">FIXTURE DATA · fake numbers for building this tab; real numbers come from scripts/batch.py</div>' : "";
  const summary = asRows(R.summary);
  if (summary) {
    // with vs without SubQ: queries and turns to find the twist
    const groups = {};
    summary.forEach((r) => {
      if (r.snap == null) return;
      const k = `${r.label || r.bb} · ${r.twist || "no flaw"} · ${r.brain || ""}`;
      (groups[k] = groups[k] || {})[r.snap ? "on" : "off"] = r;
    });
    const cmp = Object.entries(groups).filter(([, g]) => g.on && g.off).map(([k, g]) => ({
      config: k,
      "queries (SubQ)": g.on.queries_to_twist, "queries (no SubQ)": g.off.queries_to_twist,
      "turns (SubQ)": g.on.turns_to_twist, "turns (no SubQ)": g.off.turns_to_twist,
    }));
    if (cmp.length) {
      html += "<h3>WITH SUBQ vs WITHOUT SUBQ · to find the hidden flaw</h3>" + table(cmp);
    }
    const cols = ["label", "twist", "snap", "agreement", "clone_reward", "bb_reward", "bound_95",
      "lines_of_code", "queries", "turns", "speedup", "twist_found"].filter((c) => summary.some((r) => c in r));
    html += '<h3>SUMMARY <span class="meta">· <span style="color:var(--red)">✗ red row</span> = honest failure (flaw not found)</span></h3>' + table(summary, cols, (r) => (r.twist && r.twist !== "none" && r.twist_found === false ? "fail" : ""));
  }
  if (R.crossmatch) {
    const rows = asRows(R.crossmatch);
    html += "<h3>CROSS-MATCH · recovered program vs other policies of the same env</h3>" + (rows ? table(rows) : matrix(R.crossmatch));
  }
  if (R.baselines) html += "<h3>ZERO-QUERY BASELINE</h3>" + (asRows(R.baselines) ? table(asRows(R.baselines)) : table([R.baselines]));
  Object.entries(R).forEach(([k, v]) => {
    if (["summary", "crossmatch", "baselines"].includes(k) || k.startsWith("_")) return;
    html += `<h3>${esc(k.toUpperCase())}</h3>` + (asRows(v) ? table(asRows(v)) : table([v]));
  });
  box.innerHTML = html || '<div class="empty-note">Results files are empty.</div>';
}

// ---------------------------------------------------------------- boot
async function getJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); }

async function loadTwists() {
  const bb = $("selPolicy").value;
  const sel = $("selFlaw");
  sel.classList.remove("revealed");
  try {
    const j = await getJSON(`/api/twists?bb=${encodeURIComponent(bb)}`);
    sel.innerHTML = `<option value="" data-label="No flaw">No flaw</option>` + j.twists.map((t) =>
      `<option value="${esc(t.twist_id)}" data-label="${esc(t.hidden_label)}">${esc(t.hidden_label)}</option>`).join("");
    if (j.twists.length) sel.selectedIndex = 1;
  } catch (e) { sel.innerHTML = '<option value="">(unavailable)</option>'; }
}

async function loadSources() {
  try {
    const j = await getJSON("/api/sources");
    const opts = [
      ...j.fixtures.map((f) => [`fixture:${f}`, `fixture · ${f}`]),
      ...j.cache.map((c) => [`cache:${c}`, `cache · ${c}`]),
      ...j.runs.map((r) => [`run:${r}`, `run · ${r}`]),
    ];
    $("selSource").innerHTML = opts.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("") ||
      '<option value="fixture:demo_events">fixture · demo_events</option>';
  } catch (e) { $("selSource").innerHTML = '<option value="fixture:demo_events">fixture · demo_events</option>'; }
}

async function boot() {
  renderStats(); renderCurve();
  $("btnRun").onclick = onRun;
  $("btnReplay").onclick = () => onReplay();
  $("btnReveal").onclick = onRevealClick;
  $("btnFix").onclick = onFixClick;
  $("chkShield").onchange = onShieldToggle;
  $("tabLive").onclick = () => showTab("live");
  $("tabResults").onclick = () => showTab("results");
  $("selPolicy").onchange = loadTwists;

  try {
    const [cfg, pol] = await Promise.all([getJSON("/api/config"), getJSON("/api/policies")]);
    $("selBrain").innerHTML = cfg.brains.map((b) => `<option value="${esc(b.id)}">${esc(b.label)}</option>`).join("");
    $("selPolicy").innerHTML = pol.policies.map((p) => `<option value="${esc(p.bb)}">${esc(p.label)}</option>`).join("");
    if (pol.warning) setStatus(pol.warning, true);
    else if (cfg.use_mocks) setStatus("Mock mode (USE_MOCKS=1) · Run, or Replay the fixture.");
  } catch (e) { setStatus(`Could not reach the app server: ${e.message}`, true); }
  await Promise.all([loadTwists(), loadSources()]);

  const q = new URLSearchParams(location.search);
  if (q.get("tab") === "results") showTab("results");
  if (q.get("speed") && [...$("selSpeed").options].some((o) => o.value === q.get("speed"))) $("selSpeed").value = q.get("speed");
  if (q.get("replay")) {
    const src = q.get("replay");
    onReplay(src.includes(":") ? src : `fixture:${src}`, parseFloat(q.get("speed") || "1"), q.get("hold") !== "0");
  } else if (q.get("run")) {
    S.runId = q.get("run"); S.kind = "attached"; connect(S.runId);
  }
}

boot();
