# LANES.md: four parallel build sessions

Goal: **everything built by 2:30**, then two clear hours for demo quality. Put this file in the repo root next to `CLAUDE.md`.

Each person runs **two Claude Code sessions at once** in the same repo folder, each owning different folders. Four streams in parallel:

| Person | Session | Owns (only edits these) | Tasks |
|---|---|---|---|
| A | **A-World** | `blackbox/`, `tests/test_a_world*.py` | A1 → A2 → A3, then demo-seed finder |
| A | **A-Proof** | `numerics/`, `scripts/batch.py`, `tests/test_a_proof*.py` | A4 → A5, then A6 after merge |
| B | **B-Brain** | `agent/`, `store/`, `contracts/mocks.py`, `scripts/smoke.py`, `tests/test_b_brain*.py` | B1 → B2 → B3 → B4 |
| B | **B-Show** | `app/`, `fixtures/`, `scripts/make_fixture.py`, `README.md`, `tests/test_b_show*.py` | fixture → B5 → B6, then B7 after merge |

Nobody edits `contracts/types.py`, `contracts/numerics_api.py` or `contracts/events.py` without telling the other person.

---

## 1. Git rule for two sessions in one folder

Two Claude sessions in one folder will collide on git. So: **Claude sessions never run git. The human syncs.**

`sync.sh` (repo root, `chmod +x sync.sh`):
```bash
#!/usr/bin/env bash
set -e
git add -A
git commit -m "${1:-sync}" || true
git pull --rebase origin main
pytest -q tests/test_isolation.py || echo "WARNING: isolation tests failing"
git push origin main
```
Run `./sync.sh "what changed"` every ~30 min and at every checkpoint. Never `git add .env`; the `.gitignore` handles it.

---

## 2. Internal contracts (between sessions, not in contracts/)

These let B-Brain and B-Show build apart, and A-World and A-Proof build apart.

**Event files.** One event envelope (see `contracts/events.py`) per line, JSON. Runs live at `runs/<run_id>/events.jsonl`; UI fixtures at `fixtures/*.jsonl`. B-Show reads these files directly, so it never waits on the store.

**Agent entry point (B-Brain implements, B-Show calls):**
```python
# agent/loop.py
def run(bb, brain: str, *, env_factory, numerics, emit, store, run_id: str,
        budget: int, max_turns: int, live: bool, timeout_s: float | None,
        policy_label: str, demo_seed: int = 0) -> dict:
    """bb: contracts.types.BlackBox. env_factory(seed) -> (env, first_obs).
    numerics: module with the contracts/numerics_api.py functions (real or stub).
    emit(event_dict) is called from this thread; must be thread-safe on the caller side.
    Returns {"program": Program, "ledger": [LedgerEntry], "eval": EvalReport, "events_path": str}."""

# agent/fixer.py
def fix(program: dict, ledger: list[dict], reveal_label: str, brain: str, run_id: str) -> dict:
    """FIX prompt → {"code", "params", "diff", "changed_rule_ids", "explanation"}."""
```
The agent never builds envs itself. The app (B-Show) builds `env_factory` from the env id it gets via admin. The `agent.loop` CLI takes `--env` from the human.

**Black box for numerics tests (A-Proof writes, in its own tests):** `RuleBlackBox(fn, params, spec, twist=None)` implementing `BlackBox` in memory. A-Proof never waits on A-World.

---

## 3. Session prompts (paste one per Claude Code window)

**A-World**
```
Read CLAUDE.md and LANES.md. You are session A-World. Only edit blackbox/ and tests/test_a_world*.py. Never run git commands; the human syncs. Build CLAUDE.md tasks A1, A2, A3 in order, following the A-World notes in LANES.md section 4. After each task, run its "Done when" check and show me the output, then continue to the next task without waiting for me unless you're blocked. When A3 is done, build the demo-seed finder from LANES.md.
```

**A-Proof**
```
Read CLAUDE.md and LANES.md. You are session A-Proof. Only edit numerics/, scripts/batch.py and tests/test_a_proof*.py. Never run git commands; the human syncs. Build CLAUDE.md tasks A4 then A5, following the A-Proof notes in LANES.md section 4. Test everything against an in-memory RuleBlackBox you write in the tests; do not wait for the real black-box server. After each task, run its "Done when" check and show me the output, then continue without waiting unless blocked. Stop after A5 and tell me; A6 starts after the merge.
```

**B-Brain**
```
Read CLAUDE.md and LANES.md. You are session B-Brain. Only edit agent/, store/, contracts/mocks.py, scripts/smoke.py and tests/test_b_brain*.py. Never run git commands; the human syncs. Build CLAUDE.md tasks B1, B2, B3, B4 in order with USE_MOCKS=1, following the B-Brain notes in LANES.md section 4. Implement agent.loop.run and agent.fixer.fix with the exact signatures in LANES.md section 2. After each task, run its "Done when" check and show me the output, then continue without waiting unless blocked.
```

**B-Show**
```
Read CLAUDE.md and LANES.md. You are session B-Show. Only edit app/, fixtures/, scripts/make_fixture.py, README.md and tests/test_b_show*.py. Never run git commands; the human syncs. First build scripts/make_fixture.py (LANES.md section 4), then CLAUDE.md tasks B5 and B6, building and testing the UI entirely by replaying fixtures. agent.loop.run may still be a stub; call it with the signature in LANES.md section 2 and handle NotImplementedError gracefully. Follow the B-Show notes in LANES.md section 4. After each task, run its "Done when" check, then continue without waiting unless blocked. Stop after B6 and tell me.
```

---

## 4. Session notes (the traps, decided in advance)

### Mac notes (everyone)
- Use Python **3.11 or 3.12** for the venv (`python3.12 -m venv .venv`). 3.13 can lack wheels for Box2D.
- If Box2D fails to build: `xcode-select --install` (once), `brew install swig`, then `pip install -r requirements.txt` again.
- Load every SB3 model with `device="cpu"`. Tiny policy nets are faster on CPU, and Apple's GPU backend (MPS) chokes on float64.
- `render_mode="rgb_array"` needs no window or display; it works headless on macOS.
- During the demo, keep the Mac awake: `caffeinate -d` in a spare terminal.

### A-World
- **HF loading:** `load_from_hub(repo_id="sb3/ppo-CartPole-v1", filename="ppo-CartPole-v1.zip")`. If a repo also has `vec_normalize.pkl`, load it and normalize obs **inside `/act` and `/episode`** with the saved stats. Without that, the policy acts nonsensically.
- **Sanity check every policy:** mean reward over 5 seeds must be near the score on its model card. If it's far off, you're missing VecNormalize or using the wrong env version.
- **Env versions:** use `LunarLander-v3` (gymnasium ≥ 1.0) for the v2 checkpoints; same obs and actions.
- **Thread safety:** FastAPI runs sync endpoints in a threadpool. Wrap each model's `predict` in a `threading.Lock`.
- **Episodes:** `gym.make(env_id, render_mode="rgb_array" if render else None)`, `env.reset(seed=seed)`, step until terminated/truncated. Cache key `(bb, twist_id, seed, render)`. Frames: every 3rd step, 360 px wide, JPEG quality 60 (keeps WebSocket payloads small).
- **Twist report:** `python -m blackbox.twists --report` runs seeds 0–9 per twist, prints fire rate (% of steps where it changed the action) and mean reward with vs without the twist. Target 5–30% fire rate **and** a visible reward drop.
- **Demo-seed finder** (after A3): `python -m blackbox.twists --demo-seed bb-05 t-ll-1` scans seeds 0–200 and prints the 5 best: clean policy lands (reward > 200), twisted policy fails, twist fires in the first half of the episode. This is the seed the whole demo uses.

### A-Proof
- **Sandbox = AST whitelist, not blacklist.** Allowed node types: `Module, FunctionDef, arguments, arg, Return, Assign, AugAssign, If, IfExp, Expr` (docstring only), `Compare, BoolOp, BinOp, UnaryOp, Call, Name, Load, Store, Constant, Subscript, Slice, Tuple, List, keyword`, all operator nodes. `Attribute` only when the value is `Name` `math` or `np` and the attr doesn't start with `_`. `Call` only on `abs, min, max, float, int, round, len, bool` or `math.*`/`np.*`. Params are read as `P["name"]` only.
- **Return coercion:** cast numpy ints/bools to `int`; check `0 <= a < n_actions`.
- **Fit speed:** keep X as plain Python lists, not numpy rows (indexing `obs[i]` on a list is much faster in a Python loop). Subsample to ≤ 3,000 points.
- **CMA-ES loop:** manual, so the time cap is exact:
  ```python
  es = cma.CMAEvolutionStrategy([0.0]*k, 0.5, {"popsize": 16, "verbose": -9, "seed": 0})
  while not es.stop() and time.time() - t0 < max_seconds:
      Z = es.ask(); es.tell(Z, [loss(z) for z in Z])
  ```
  Track the best seen, including the initial params, so fitting never makes things worse.
- **Prober and binary features:** features with ≤ 2 unique values in seed obs (LunarLander legs) are swept only over those values, never Gaussian-noised.
- **Prober ranges:** use empirical p1–p99 from seed obs, not spec bounds (some spec bounds are ±inf).
- **A5 "Done when" on RuleBlackBox:** a CartPole-style rule policy + a `cart_v > 0.5 → push_left` twist; a program fit on clean data; ≥ 80% of counterexamples satisfy the twist condition. Re-check on real bb-01 at integration.
- **evaluate():** `env_factory(seed)` returns `(env, first_obs)`; close envs after use.

### B-Brain
- **Clients:** `openai.OpenAI(base_url=..., api_key=..., timeout=60)`. Muse Spark is a reasoning model: read `choices[0].message.content` for the answer, set `max_tokens` ≥ 4000. If a provider rejects `temperature` or `response_format`, retry once without it and remember that for the session.
- **Live mode:** per-call timeout 35 s; `LIVE_MAX_TURNS` turns; whole run capped at `LIVE_TIMEOUT_S`.
- **JSON robustness:** strip ```json fences, take the first balanced `{...}`, validate with pydantic, retry ≤ 2× with the validation error appended.
- **Ledger:** `rule_id: null` → assign the next `rN`. If the brain sends `status: "confirmed"`, downgrade it to `inferred`. Only `agent/probes.py` sets `confirmed`/`refuted`.
- **Compact context:** round floats to 3 decimals, cap tables (≤ 12 counterexamples, ≤ 20 probe rows), fixed-width columns. `CONTEXT_MODE=full` appends all query batches as CSV.
- **Program versions:** `changed_lines` via `difflib` against the previous version's code.
- **Frames:** emit a `frames` event only when a new program version improves holdout agreement (keeps payloads down).
- **Mock path must be fast:** with USE_MOCKS=1 and a real brain, a run should finish in < 2 min. That's what B-Show and integration rely on.
- **Store:** JsonStore first (works offline). MongoStore only after B3 passes, ≤ 15 min, auto-fallback to JSON (CLAUDE.md B4).

### B-Show
- **Fixture first** (`scripts/make_fixture.py` → `fixtures/demo_events.jsonl`): a fake but realistic run so the UI is built without any agent:
  - real CartPole frames (render a simple hand policy for "bb" and a worse one for "clone v1", a better one for "clone v3")
  - 3 program versions with growing code, a ledger that goes inferred → refuted / confirmed, metric points 40% → 93%, query counts climbing, then reveal, fix and run_finished events
  - realistic timestamps over ~60 s
- **WebSocket bridge:** the run thread calls `emit(event)`. In `emit`, append to a per-run buffer and `loop.call_soon_threadsafe(queue.put_nowait, event)` for each connected socket. **On connect, send the whole buffer first**, so a page that connects late (or reloads mid-demo) catches up.
- **Replay:** `/api/replay` reads a `.jsonl` (fixture, `runs/<id>`, or `cache/<key>/events.jsonl`) and re-emits on a timer with original spacing ÷ speed, through the same bridge.
- **Clip playback:** one `setInterval` at `fps` drives both `<img>` elements from the same frame index (`img.src = "data:image/jpeg;base64," + b64`). Red border class on the clone panel when `disagree[i]`. Swap to new clips at the next loop start, not mid-clip.
- **Code panel:** use `changed_lines` from the event; glow those lines for 3 s.
- **Curve:** SVG `<polyline>` in a fixed viewBox, axes 0–100% and 0–budget.
- **Projector check:** 1920×1080, no scrollbars, ledger ≥ 18 px, code ≥ 16 px, high-contrast dark theme. Footer credit line from CLAUDE.md B6.
- **Live fallback banner:** "Live run exceeded time limit, replaying cached run of this exact configuration." Never silent.

---

## 5. Timeline (now → win)

| Time | A-World | A-Proof | B-Brain | B-Show |
|---|---|---|---|---|
| 11:30–11:45 | Skeleton pushed. Both: clone, `.venv`, `.env`, **claim Meta credits**, start all 4 sessions | | | |
| 11:45–1:15 | A1→A3, demo seed | A4→A5 | B1→B4 | fixture→B5→B6 |
| **1:15–1:45** | **Integration** (both people together, section 6) | | | |
| 1:45–2:30 | Tune LunarLander twists + demo seed | A6 batch: start runs for both brains in background | Live-path speed (≤ 75 s on bb-05) | B7: fix + shield + results tab |
| **2:30** | **Feature freeze.** Only bug fixes and demo polish after this. | | | |
| 2:30–3:15 | Demo polish (section 7) | Fill cache for every policy × flaw (Muse + SubQ) | README + results | UI final pass on the projector |
| 3:15–3:45 | Record backup run of the full demo; record 1-min video footage | | | |
| 3:45–4:15 | Voiceover, edit, upload; submit (repo public, link in incognito, both teammates) | | | |
| 4:15–5:00 | Rehearse the 3-minute pitch at least 5 times with a timer; drill Q&A | | | |

**Cut rule:** if CartPole isn't end to end by 1:45, drop Acrobot, MountainCar and all stretch; CartPole + LunarLander only.

---

## 6. Integration (1:15–1:45)

1. `./sync.sh "pre-merge"` on both laptops, then pull.
2. Laptop A runs the black box: `uvicorn blackbox.server:app --host 0.0.0.0 --port 8001`. Laptop B sets `BLACKBOX_URL=http://<A's local IP>:8001` (or run both servers on one laptop; simpler and safer on venue Wi-Fi).
3. `USE_MOCKS=0 python -m scripts.smoke` → all green.
4. Headless: `python -m agent.loop --bb bb-01 --env CartPole-v1 --brain muse` with t-cp-1 set via the UI or admin endpoint. Twist rule should show up confirmed.
5. UI: pick bb-01 + Hidden flaw, Run, watch every panel move. Reveal → Fix → Shield.
6. Same headless run with `--brain subq`.
7. `pytest -q` all green. `./sync.sh "integrated"`.

**Demo machine decision:** run **everything on one laptop** for the demo (both servers local). Fewer network hops, no venue Wi-Fi between your two processes.

---

## 7. Demo polish checklist (from 2:30)

- [ ] Hero config locked: bb-05 (LunarLander-PPO), t-ll-1, demo seed from the finder. Clean lands, twisted crashes on the right side. Clone reproduces the crash (that's the proof). Shield lands it.
- [ ] Live run on the hero config finishes in ≤ 60 s, 3 times in a row. If not, lower turns/queries until it does.
- [ ] Cache exists for every policy × flaw the judge can pick, for Muse (and SubQ for the results tab). Fallback banner tested.
- [ ] Results tab shows both brains, cross-match, zero-query baseline, one honest failure.
- [ ] Browser full-screen, zoom set for the projector, notifications off, laptop on power, Wi-Fi dependency known (the brain API is the only network call).
- [ ] Backup recording of the full live demo saved locally, one click away.
- [ ] Pitch rehearsed 5×, under 3:00, every placeholder filled with real numbers.
