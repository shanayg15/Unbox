# CLAUDE.md: Unbox build spec

> Unbox turns a black-box robot policy (a neural net behind an API) into short readable Python code that behaves the same, with a ledger of evidence for every rule. Then it finds the rule behind a hidden flaw, fixes it, and verifies nothing else changed.

Hackathon: AIE Code Summit, Sat Nov 7 2026. Track: **Software Archaeology**. Submissions due **5:00 PM**. Repo must be public. Only work done today counts.

---

## 0. How to use this file

- This file is the build spec. `LANES.md` (next to it) says who builds what, in four parallel Claude Code sessions: **A-World, A-Proof** (Person A) and **B-Brain, B-Show** (Person B). Each session's prompt and owned folders are in LANES.md section 3; stay inside your owned folders.
- Every task has a **Done when** check. Run it and show the output before moving to the next task.
- Sessions only touch through `contracts/` (section 5) and the internal contracts in LANES.md section 2. Lane B runs on mocks until integration. Lane A never needs Lane B's code until A6.

**Git workflow:**
- Claude sessions never run git commands. The human runs `./sync.sh "message"` (LANES.md section 1).
- Everyone works on `main`. Sessions own separate folders, so conflicts are rare.
- Shared files (`requirements.txt`, `.env.example`, `contracts/`): add lines, never rewrite; tell the human when you touch one.
- Never commit `.env`. Never force-push.

---

## 1. Hard rules (do not break these)

1. **Isolation.** The agent never sees weights, checkpoint names, env ids or the active twist.
   - `agent/` must never import `stable_baselines3`, `huggingface_sb3`, `rl_zoo3`, `torch`, or anything from `blackbox/` except `blackbox/client.py`.
   - `agent/` never calls `/admin/*` endpoints. Only `app/` holds the admin token.
   - `tests/test_isolation.py` enforces both with a grep. Keep it passing.
2. **The brain prompt contains only:** the sensor spec (obs names, bounds, action names), evidence (query results, counterexamples, pattern hints), the current program, the ledger, and the remaining budget. Never the env id, policy label or twist. (Exception: the zero-query baseline in A6, which is explicitly a baseline.)
3. **The LLM never sets numbers.** Every threshold or gain in a program is a named parameter in `P` with an initial guess. The optimizer fits it.
4. **The data decides.** Agreement, eval metrics, and the `confirmed`/`refuted` ledger statuses are computed by Python, never by the LLM.
5. **Log every brain call** to `runs/<run_id>/brain_calls.jsonl`: provider, model, latency_ms, prompt_tokens, completion_tokens, ok/error. This is the evidence for each sponsor prize.
6. **UI is plain HTML/JS served by FastAPI.** No Streamlit (banned by the rules). The live run is the feature; the results tab is supporting.
7. **Credit third-party policies** (SB3 RL Zoo checkpoints from Hugging Face) in the README and in the UI footer.

---

## 2. Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate    # use 3.11 on both Macs (brew install python@3.11)
pip install swig            # Box2D (LunarLander) needs swig before box2d builds
pip install -r requirements.txt
cp .env.example .env        # fill in keys
```

`requirements.txt`
```
gymnasium[box2d]>=1.0
stable-baselines3>=2.3
huggingface_sb3
numpy
scipy
cma
scikit-learn
fastapi
uvicorn[standard]
httpx
openai>=1.40
pydantic>=2
pymongo
python-dotenv
pillow
pytest
```

`.env.example`
```
# Brain (writes code): muse
BRAIN=muse

# Meta Model API (OpenAI-compatible)
MODEL_API_KEY=
META_BASE_URL=https://api.meta.ai/v1
META_MODEL=muse-spark-1.1          # run GET /v1/models and use the newest Muse Spark id

# SubQ Snap: Decisions API (NOT chat). POST {SUBQ_BASE_URL}/systemone. Evidence scanner, see section 7 B2b.
SUBQ_API_KEY=
SUBQ_BASE_URL=https://api.subq.ai/v1
SUBQ_MODEL=subq-snap-1.1
SNAP=1                             # 1 = run the SubQ evidence scan each turn; 0 = ablation (batch only)

# Black-box server
BLACKBOX_URL=http://localhost:8001
BLACKBOX_ADMIN_TOKEN=change-me

# Storage: json (default) | mongo  (both teammates point at the SAME Atlas cluster)
STORE=json
MONGODB_URI=                       # mongodb+srv://... from Atlas → Connect → Drivers → Python
MONGODB_DB=unbox

# Dev
USE_MOCKS=1                        # 1 = Lane B runs on contracts/mocks.py
QUERY_BUDGET=5000
LIVE_MAX_TURNS=4
LIVE_TIMEOUT_S=75
CONTEXT_MODE=compact               # compact | full (full = all raw evidence; showcases long context)
```

**Run**
```bash
uvicorn blackbox.server:app --port 8001          # Lane A
uvicorn app.server:app --port 8000               # Lane B → open http://localhost:8000
python -m scripts.smoke                          # checks keys, black box, numerics
python -m agent.loop --bb bb-01 --twist t-cp-1 --brain muse   # headless run
python -m scripts.batch --snap 1,0               # A6: precompute results + caches, with/without SubQ
pytest -q
```

---

## 3. Repo layout and owners

```
unbox/
  CLAUDE.md
  README.md                 # B writes; A adds policy credits
  requirements.txt  .env.example
  contracts/                # SHARED. Copied verbatim from section 5. Change only together.
    types.py
    numerics_api.py
    events.py
    mocks.py                # B owns (MockBlackBox, numerics stub)
  blackbox/                 # A
    policies.py             # registry + loading
    twists.py               # hidden flaws
    server.py               # FastAPI :8001
    client.py               # HTTP client implementing contracts.types.BlackBox
  numerics/                 # A (pure Python, no LLM)
    sandbox.py  fit.py  prober.py  evaluate.py  patterns.py  shield.py
    __init__.py             # re-exports everything in contracts/numerics_api.py
  agent/                    # B
    brain.py  prompts.py  protocol.py  probes.py  ledger.py  loop.py  fixer.py
  store/                    # B
    store.py                # json + mongo backends, same interface
  app/                      # B
    server.py               # FastAPI :8000, REST + WebSocket, serves static/
    static/index.html  static/app.js  static/style.css
  scripts/
    smoke.py                # B-Brain
    batch.py                # A-Proof (after merge)
    make_fixture.py         # B-Show
  fixtures/                 # B-Show: fake event streams for building the UI
  sync.sh                   # human-only git sync (LANES.md section 1)
  runs/                     # per-run logs (gitignored except demo runs)
  cache/                    # precomputed runs for the live-demo fallback
  results/                  # batch outputs for the results tab
  tests/
    test_isolation.py       # whoever creates the repo
    test_a_*.py  test_b_*.py
```

---

## 4. Build order and the merge point

The full timeline is in LANES.md section 5. Key checkpoints:
```
11:45–1:15   Four sessions in parallel (A-World, A-Proof, B-Brain, B-Show), Lane B on mocks
1:15–1:45    INTEGRATION: one fresh Integrator session on the demo laptop, USE_MOCKS=0, CartPole end to end
1:45–2:30    A6 batch (SubQ on/off ablation) · B7 fix + shield + results tab · LunarLander live path ≤ 75 s
2:30         FEATURE FREEZE (bug fixes and demo polish only after this)
2:30–4:15    Cache every combo, README, backup recording, 1-min video, submit
4:15–5:00    Rehearse the pitch
```

**Cut rule:** if CartPole isn't end to end by 1:45, drop Acrobot/MountainCar and all stretch work; make CartPole + LunarLander great.

---

## 5. Contracts (copy verbatim, then don't drift)

### `contracts/types.py`
```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, Union

Obs = list[float]
Action = Union[int, list[float]]
Params = dict[str, float]
ProgramFn = Callable[[Obs, Params], Action]
ActionType = Literal["discrete", "continuous"]


@dataclass
class SensorSpec:
    obs_names: list[str]
    obs_low: list[float]           # empirical p1 if the space is unbounded
    obs_high: list[float]          # empirical p99 if the space is unbounded
    action_type: ActionType
    action_names: list[str]        # discrete: one name per action id; continuous: one per dim
    action_low: list[float] | None = None
    action_high: list[float] | None = None


@dataclass
class Episode:
    seed: int
    obs: list[Obs]
    actions: list[Action]
    rewards: list[float]
    total_reward: float
    terminated: bool
    truncated: bool
    frames: list[str] | None = None        # base64 JPEG, one every `frame_every` steps
    frame_every: int = 2


@dataclass
class QueryBatch:
    query_id: str                           # "q17"; assigned by the black box
    obs: list[Obs]
    actions: list[Action]
    source: Literal["rollout", "probe", "prober", "holdout"]
    rule_id: str | None = None
    inference_us: float | None = None       # mean model time per obs, measured server-side


class BlackBox(Protocol):
    def spec(self) -> SensorSpec: ...
    def act(self, obs_batch: list[Obs], source: str = "probe", rule_id: str | None = None) -> QueryBatch: ...
    def episode(self, seed: int, render: bool = False, frame_every: int = 2) -> Episode: ...
    def query_count(self) -> int: ...


@dataclass
class Program:
    version: int
    code: str
    params: Params
    agreement: float | None = None          # open-loop agreement on holdout after fitting


@dataclass
class Counterexample:
    obs: Obs
    bb_action: Action
    clone_action: Action
    query_id: str


@dataclass
class EvalReport:
    open_loop_agreement: float
    n_holdout_obs: int
    clone_mean_reward: float | None = None
    bb_mean_reward: float | None = None
    n_episodes: int = 0
    n_failures: int = 0
    failure_rate_bound_95: float | None = None   # rule of three: 3/n when n_failures == 0
    us_per_action_clone: float | None = None
    us_per_action_bb: float | None = None


LedgerStatus = Literal["observed", "inferred", "probing", "confirmed", "refuted"]


@dataclass
class LedgerEntry:
    rule_id: str                             # "r1", "r2", ... assigned by Python
    text: str                                # plain English, one line
    status: LedgerStatus
    confidence: float                        # 0..1
    evidence: list[str] = field(default_factory=list)   # query_ids
    program_version: int | None = None
```

### `contracts/numerics_api.py`
Signatures only. Lane A implements them in `numerics/`; Lane B calls them (or the stub in `mocks.py`).
```python
from contracts.types import *

class ProgramError(Exception):
    """Bad program. str(e) is fed back to the brain verbatim, so make it helpful."""

def load_program(code: str, spec: SensorSpec) -> ProgramFn: ...
    # AST-validates, compiles, test-runs on 50 sample obs, returns fn(obs, P) -> Action.

def agreement(fn: ProgramFn, params: Params, X: list[Obs], y: list[Action], spec: SensorSpec) -> float: ...

def fit_params(fn: ProgramFn, init: Params, X: list[Obs], y: list[Action], spec: SensorSpec,
               max_seconds: float = 8.0) -> tuple[Params, float]: ...
    # returns (best_params, agreement_on_X)

def pattern_hints(X: list[Obs], y: list[Action], spec: SensorSpec, max_depth: int = 4) -> str: ...
    # decision-tree splits as readable text, with per-leaf sample counts and accuracy

def find_disagreements(fn: ProgramFn, params: Params, bb: BlackBox, seed_obs: list[Obs],
                       spec: SensorSpec, k: int = 8, max_queries: int = 400) -> list[Counterexample]: ...

def evaluate(fn: ProgramFn, params: Params, bb: BlackBox, spec: SensorSpec,
             holdout: list[QueryBatch], env_factory: Callable[[int], object],
             seeds: list[int]) -> EvalReport: ...
    # env_factory(seed) -> gymnasium env already reset with that seed; returns (env, first_obs)

def shielded_action(recovered: ProgramFn, rec_params: Params, fixed: ProgramFn, fix_params: Params,
                    obs: Obs, bb_action: Action) -> Action: ...
    # if recovered(obs) != fixed(obs): return fixed(obs), else return bb_action
```

### `contracts/events.py`
Every run emits these events, in order, to the store (`runs/<run_id>/events.jsonl`) and to the UI WebSocket. The UI is built against this file only.
```python
# Envelope: {"type": str, "run_id": str, "t": float (seconds since run start), "data": {...}}

EVENT_TYPES = {
    "run_started":  "{policy_label, brain, model, budget, live: bool}",
    "status":       "{text}                                  # one-line narration, e.g. 'Probing r3: sweep x 0.3→0.8'",
    "query_count":  "{n}",
    "ledger":       "{entries: [LedgerEntry as dict]}        # full snapshot every time",
    "program":      "{version, code, params, changed_lines: [int], agreement}",
    "metric":       "{queries, agreement}                    # one point on the match-vs-queries curve",
    "frames":       "{version, seed, fps, bb: [b64], clone: [b64], disagree: [bool]}  # whole clip",
    "reveal":       "{label, feature, highlight_rule_ids: [str]}",
    "fix":          "{code, params, diff: str}",
    "shield":       "{on: bool, bb: [b64] | null}            # shielded black-box clip when on",
    "run_finished": "{eval: EvalReport as dict, total_queries, turns, seconds, cached: bool}",
    "error":        "{message}",
    "snap":         "{turn, n_rows, input_tokens, feature_relevance: {feature: p}, anomaly_feature: {choice, probabilities, confidence}, next_probe: {choice, probabilities, confidence}, rule_support: {rule_id: p}}",
}
```

---

## 6. Lane A: World & Proof (black box + numerics)

### A1. Black-box server skeleton + CartPole
`blackbox/policies.py`: registry of opaque ids. Verify each HF repo id loads; drop any that fails within 10 minutes.

| bb id | label (UI only) | env id | HF repo (verify) | role |
|---|---|---|---|---|
| bb-01 | CartPole-PPO | CartPole-v1 | `sb3/ppo-CartPole-v1` | integration target |
| bb-02 | CartPole-DQN | CartPole-v1 | `sb3/dqn-CartPole-v1` | cross-match |
| bb-05 | LunarLander-PPO | LunarLander-v3 | `sb3/ppo-LunarLander-v2` | **live demo** |
| bb-06 | LunarLander-DQN | LunarLander-v3 | `sb3/dqn-LunarLander-v2` | cross-match |
| bb-03 | MountainCar-DQN | MountainCar-v0 | `sb3/dqn-MountainCar-v0` | results table |
| bb-04 | Acrobot-PPO | Acrobot-v1 | `sb3/ppo-Acrobot-v1` | results table |

Loading tips:
- `load_from_hub(repo_id, filename="<algo>-<EnvName>.zip")`, then `PPO.load(path, custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0})` to dodge schedule-unpickling errors.
- LunarLander-v2 checkpoints run on gymnasium's `LunarLander-v3` (same 8-dim obs, 4 actions).
- Always `model.predict(np.array(batch), deterministic=True)` on the whole batch at once.

Obs names (sensor spec):
- CartPole: `cart_x, cart_v, pole_angle, pole_ang_v`; actions `push_left, push_right`
- LunarLander: `x, y, vx, vy, angle, ang_v, leg_left, leg_right`; actions `noop, fire_left, fire_main, fire_right`
- MountainCar: `position, velocity`; actions `push_left, noop, push_right`
- Acrobot: `cos1, sin1, cos2, sin2, v1, v2`; actions `torque_neg, torque_0, torque_pos`

Public endpoints (no auth):
- `GET  /spec?bb=bb-01` → `SensorSpec`
- `POST /act` `{bb, obs: [[...]], source, rule_id}` → `{query_id, actions, inference_us}`. Increments the query counter by `len(obs)`. Applies the active twist.
- `POST /episode` `{bb, seed, render, frame_every}` → `Episode`. Policy rolls itself out (with the twist). Cache by `(bb, twist, seed, render)`.
- `GET  /count?bb=` → `{queries}`

`blackbox/client.py`: `HttpBlackBox(bb_id, url)` implementing `contracts.types.BlackBox` with httpx. No admin calls in this file.

**Done when:** `/act` on 256 obs returns 256 actions; `/episode` on bb-01 seed 0 returns reward ≈ 500; `/episode?render=true` frames decode as JPEGs.

### A2. Hidden flaws (twists) + admin endpoints
`blackbox/twists.py`: each twist = `{twist_id, env, hidden_label, reveal_label, feature, condition(obs)->bool, override(obs, a)->a}`. Applied server-side after the policy acts.

Starting set (tune thresholds; factory-language reveal labels are what the audience sees):

| twist_id | env | rule (tune the number) | reveal label |
|---|---|---|---|
| t-ll-1 | LunarLander | `x > 0.2` and action == fire_main → noop | "Skips its safety action on the right side" |
| t-ll-2 | LunarLander | `angle > 0.15` → fire_left | "Over-corrects when it tilts" |
| t-ll-3 | LunarLander | `y > 1.0` and `vy < -0.3` → noop | "Ignores its speed limit up high" |
| t-cp-1 | CartPole | `cart_v > 0.5` → push_left | "Over-reacts when moving fast" |
| t-cp-2 | CartPole | `cart_x < -0.5` → push_right | "Drifts in one zone of the track" |
| t-mc-1 | MountainCar | `position < -1.0` → noop | "Freezes in one corner of its workspace" |

Admin endpoints (header `X-Admin-Token`):
- `GET  /admin/policies` → `[{bb, label, env_id}]`
- `GET  /admin/twists?bb=` → `[{twist_id, hidden_label}]` (hidden labels only: "Hidden flaw #1"…)
- `POST /admin/config` `{bb, twist_id | null}` → sets active twist, clears episode cache, resets counter
- `GET  /admin/reveal?bb=` → `{twist_id, reveal_label, feature, rule_text}`

**Done when:** `python -m blackbox.twists --report` prints each twist's fire rate on seeds 0–9: **5–30% of steps** (visible in the demo, not dominant), and actions differ from the clean policy only where the condition holds.

### A3. Remaining policies + rendering
Load bb-02…bb-06, frames via `env.render()` (`render_mode="rgb_array"`), downscale to 480 px wide, JPEG quality 70, base64.

**Done when:** every bb id serves `/spec`, `/act`, `/episode` (rendered), and twists for its env apply.

### A4. Numerics core: sandbox, agreement, fit, patterns
`numerics/sandbox.py` `load_program(code, spec)`:
- AST whitelist: exactly one `def policy(obs, P):`. Allowed: assignments, `if/elif/else`, `return`, arithmetic, comparisons, boolean ops, calls to `abs, min, max, float, int, round, len` and `math.*`, `np.*`. **Forbidden:** `import`, `while`, `for`, `lambda`, `global`, `with`, `try`, recursion, any attribute or name starting with `_`. (No loops = no infinite loops = no timeout machinery needed.)
- Exec with `{"__builtins__": SAFE_BUILTINS, "math": math, "np": np}`.
- Test-run on 50 sample obs (uniform in spec bounds); actions must be a valid id (discrete) or a list of the right length (continuous).
- Raise `ProgramError` with a specific message (line number, bad node, bad return value).

`numerics/fit.py`:
- `agreement`: discrete = exact-match fraction; continuous = fraction with L2 error ≤ 0.1 × action range.
- `fit_params`: discrete → CMA-ES over normalized params (`p = init + scale*z`, `scale = max(|init|, 0.1)`), popsize 16, stop at `max_seconds`, objective `1 - agreement` on ≤ 3,000 subsampled points. Continuous → `scipy.optimize.least_squares` on residuals. Return best params and agreement. Never raise on a bad param region; just score it 0.

`numerics/patterns.py` `pattern_hints`: `DecisionTreeClassifier(max_depth=4, min_samples_leaf=20)` → `export_text` with obs names and action names, plus per-leaf counts and purity. Continuous: one `DecisionTreeRegressor(max_depth=3)` per action dim.

**Done when** (`tests/test_a_fit.py`): with a rule-based fake black box (`RuleBlackBox(fn, true_params)`, written in the test) and a program with the right structure but wrong params, `fit_params` raises agreement from < 0.7 to > 0.97. Sandbox rejects an `import os`, a `while True`, and `obs.__class__`.

### A5. Prober, evaluator, shield
`numerics/prober.py` `find_disagreements`:
1. Candidates: seed obs + Gaussian noise (σ = 0.1 × empirical std per feature) and per-feature sweeps across the empirical p1–p99 range holding other features at their median.
2. Query the black box in batches (`source="prober"`), compare with the clone, keep disagreements.
3. One refinement round: perturb around found disagreements to map the region.
4. Return the `k` most diverse disagreements (farthest-point selection). Respect `max_queries` exactly.

`numerics/evaluate.py` `evaluate`:
- Open-loop agreement on the holdout batches.
- Closed-loop: for each seed, black-box `episode(seed)` total reward vs the clone driving a local env from the same seed. Failure = clone reward < bb reward − max(10, 0.1·|bb reward|).
- `failure_rate_bound_95 = 3/n` when there are 0 failures.
- `us_per_action_clone` timed locally over 1,000 calls; `us_per_action_bb` from the server's `inference_us`.

`numerics/shield.py`: per the contract.

**Done when:** against bb-01 with t-cp-1 active and a program fit only on clean behavior, ≥ 80% of returned counterexamples satisfy the twist condition (`cart_v > threshold`). `evaluate` returns a full `EvalReport` in < 20 s for 20 seeds.

### A6 (after merge). Batch, cross-match, baselines → `scripts/batch.py`
For every bb id × {no twist, each twist} × `SNAP` in {1, 0} (Muse brain), add to each row `snap: bool`, `queries_to_twist`, `turns_to_twist` (first time the twist rule is confirmed; null if never):
- run `agent.loop.run(..., live=False)` headless, save the run, copy demo-relevant runs to `cache/<bb>__<twist>__<brain>/`.
- **Cross-match:** evaluate each recovered program against the other policy of the same env (bb-01 vs bb-02, bb-05 vs bb-06). Expect diagonal high, off-diagonal lower.
- **Zero-query baseline:** brain gets the task name + sensor spec and no queries (`prompts.ZERO_QUERY`), write a controller; evaluate it. Expect decent reward, low agreement.
- Write `results/summary.json`: per row `{bb, label, twist, brain, model, agreement, clone_reward, bb_reward, bound_95, lines_of_code, queries, turns, seconds, speedup, twist_found: bool}`, plus `results/crossmatch.json` and `results/baselines.json`.
- `twist_found` = a confirmed ledger rule mentions the twist's feature with a threshold within 20% of the true one.

---

## 7. Lane B: Brain & Show (agent, store, app, UI)

### B1. Mocks + smoke
`contracts/mocks.py`:
- `MockBlackBox`: a local CartPole env with a hand-written policy (`push_right if pole_angle + 0.5*pole_ang_v > 0 else push_left`) plus the t-cp-1 twist; implements `BlackBox` fully (query ids, counter, rendered episodes).
- `numerics_stub`: same signatures as `contracts/numerics_api.py`; `load_program` via plain `exec`, `fit_params` returns `(init, agreement)`, `find_disagreements` = random sampling, `evaluate` = open-loop only.
- `agent/loop.py` picks real vs mock with `USE_MOCKS`.

`scripts/smoke.py`: pings the configured brain(s), the black box (real or mock), and numerics; prints a green/red table.

**Done when:** `USE_MOCKS=1 python -m scripts.smoke` is all green except whichever API key is missing.

### B2. Brain client (Muse writes the code)
`agent/brain.py`:
- `Brain("muse")` → `openai.OpenAI(base_url=META_BASE_URL, api_key=MODEL_API_KEY)`; model `META_MODEL`.
- `turn(messages) -> dict`: chat completions; extract JSON (raw object, or a ```json fence, or the first balanced `{...}`); validate with `protocol.BrainTurn`; on failure, retry up to 2× appending the validation error. Don't depend on `response_format` support; try it, fall back silently.
- Retry 429/5xx with backoff (1 s, 2 s, 4 s). Log every call (rule 5).
- `python -m agent.brain --ping` sends a toy turn and prints the parsed JSON.
- (Optional, only if the SubQ booth enables `subq-preview` chat for our key: a second provider `"subq_chat"` with the same client code. Not on the critical path.)

**Done when:** the Muse ping works and appears in `brain_calls.jsonl`.

### B2b. SubQ Snap evidence scan (`agent/snap.py`)
SubQ Snap 1.1 is a **decision model**, not a chat model: it answers named yes/no, choice and score questions with calibrated probabilities over up to **384K tokens** of input, and generates no text. Unbox uses it as the **evidence scanner**: every turn it reads the *entire raw query log* (no summarizing) and tells Muse where to look.

API: `POST {SUBQ_BASE_URL}/systemone`, `Authorization: Bearer {SUBQ_API_KEY}`, body `{"model": SUBQ_MODEL, "state": <text>, "questions": {name: question}}`. Question types: `"noul"` (yes/no → `answers[name].noul` = P(yes)), `"choice"` (`criteria` = map of option → description or null → `answers[name].choice`, `.probabilities`, `.confidence`). Max 128 questions per call. Errors: 400 over token limit, 402 out of credits, 422 bad request, 429 rate limit. Docs: https://docs.subq.ai/endpoints/decisions-api/

`scan(spec, rows, counterexamples, ledger, run_id) -> dict`:
- **state** (text): sensor spec, then every query row as one line `q12 | x=0.312 y=1.020 ... -> fire_main` (all sources, 3 decimals), then counterexamples labeled `[C1]..[Ck]`, then active ledger rules labeled `[r1]..`. Keep under ~1.2M characters (≈ 300K tokens); if over, drop the oldest rollout rows first and record `n_rows` actually sent.
- **questions:**
  - `rel_<feature>` (noul), one per obs feature: "Across all rows, does the robot's action depend on <feature>, i.e. do states that differ mainly in <feature> get different actions?"
  - `anomaly_feature` (choice over feature names + `none`): "Which feature best separates the counterexamples [C*] from the other rows?"
  - `next_probe` (choice over feature names): "Probing which feature next is most likely to reveal behavior the current rules do not explain?"
  - `rule_<rid>` (noul), one per non-refuted ledger rule: "Is this rule consistent with all rows: <rule text>?"
- Return `{feature_relevance, anomaly_feature, next_probe, rule_support, n_rows, input_tokens}`; emit a `snap` event; log the call to `brain_calls.jsonl` with `provider="subq"`, `model`, `input_tokens`.
- **Advisory only (rule 4):** Snap probabilities are shown and fed to Muse, but `confirmed`/`refuted` still come only from Python probe verdicts.
- Timeout 20 s; on any error, log it, emit `status` "SubQ scan skipped", and continue the turn without it. Never block the loop.

**Done when:** `python -m agent.snap --ping` sends a 2-question request and prints the answers, and on the mock run the `snap` event shows `cart_v` (the t-cp-1 twist feature) as `anomaly_feature` with the highest probability at least once.

### B3. Protocol, prompts, probes, ledger, loop (on mocks)
`agent/protocol.py` (pydantic):
```python
class Sweep(BaseModel):
    feature: str; start: float; stop: float; n: int = Field(le=60)

class Expect(BaseModel):
    feature: str; op: Literal[">", "<"]; value: float
    action: int | list[float]            # predicted black-box action inside this region

class ProbeSpec(BaseModel):
    rule_id: str | None                  # None = new rule
    base: Literal["median"] | list[float] = "median"
    sweep: Sweep
    sweep2: Sweep | None = None          # 2-D grid; n ≤ 15 each
    expect: Expect

class LedgerUpdate(BaseModel):
    rule_id: str | None                  # None = new rule (Python assigns r1, r2...)
    text: str
    status: Literal["observed", "inferred", "probing", "refuted"]   # never "confirmed"
    confidence: float
    evidence: list[str] = []

class ProgramOut(BaseModel):
    code: str; params: dict[str, float]

class BrainTurn(BaseModel):
    summary: str                         # one sentence, shown as the status line
    ledger: list[LedgerUpdate] = []
    program: ProgramOut | None = None
    probes: list[ProbeSpec] = Field(default=[], max_length=3)
    done: bool = False
```

`agent/probes.py`: expand a `ProbeSpec` into obs (median of seen data, overwrite swept features), query the black box (`source="probe"`, `rule_id`), then the **Python verdict**: among probe points inside the `expect` region, fraction where bb action == expected → ≥ 0.85 `confirmed`, ≤ 0.4 `refuted`, else stays `probing` with `confidence = fraction`. Attach the query id as evidence.

`agent/prompts.py`:

SYSTEM (send as-is, fill `{...}`):
```
You are Unbox, a reverse-engineering agent. A robot controller (a neural network) is hidden behind an API.
You only see what it does. Write a short, readable Python program that makes the same decisions, and keep
a ledger of the rules you believe it follows, each backed by evidence.

Respond with ONLY one JSON object matching this schema: {schema}

Program rules:
- def policy(obs, P): obs is a list of floats in sensor-spec order. Return an int action id (discrete) or a
  list of floats (continuous).
- Start by naming features: x = obs[0], etc. Then if/elif/else rules. Arithmetic, comparisons, math.*, np.* only.
  No loops, no imports, nothing starting with "_". At most 40 lines.
- Every threshold or gain is a parameter in P with an initial guess. Never hard-code a number you guessed.
  An optimizer fits P to the data after you answer.
- Prefer a few rules that explain most decisions. Add a rule only when counterexamples demand it.

Ledger rules:
- One entry per rule, plain English, citing query ids as evidence.
- Status: observed (directly visible in data), inferred (your hypothesis), probing (you are testing it now),
  refuted (evidence contradicts it). You can never mark a rule confirmed; the system does that from probes.

Probes (max 3 per turn): design experiments that could REFUTE your hypothesis, not just support it.
Set done=true when agreement stops improving.
```

TURN (context; `compact` mode by default):
```
SENSOR SPEC: {spec}
BUDGET: {queries_used}/{budget} queries, turn {turn}/{max_turns}
CURRENT PROGRAM (v{version}, holdout agreement {agreement:.1%}, fitted params {params}):
{code}
LEDGER: {ledger_table}
PATTERN HINTS (decision tree on {n} labeled states): {hints}
SUBQ EVIDENCE SCAN (SubQ Snap read all {n_rows} raw rows): feature relevance {relevance}; counterexamples best separated by {anomaly_feature} (p={p}); suggested next probe: {next_probe}; rule support {rule_support}
PROBE RESULTS since last turn: {probe_table}
COUNTEREXAMPLES (clone disagrees with robot): {counterexample_table}
LAST ERROR (if any): {error}
```
`CONTEXT_MODE=full` also appends every query batch as CSV to the Muse prompt (rarely needed now that SubQ Snap reads the full raw log).

FIX (after reveal):
```
Here is the recovered program and ledger. Engineers report this unwanted behavior: "{reveal_label}".
Return JSON {"program": {"code", "params"}, "changed_rule_ids": [...], "explanation": "..."}.
Change as few lines as possible. Every other behavior must stay identical.
```

ZERO_QUERY (baseline only, used by A6): task name + sensor spec, "write a controller that performs this task well", same program rules.

`agent/loop.py` `run(...)`: exact signature in LANES.md section 2 (the app passes in `bb`, `env_factory` and `numerics`; the agent never builds envs or learns the env id). Steps:
```
1. spec = bb.spec(); rollouts on seeds 0..7 (source=rollout); holdout on seeds 100..107 (source=holdout)
2. hints = pattern_hints(rollout data)
3. for turn in 1..max_turns:
     if SNAP=1: scan = snap.scan(...)               → emit snap   (skip silently on error)
     out = brain.turn(context incl. scan)           → emit status(out.summary)
     apply ledger updates                            → emit ledger
     if out.program: fn = load_program(...)          (ProgramError → feed back next turn)
                     params, _ = fit_params(...)     on all non-holdout data
                     agreement on holdout            → emit program, metric
                     clip = render_pair(fn, params, demo_seed) → emit frames
     run probes → verdicts                           → emit ledger, query_count
     cex = find_disagreements(...)                   → add to data, emit query_count
     refresh hints every 2 turns
     stop if agreement ≥ 0.97, no +1% gain in 2 turns, out.done, budget spent, or live timeout
4. report = evaluate(...)                            → emit run_finished
```
`render_pair`: black-box clip from `bb.episode(demo_seed, render=True)` (cached); clone clip from a local env with the same seed driven by the program; `disagree[t] = fn(bb_obs[t]) != bb_action[t]` on the black box's trajectory.

`agent/fixer.py` `fix(...)`: FIX prompt wrapper, signature in LANES.md section 2.

**Done when:** `USE_MOCKS=1 python -m agent.loop --bb mock --brain muse` finishes in ≤ 6 turns, the final program has ≥ 0.9 agreement on the mock, the ledger has at least one refuted and one confirmed rule, and `runs/<id>/events.jsonl` validates against `contracts/events.py`.

### B4. Store
`store/store.py`: `Store` interface with `save_event, save_brain_call, save_program, save_ledger, save_eval, list_runs, load_events(run_id)`. `JsonStore` writes `runs/<run_id>/*.jsonl`. `MongoStore` (database `MONGODB_DB`; collections `runs, events, programs, ledger, queries, brain_calls`) only if Atlas works in ≤ 15 min; otherwise stay on JSON.
- `MongoClient(uri, serverSelectionTimeoutMS=5000)` and ping on startup. If the ping fails, log one warning and **fall back to JsonStore automatically**. Never let the database hang a live run.
- Use `insert_many` for query batches (the free tier caps throughput). Index `events(run_id, t)` and `queries(run_id)`.
- Always write JSON too (cheap), so runs replay even if Atlas is down.

**Done when:** a run's full event log can be reloaded and replayed.

### B5. App server + replay
`app/server.py` (FastAPI :8000):
- `GET  /` → `static/index.html`
- `GET  /api/policies` → from black-box `/admin/policies` (mock list when `USE_MOCKS=1`)
- `GET  /api/twists?bb=` → hidden labels only
- `POST /api/run` `{bb, twist_id, brain, live: true}` → sets the twist via admin, starts `agent.loop.run` in a background thread, returns `{run_id}`
- `WS   /ws/{run_id}` → streams events (thread → asyncio via a thread-safe queue)
- `POST /api/replay` `{run_id | cache_key, speed}` → re-emits stored events with original timing ÷ speed (backup demo + UI dev tool)
- `POST /api/reveal` `{run_id}` → admin reveal → emit `reveal`; highlight ledger rules whose text mentions the twist feature
- `POST /api/fix` `{run_id}` → FIX prompt → emit `fix`
- `POST /api/shield` `{run_id, on}` → run the demo seed locally with `shielded_action` per step (bb action from `/act`) → emit `shield`
- `GET  /api/results` → `results/*.json`
- **Live fallback:** if a live run hits `LIVE_TIMEOUT_S` before finishing, the UI shows "switching to cached run of this exact configuration" and replays `cache/<bb>__<twist>__<brain>/`. Never hide that it's cached.

**Done when:** `POST /api/replay` on a mock run animates the UI end to end.

### B6. UI (`app/static/`): split screen, vanilla JS, dark theme
```
┌──────────────────────────────────┬──────────────────────────────────────┐
│ BLACK BOX · the robot's net      │ JUDGE CONTROLS                       │
│                                  │ Policy [▼]  Flaw [Hidden flaw #2 ▼]  │
│          [clip]                  │ Brain [Muse ▼]  ▶ Run  Reveal  Fix   │
│                                  │ Shield ○   Queries 1,284  Match 87% ▲│
├──────────────────────────────────┼──────────────────────────────────────┤
│ CLONE · running recovered code   │ LEDGER (latest 8 rows, status icons) │
│          [clip]                  ├─────────────────────┬────────────────┤
│ red border = clone disagrees     │ RECOVERED CODE      │ MATCH vs       │
│ same seed · synced               │ (changed lines glow)│ QUERIES (SVG)  │
└──────────────────────────────────┴─────────────────────┴────────────────┘
  Footer: "Policies under study: SB3 RL Zoo (third-party). Everything else built Nov 7."
```
```css
body   { display:grid; grid-template-columns:1fr 1fr; height:100vh; margin:0; }
.left  { display:grid; grid-template-rows:1fr 1fr; }
.right { display:grid; grid-template-rows:auto 1fr 1.1fr; }
.right .bottom { display:grid; grid-template-columns:3fr 2fr; }
.sim img { width:100%; height:100%; object-fit:contain; }
```
- Clips: play both `frames` arrays in lockstep at `fps`; loop until a newer version arrives. Red border on the clone panel on frames where `disagree[t]`.
- Ledger icons: ✓ confirmed, ✗ refuted (strikethrough), ? inferred, ◎ observed, 🔍 probing. Highlighted rows on reveal.
- Code panel: monospace ≥ 16 px; diff against the previous version and glow changed lines for 3 s.
- Curve: plain SVG polyline of `metric` events (queries on x, agreement on y, 0–100%).
- Status line under the controls shows the latest `status.text`.
- **SubQ panel (from `snap` events):** a one-line strip under the ledger header: "SubQ Snap read 4,812 raw rows (212K tokens) · anomaly most likely in **x** (p 0.91) · probe next: **x**", plus a small horizontal bar per feature for `feature_relevance`. Each ledger row shows a small `SubQ p=0.88` chip from `rule_support`, visually distinct from the Python-verdict icon.
- Reveal swaps the flaw dropdown text to the reveal label. Shield toggles the black-box clip to the shielded clip and relabels the panel "BLACK BOX + SHIELD (fixed rule)".
- Results tab: table from `/api/results` with a **with SubQ vs without SubQ** comparison (queries and turns to find the twist), cross-match matrix, baseline row, and one honest failure row.
- Sizing: works at 1920×1080 with no scrolling; ledger ≥ 18 px.

**Done when:** replaying a mock run shows every panel moving correctly; the screenshot at 1920×1080 has no scrollbars.

### B7 (after merge). Fix + shield + results tab wired to real data
**Done when:** on bb-05 + a LunarLander twist: Reveal → Fix → Shield on shows the flaw gone on the same seed, with everything else visibly unchanged.

---

## 8. Integration checklist (1:15–1:45, both)

1. `USE_MOCKS=0`, start both servers, `python -m scripts.smoke` all green.
2. From the UI: bb-01 + t-cp-1 + Muse → run finishes; twist found in the ledger.
3. `snap` events appear in that run and SubQ calls show up in `brain_calls.jsonl`; repeat headless with `SNAP=0` to confirm the ablation path works.
4. `pytest -q` passes, including `test_isolation.py`.
5. Then tune the **live path** on bb-05 (LunarLander) so a live run finishes in ≤ `LIVE_TIMEOUT_S`: lower `LIVE_MAX_TURNS`, fit `max_seconds`, prober `max_queries`.

---

## 9. Sponsor usage (what each one is visibly doing)

| Sponsor | Where | Evidence for judges |
|---|---|---|
| **Meta Model API (Muse Spark)** | `BRAIN=muse`: proposes program structure, designs probes, revises from counterexamples, writes ledger + fixes. **Default brain for the live demo.** | Live run + `brain_calls.jsonl` + results-tab column |
| **SubQ Snap 1.1** (Decisions API) | Evidence scanner every turn: reads the **entire raw query log** (up to 384K tokens, no summarizing) and returns calibrated probabilities for which features matter, where the anomaly is, what to probe next, and how well each rule fits. Steers Muse. | `snap` panel live in the UI + ablation in results: queries/turns to find the twist **with vs without SubQ** (`SNAP=1` vs `SNAP=0`) |
| **MongoDB Atlas** (optional) | `STORE=mongo`: queries, ledger, program versions, evals | Collections populated by a demo run |

Division of labor: **SubQ Snap reads everything and decides where to look; Muse writes and revises the code; Python decides what's true.** The SubQ ablation (same runs with `SNAP=0`) is the headline evidence for the SubQ prize.

Prizes this maps to: Overall · Best Use of Meta Model API ($5K) · Best Use of Subquadratic ($5K 1st / $1K 2nd). There is no separate MongoDB prize; Atlas is optional.

---

## 10. README must include
- One-liner + 3-sentence explanation + screenshot/GIF
- How to run (section 2 commands)
- **Built during the hackathon:** every directory above
- **Third-party:** SB3 RL Zoo checkpoints (link each HF repo), Gymnasium envs
- Sponsor usage table (section 9)
- Results table (copy from `results/summary.json`)
