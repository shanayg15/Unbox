# Unbox

Unbox turns a black-box robot policy (a neural net behind an API) into short readable Python code that behaves the same, with a ledger of evidence for every rule. Then it finds the rule behind a hidden flaw, fixes it, and verifies nothing else changed.

## Overview

A trained controller sits behind an API: Unbox can only send it sensor readings and see which action it picks. An LLM (Muse Spark) writes a small `policy(obs, P)` program and a ledger of plain-English rules; an optimizer fits every number in it, Python probes confirm or refute each rule, and SubQ Snap scans the entire raw query log every turn to say where to look next. When engineers report a flaw ("over-reacts when moving fast"), the matching rule is already in the ledger. Unbox makes the smallest code change that removes it, then shields the robot's own output with that fix and shows that nothing else changed.

<!-- TODO (demo polish): screenshot or GIF of the live run -->

## How to run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install swig                     # Box2D (LunarLander) needs swig first
pip install -r requirements.txt
cp .env.example .env                 # fill in keys
```

```bash
uvicorn blackbox.server:app --port 8001          # the black box (policies + hidden flaws)
uvicorn app.server:app --port 8000               # the UI → open http://localhost:8000
python -m scripts.smoke                          # checks keys, black box, numerics
python -m agent.loop --bb bb-01 --twist t-cp-1 --brain muse   # headless run
python -m scripts.batch --snap 1,0               # results + caches, with/without SubQ
pytest -q
```

### UI without any agent (fixtures and replay)

The UI is built against `contracts/events.py` only, so it runs with no brain, no black box and no network:

```bash
python -m scripts.make_fixture                   # → fixtures/demo_events.jsonl (real CartPole frames)
uvicorn app.server:app --port 8000
open "http://localhost:8000/?replay=demo_events&speed=2"
```

- `?replay=demo_events`, `?replay=run:<run_id>`, `?replay=cache:<key>` replay a stored event log with its original timing (`&speed=` divides it, `0` = instant). Reveal / Fix / Shield stay on the buttons, like a live run.
- `?results=fixture&tab=results` shows the Results tab with clearly-labelled fake numbers from `fixtures/results/`.
- If a live run cannot finish (agent not built, error, or over `LIVE_TIMEOUT_S`), the UI says so in a banner and replays `cache/<bb>__<twist>__<brain>/`. It never hides that a run is cached.

## Built during the hackathon (Nov 7, 2026)

| Path | What |
|---|---|
| `contracts/` | Shared types, numerics API, event schema, mocks |
| `blackbox/` | Policy server: SB3 checkpoints behind `/act`, `/episode`, plus hidden flaws and admin endpoints |
| `numerics/` | Sandbox, parameter fitting (CMA-ES), pattern hints, counterexample prober, evaluator, shield |
| `agent/` | Brain client, SubQ Snap evidence scan, prompts, probes, ledger, run loop, fixer |
| `store/` | JSON + MongoDB Atlas event store |
| `app/` | FastAPI server, WebSocket bridge, replay, plain HTML/JS UI |
| `scripts/` | Smoke test, batch runner, UI fixture generator |
| `fixtures/`, `tests/` | UI fixtures, test suite |

## Third-party

- Policies under study: [Stable-Baselines3 RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) checkpoints on Hugging Face: [sb3/ppo-CartPole-v1](https://huggingface.co/sb3/ppo-CartPole-v1), [sb3/dqn-CartPole-v1](https://huggingface.co/sb3/dqn-CartPole-v1), [sb3/ppo-LunarLander-v2](https://huggingface.co/sb3/ppo-LunarLander-v2), [sb3/dqn-LunarLander-v2](https://huggingface.co/sb3/dqn-LunarLander-v2), [sb3/dqn-MountainCar-v0](https://huggingface.co/sb3/dqn-MountainCar-v0), [sb3/ppo-Acrobot-v1](https://huggingface.co/sb3/ppo-Acrobot-v1)
- Environments: [Gymnasium](https://gymnasium.farama.org/) (CartPole, LunarLander, MountainCar, Acrobot)

## Sponsor usage

| Sponsor | Where | Evidence for judges |
|---|---|---|
| **Meta Model API (Muse Spark)** | Proposes program structure, designs probes, revises from counterexamples, writes ledger and fixes. Default brain for the live demo. | Live run, `runs/<id>/brain_calls.jsonl`, results-tab column |
| **SubQ Snap 1.1** (Decisions API) | Evidence scanner every turn: reads the entire raw query log (up to 384K tokens, no summarizing) and returns calibrated probabilities for which features matter, where the anomaly is, what to probe next, and how well each rule fits. Shown live in the UI (SubQ strip, relevance bars, `SubQ p=` chips). | SubQ panel in the UI, plus the ablation: queries/turns to find the flaw with vs without SubQ (`SNAP=1` vs `SNAP=0`) |
| **MongoDB Atlas** (optional) | `STORE=mongo`: queries, ledger, program versions, evals | Collections populated by a demo run |

SubQ Snap reads everything and decides where to look; Muse writes and revises the code; Python decides what's true.

## Results

<!-- Filled from results/summary.json after the A6 batch run (scripts/batch.py). -->
Pending: the batch run (`python -m scripts.batch --snap 1,0`) writes `results/summary.json`, `results/crossmatch.json` and `results/baselines.json`; the table goes here and the same data shows in the UI's Results tab.
