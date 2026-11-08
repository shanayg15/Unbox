# Unbox UI guide

Unbox watches a robot's neural network through an API and writes short, readable Python that makes the same decisions. It proves each rule with experiments, finds the rule behind a failure, fixes it, and applies the fix to the deployed network as a runtime guard. The network's weights are never changed.

Robots in this demo: **CartPole, PPO** (balance a pole on a moving cart; start state #152) and **MountainCar, DQN** (drive an underpowered car up a hill; start state #13; planted flaw: position < -0.5 → noop). Hero: **CartPole, PPO · Planted flaw 2** (reported issue: "Drifts in one zone of the track").

---

## Start everything from a fresh terminal

Terminal 1, the black box (the robots behind an API):

```bash
cd ~/Unbox && source .venv/bin/activate && uvicorn blackbox.server:app --host localhost --port 8001
```

Terminal 2, the app:

```bash
cd ~/Unbox && source .venv/bin/activate && uvicorn app.server:app --host localhost --port 8000
```

Terminal 3, keep the Mac awake:

```bash
caffeinate -d
```

Open http://localhost:8000, full screen. Optional pre-flight check (all green):

```bash
cd ~/Unbox && source .venv/bin/activate && python -m scripts.smoke
```

`.env` settings used: `STORE=json`, `SNAP=1`, `LIVE_MAX_TURNS=4`, `LIVE_TIMEOUT_S=75`, `DEMO_SEED_BB_01=152`, `DEMO_SEED_BB_03=13`. Keep `STORE=json`: Atlas works but slowed a live run from 13 s to 56 s.

---

## The screen

### Left: two simulations, always playing
- **Deployed policy**: the real neural network driving the robot. Pill: Running, Original or With fix. After the fix the title reads **Deployed policy · fix applied** and the border turns green.
- **Recovered policy**: Unbox's readable Python driving the same robot from the same start state, in sync. Pill: version and match, e.g. "v2 · 100.0% match".
- Both clips loop. At the end of each loop the last frame holds for about 2 s with the outcome: "Failed: the pole fell", "Balanced for the full episode", "Reached the goal", "Did not reach the goal in time".
- **Red outline** on the recovered policy: at that exact frame the network would have chosen a different action.

### Right: four steps, each with a status pill
1. **Configure**: Robot policy, Scenario ("Normal operation" or "Planted flaw N"; planted flaws are hidden from Unbox), **Run analysis**. Run is disabled while a run is in progress (only one run per robot at a time). **Earlier runs** replays a stored real run of any configuration, in at most 20 s.
2. **Analysis**
   - **Match on unseen situations**: share of states from 8 episodes the copy never trained on where it picks exactly the network's action. The same number appears in the pill, the chart and the report.
   - **Network decisions analysed**: every network decision Unbox has seen (its own runs plus experiments).
   - **Recovered version** and **Elapsed**.
   - **Activity**: the last seven things Unbox did (Muse thinking, probes, fitting, SubQ scans).
   - **Agreement with the network**: best match so far against experiments run.
   - **Evidence scan · SubQ Snap**: SubQ reads every raw row each turn and names the sensor that best explains where the copy disagrees.
   - **Recovered policy**: the code; new lines glow for 3 s; fitted parameter values below.
   - **Rule ledger**: Confirmed / Under test / Rejected. Status comes from probe experiments, never from the language model.
3. **Remediation**
   - Normal operation: "No failure detected … No remediation is needed." Pill: Not needed.
   - Planted flaw: **Suspected failure** (found by Unbox before anyone reports it) and one button, **Apply fix**. It runs three steps: compare with the reported issue → Muse edits the recovered policy → verify on the deployed policy. Then it shows **Reported issue** next to **Unbox diagnosis** with a **Diagnosis confirmed** badge, the code diff, and two checks: **No side effects** (same action as the original network on X% of held-out situations outside the flaw zone) and **Outcome** (reward and success on 5 test starts, without vs with the fix). **Original / With fix** switches the top simulation to compare.
4. **Diagnostic report**: Summary, Failure mode (with code line), Trigger region and frequency, Operational impact, Evidence, Recommended retraining, Regression suite, Runtime guard verification. Downloads: full report (.md), targeted training set (.csv), regression tests (.json).

### Results tab
An abstract and five numbered tables computed from `results/*.json`: headline measures with method and n; SubQ on vs off queries to detection (with totals); all runs (red rows would be missed detections); cross-policy agreement; zero-query baseline.

---

## 3-minute demo click path

| Time | Click | Say |
|---|---|---|
| 0:00 | CartPole, PPO · Normal operation · **Run analysis** (about 15 s) | "This robot is a neural network behind an API. Unbox only asks it what it would do. In 15 seconds: a ten-line program that matches it 99.9%. No failure, no fix needed." |
| 0:30 | Scenario **Planted flaw 2** · **Run analysis** (45 to 70 s) | "Now the same robot with a hidden flaw. Watch the top: it fails. SubQ reads every raw decision and points at the cart sensors; Muse writes the rules; Python proves them with experiments. The copy fails the same way." |
| 1:40 | Point at **Remediation → Suspected failure** | "Unbox already found it: when the cart is left of −0.08 the network always pushes right." |
| 1:50 | **Apply fix** (about 15 s) | "Engineers report the robot drifts in one zone. Diagnosis confirmed. Muse removes two lines. The fix runs as a guard around the real network: it now balances the whole episode, reward 178 to 500, success 1 of 5 to 5 of 5, and 100% identical everywhere else." |
| 2:30 | Scroll to **Diagnostic report** | "This is what a robotics team receives: the failing decision, where it fires, what it costs, a targeted training set and the regression tests." |
| 2:50 | **Results** tab | Read the abstract. |

---

## Known limits (be ready for the question)

- **One-to-one copies:** CartPole PPO flaws 1 and 2: the copy fails at exactly the robot's step (0 differing frames). MountainCar flaw 1 from start state #13: the robot gets stuck in the left corner and never reaches the goal, and its copy gets stuck the same way; after Apply fix the real robot reaches the goal at step 102 and the fixed copy at step 109.
- **Fix results** (5 test starts, live black box): CartPole flaw 1 reward 345 → 500, success 3/5 → 5/5; CartPole flaw 2 reward 178 → 500, 1/5 → 5/5; MountainCar flaw 1 reward −200 → −132, reached the goal 0/5 → 4/5.

## If something fails on stage

- **Expected:** Muse takes 15 to 25 s per turn, so some live runs cannot reach 97% match inside 75 s. The app then switches by itself to our earlier run of the same robot and scenario, shows the banner "The live run could not finish within the time limit. Showing our earlier run of this exact configuration.", and replays it in at most 20 s. Apply fix and the report still run live on that data. Say: "Live runs are capped at 75 seconds; this is our earlier run of this exact configuration."
- **Muse or the internet is down:** Earlier runs → CartPole, PPO · planted flaw 2. Apply fix needs Muse; if it fails, show the report and the Results tab.
- **"A run on this robot is still in progress"**: wait for it to finish (at most about 80 s).
- **Page stuck or reloaded:** reload again; `?run=…` in the address bar rebuilds the run.
- **Black box or app error:** restart that terminal, reload the page.
