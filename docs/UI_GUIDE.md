# Unbox UI guide

Unbox watches a robot's neural network through an API and writes short, readable Python that makes the same decisions. It keeps a ledger of the rules it believes, proves each one with experiments, finds the rule behind a planted flaw, fixes it in the copy, and guards the real network with the fix. The network's weights are never changed.

This demo covers two robots: **CartPole** (balance a pole on a moving cart; PPO and DQN versions, planted flaws 1 and 2) and **MountainCar** (drive an underpowered car up a hill; DQN, planted flaw 1 "Freezes in one corner of its workspace", start state #28). The hero is **CartPole, PPO with planted flaw 2** ("Drifts in one zone of the track", start state #152).

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

Terminal 3, keep the Mac awake during the demo:

```bash
caffeinate -d
```

Open http://localhost:8000 in the browser, full screen. Optional check before going on stage (all rows should be green):

```bash
cd ~/Unbox && source .venv/bin/activate && python -m scripts.smoke
```

Settings live in `.env`: `STORE=json`, `SNAP=1`, `LIVE_MAX_TURNS=4`, `LIVE_TIMEOUT_S=75`, `DEMO_SEED_BB_01=152`, `DEMO_SEED_BB_03=28`. Leave `STORE=json` for the demo: Atlas works, but its writes slowed the live run from 13 s to 56 s.

---

## What is on the screen

### Left column (always visible)

- **The robot (real network)**: the real neural network driving the robot from one fixed start state (CartPole: start state #152). With the guard on, the title becomes **The robot (real network) + guard**.
- **Our copy (readable code)**: our recovered Python program driving the same robot from the same start state. It says "Waiting for the first copy…" until version 1 exists.
- **Red outline on Our copy**: at the exact moment shown, the real network would have chosen a different action than our copy did. The flag is computed for every frame we show, so it lines up frame by frame.
- **"Episode ended (crashed)" / "(reached goal)" / "Ran the full episode"**: shown when that panel's run is over; the last frame is held.
- **Replay clip**: plays both panels again from the start, in sync. Clips play once when they arrive and then hold the last frame; they never loop.
- The top-right of each panel shows the start state and, for the copy, the version and on how many frames it differs from the robot.

### Right column: five numbered sections (the nav at the top jumps to each)

**1 Setup**
- **Robot**: which network to study, e.g. "CartPole, PPO: balances a pole on a moving cart". PPO and DQN are two different training algorithms.
- **Flaw**: "No planted flaw", or "Planted flaw N (secret until Reveal)". A planted flaw is a hidden rule we added to the network's output on the server. Unbox never sees which one is active.
- **Brain: Muse Spark · Evidence scanner: SubQ Snap 1.1**: Muse writes and revises the code and designs experiments; SubQ reads every raw query row each turn and says where to look; plain Python decides what is true (match numbers, confirmed/ruled out).
- **Run**: starts a live run. The status line under it narrates what is happening.
- **Backup**: a menu of our earlier real runs, one per robot and flaw. Picking one replays it (at most 30 s) with a banner saying it is an earlier run. Reveal, Fix and Guard still work on it.
- The badge next to Backup says **Live run · running / finished**, **Earlier run** (automatic fallback) or **Backup replay**.

**2 Learning**
- **Queries asked**: how many situations we have shown the robot so far.
- **Our copy matches the robot on X% of new situations**: share of held-out situations (8 episodes the copy never trained on) where our copy picks exactly the robot's action. **This is the one match number**: the same value is in the code card header, on the chart, and in the "Finished" line. It always belongs to the best version so far.
- **Chart**: best match so far (it never drops) against queries asked. The ringed point marked "final" is the result.
- **SubQ line**: "SubQ read N raw rows (K tokens) · most likely cause: feature". The bars are SubQ's probabilities for which sensor best separates the cases where our copy is wrong.

**3 Finding**
- **Finding card**: before Reveal it shows the **Top suspect**, the proven rule most likely to be the flaw. **Reveal the planted flaw** asks the black box what was planted and shows two columns: **Reported flaw** (the plain-English label) and **Unbox found** (our proven rule), plus a **Match** or **Not found** badge. Match means a proven rule names the flaw's sensor with a threshold within 20% of the true one (the same test as the batch results).
- **Our copy · vN · L lines · X%**: the recovered program. Lines that changed in the newest version glow for 3 seconds. Under it are the fitted parameter values (e.g. `thr_x_left = -0.080`). Muse names the parameters; an optimizer sets every number.
- **Ledger**: the rules Unbox believes. **Confirmed**: an experiment proved it (85% or more of probe points agree). **Testing**: seen in data, a hypothesis, or being tested now. **Ruled out (n)**: click to expand; experiments contradicted these. On Reveal, the flaw's rule is highlighted.

**4 Fix**
- **Fix our copy**: Muse makes the smallest edit to the copy that removes the reported behavior. The diff shows exactly which lines changed (CartPole flaw 2: two lines removed).
- **Guard the real network**: the real network keeps driving; at each step, if the flawed copy and the fixed copy disagree, the guard uses the fixed copy's action, otherwise it passes the robot's own action through. The robot panel replays the same start state with the guard on.
- **"Outside the flaw zone, the guarded robot acts exactly like the original on X% of held-out situations"**: proof the guard changes nothing else.
- **"Reward on the same 5 test starts: robot alone A → with guard B · success n/5 → m/5"**: what the flaw costs, measured live.

**5 Report: what your robotics team gets** (the product)
- **Failing decision**: the flaw as one sentence, with its line in our copy.
- **Where and how often**: the region in sensor terms (e.g. `cart_x < -0.08`), the share of the robot's own steps inside it, and how many of its 8 rollout runs enter it.
- **What it costs**: robot vs guarded robot reward (live runs, guard on).
- **Evidence**: probe queries that confirmed it, which feature SubQ flagged first, rules proven vs ruled out.
- **Post-training action**: "Retrain on situations where …; correct action there: …". **Download targeted training set (CSV)**: situations inside the region, labeled with the fixed copy's action.
- **Regression tests**: the other proven rules that must not change after retraining. **Download regression tests (JSON)**.
- **Guard check**: the same line as in section 4.
- **Download full report (Markdown)**: all of the above plus the code and ledger.
- For a run with no planted flaw: "No failing decision found. Readable copy matches the robot on X%", plus the regression tests and report.

### Results tab
- **Key results**: large numbers computed from `results/*.json` (never typed in): flaws found with and without SubQ, median match, how many fewer queries SubQ needed, speed-up of the copies, own robot vs sibling robot, and Muse alone without asking the robot.
- **SubQ on vs off: queries to find the flaw** (with a totals row), **Summary per run** (a red row is the honest failure: MountainCar flaw 1 without SubQ), **Cross-match**, **Zero-query baseline**.

---

## The 3-minute demo click path

| Time | Click | Say |
|---|---|---|
| 0:00 | Page open, Live tab, Robot = **CartPole, PPO** | "This robot is a neural network behind an API. We can only ask it what it would do." |
| 0:15 | Flaw = **No planted flaw** → **Run** (about 15 s) | "Unbox watches it, writes code, tests it. Seven lines of Python, 99.9% match on situations it never saw." |
| 0:40 | Flaw = **Planted flaw 2** → **Run** (about 60 s) | "Now the same robot with a secret flaw. Watch it fall at about one second. SubQ reads all the raw evidence and points at cart_x; Muse writes the rule; Python proves it with experiments." Point at the Finding card: **Top suspect: cart_x < -0.08 → push_right**. Point at our copy falling at the same moment (no red outline). |
| 1:45 | **Reveal the planted flaw** | "Engineers' report: drifts in one zone of the track. Unbox found it: **Match**, line 6 of the copy." |
| 2:00 | **Fix our copy** (about 25 s) | "Muse removes two lines. Nothing else changes." Scroll to the Report while it works. |
| 2:25 | **Guard the real network** | "Same start state, the real network plus the fix as a guard: it now balances the whole episode. 100% identical everywhere outside the flaw zone. Reward 178 → 500, success 1 of 5 → 5 of 5." |
| 2:40 | Scroll to **5 Report** | "This is what a robotics lab buys: the exact failing decision, where it fires, what it costs, the training set to fix it, and the regression tests." |
| 2:55 | **Results** tab | Read the Key results strip. |

If you are short on time, skip the no-flaw run (0:15).

---

## If something fails on stage

Expect this one: Muse takes 15 to 25 s per turn, so about half of live runs cannot converge (97% match) inside 75 s. The app then switches by itself to our earlier run of the same robot and flaw, says so in a banner, and replays it in at most 20 s. Everything after that (Reveal, Fix, Guard, Report, downloads, the reward and success numbers) is computed live on that run. Say: "Live runs are capped at 75 seconds; this is our earlier run of this exact configuration."

- **The run goes past 75 s**: the app switches automatically to our earlier run of the same robot and flaw, with the banner "The live run could not finish within the time limit. Showing our earlier run of this exact configuration." Say so and keep going; Reveal, Fix and Guard still work.
- **Muse or the internet is down** (status shows an error, or nothing moves for 30 s): open **Backup** → **CartPole, PPO · planted flaw 2**. Reveal and Guard need only the local black box. **Fix needs Muse**; if it fails, show the Report card and the Results tab instead.
- **The page looks stuck or you reloaded it**: reload again. The address bar keeps `?run=…`, and the page rebuilds the whole run from the server.
- **Black box error** ("could not set the hidden flaw" or "black box unreachable"): restart Terminal 1 (command above), then press Run again.
- **App error**: restart Terminal 2 and reload the page.
- **Two runs on the same robot at once** confuse the planted flaw (it is one global setting per robot). Run one at a time.
- **Guard shows no cost line on a Backup replay**: expected. The cost is measured on live runs only.
