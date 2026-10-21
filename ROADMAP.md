# ROADMAP — ruvector-vfa

Task list for the unattended `/ship-next` loop. Run it with:

```bash
npx @shaal/shipyard --tasks ROADMAP.md     # drain the whole list, one merged PR per task
npx @shaal/shipyard 1 --tasks ROADMAP.md   # ship exactly one task
# or, inside Claude Code: /ship-next ROADMAP.md
```

## Rules for the loop (read before every task)

- Do the **first unchecked task** in document order. Never skip ahead. Each task lists `depends:`.
- Only top-level task lines carry a checkbox. Acceptance items are plain bullets. Do not add checkboxes inside a task.
- Each task fits in one headless turn: no background jobs, and every long run stays inside its stated budget.
- One shell command runs for at most 10 minutes. Chunk anything longer and make it resumable. Refer to `CLAUDE.md` → *Runtime limits*.
- Never depend on files a previous task left in `runs/`. Refer to `CLAUDE.md` → *Data handoffs*.
- "Verify" names the exact commands the confidence gate must run and quote. Refer to `CLAUDE.md` → *Verification probes*.
- Numeric targets: if a target misses after honest work, follow `CLAUDE.md` → *Missed numeric targets*. Never tune the frozen benchmark.
- Requirements source: `docs/BRIEF.md` (the original brief) and `docs/ARCHITECTURE.md` (design decisions).

Phases: **A** simulation foundation · **B** System 1 (env, skill, safety, baseline) · **C** RuVector-style memory · **D** dual-process cognition · **E** demo, experiments, docs · **F** stretch (residual learning, real RuVector Rust backend).

---

## Phase A — Simulation foundation

- [x] **T01 — Project scaffold (uv, package layout, lint, tests, CI)**
  - Goal: an empty but runnable Python project that every later task builds on.
  - Deliverables:
    - `pyproject.toml` (hatchling): name `ruvector-vfa`, import package `vfa` under `src/vfa/`, `requires-python = ">=3.11,<3.14"`.
    - Runtime deps: `mujoco>=3.3,<4`, `gymnasium>=1.0`, `numpy>=1.26`, `matplotlib>=3.8`. Extras: `dev` = `pytest`, `ruff`; `video` = `imageio[ffmpeg]`.
    - `[project.scripts] vfa = "vfa.cli:main"` with a stub CLI that prints the version and the subcommand list.
    - Empty subpackages with docstrings: `sim`, `control`, `skills`, `memory`, `cognition`, `experiments`, `viz`.
    - `requirements.txt` exported from the lock (`uv export --no-hashes`) for non-uv users.
    - pytest config: markers `slow` and `render`; ruff config (line length 100, rules E,F,I,UP,B).
    - `tests/test_smoke.py`: imports `vfa`, `mujoco`, `gymnasium`; builds a trivial MJCF; steps it 100 times.
    - `.github/workflows/ci.yml`: ubuntu, Python 3.12, uv, `ruff check`, `pytest -m "not slow and not render"`.
    - Commit `uv.lock`.
  - Acceptance:
    - `uv sync` succeeds from a clean clone on this Mac.
    - `uv run vfa --help` prints usage.
    - Fast tests pass and ruff is clean.
  - Verify: `uv sync && uv run vfa --help && uv run pytest -m "not slow" -q && uv run ruff check .`
  - Budget: ~20 min. Tests < 10 s.
  - depends: none

- [x] **T02 — Panda + table + cube scene (MjSpec builder, torque actuators, wrist F/T)**
  - Goal: a MuJoCo scene with force sensing, built in code from a pristine vendored robot model.
  - Deliverables:
    - Vendor the pristine MuJoCo Menagerie Panda at commit `feadf76d42f8a2162426f7d226a3b539556b3bf5` into `assets/menagerie/franka_emika_panda/` (`panda.xml`, `hand.xml`, `assets/`, `LICENSE`). Source: `~/.cache/robot_descriptions/mujoco_menagerie` if present, else a sparse clone of `github.com/google-deepmind/mujoco_menagerie` at that SHA. Add `assets/menagerie/NOTICE.md` (Apache-2.0, upstream URL, SHA). Do not edit the vendored XML.
    - `vfa/sim/scene.py`: `build_scene(cfg) -> mujoco.MjModel` using `mujoco.MjSpec`:
      - Replace the 7 arm position servos with torque `motor` actuators (limits 87,87,87,87,12,12,12 N·m).
      - Gripper: keep the menagerie tendon actuator for now. T05 decides whether it becomes a force motor.
      - Add a table, a free-joint cube (default 5 cm, 0.1 kg), a place-target site, and an `ee` site at the fingertip center.
      - Add a wrist site between `link7` and `hand` with `force` and `torque` sensors. Add finger-touch sensors or contact helpers.
      - Expose hooks to set cube mass, friction, size, and contact softness (`solref`) per episode.
    - `vfa/sim/scene_check.py` CLI (`uv run python -m vfa.sim.scene_check --png runs/scene.png`): builds the scene, holds the `home` pose with gravity compensation for 2 s, prints the F/T reading, renders a PNG offscreen.
    - Tests: model compiles; 7 arm torque actuators + gripper actuator exist; at rest the F/T force magnitude ≈ hand+fingers weight (±15 %); gravity-compensated hold drifts < 1 cm in 2 s.
  - Acceptance:
    - All tests pass. The PNG shows arm, table, and cube (Read the PNG and describe it).
  - Verify: `uv run pytest -q tests/test_scene.py && uv run python -m vfa.sim.scene_check --png runs/scene.png`
  - Budget: ~45 min. Tests < 15 s.
  - depends: T01

- [x] **T03 — Operational-space / impedance controller + gripper interface**
  - Goal: System 1's low-level controller — compliant Cartesian control with torque output.
  - Deliverables:
    - `vfa/control/osc.py`: `OperationalSpaceController` — task-space impedance (position + orientation) with configurable stiffness/damping, gravity and Coriolis compensation (`qfrc_bias`), nullspace posture term toward `home`, torque clipping to actuator limits. Uses `mj_jacSite` and the mass matrix from `mj_fullM`.
    - `vfa/control/gripper.py`: `Gripper` with `open()`, `close(width)`, `squeeze(force_n)`, and `width()` readback. Document how a force request maps onto the current actuator.
    - Fixed top-down end-effector orientation by default (yaw optional).
  - Acceptance:
    - Reach test: 10 random targets inside the workspace box, final error < 1 cm and orientation error < 5° within 2 s of sim time.
    - Compliance test: a lowered stiffness gives a measurably larger deflection under the same external push (`xfrc_applied`).
    - Throughput: a 10 s simulated lift-length episode runs in ≤ 1 s wall time on one core (≥ 10× real time). Print the measured number. If Python per-step overhead is too high, run the OSC law at a lower `control_hz` (e.g. 100–250 Hz) and hold torques across physics substeps; pick the highest rate that meets the target and record it in `docs/ARCHITECTURE.md` §2.
  - Verify: `uv run pytest -q tests/test_osc.py`
  - Budget: ~45 min. Tests < 20 s.
  - depends: T02

- [x] **T04 — Object property model (hidden classes, break rule, randomization)**
  - Goal: situations where the *right* force depends on properties the camera cannot see.
  - Deliverables:
    - `vfa/sim/objects.py`: `ObjectClass` and `ObjectSpec` dataclasses; `sample_object(cls, rng) -> ObjectSpec`. Start with the five classes in `docs/ARCHITECTURE.md` §4 (`light_rigid`, `heavy_rigid`, `slippery`, `fragile_soft`, `heavy_slippery`). Ranges live in `configs/objects.toml`.
    - Hidden properties: mass, friction, contact softness, break-force threshold. Visible properties: size (overlapping across classes) and pose. All cubes render the same color, so vision cannot identify the class.
    - Break rule: if the summed finger-to-object normal force exceeds the break threshold, mark the object `broken` (color it red) and end the attempt as a failure.
    - Optional perception noise on the observed cube pose (σ in config, default 0 for benchmark v1).
    - `apply_object(model, data, spec)` that writes the properties through the T02 hooks.
    - Cube geom `priority=1` and `condim=4` (ARCHITECTURE D8). MuJoCo otherwise takes the max friction of the two geoms, and the finger pads would hide the cube's friction.
  - Acceptance:
    - Tests: changing the cube friction changes the finger-contact friction (read `data.contact[i].friction` during a grasp);
      sampling is deterministic per seed; applied mass/friction/solref read back from the model; the break rule fires when a test squeezes a fragile cube past its threshold and does not fire below it.
  - Verify: `uv run pytest -q tests/test_objects.py`
  - Budget: ~30 min. Tests < 15 s.
  - depends: T03

- [x] **T05 — Physics feasibility spike: does force actually decide success?**
  - Goal: prove the core premise *before* building memory. If low grip does not slip and high grip does not break, nothing downstream can show improvement.
  - Deliverables:
    - `scripts/spike_grip.py`: a scripted grasp (OSC approach → descend → close at a given speed to a given force → lift 10 cm → hold 1 s). Sweep grip force × close speed × the five classes (≥ 5 samples per cell).
    - Measure the real squeeze force the gripper produces versus the command. If the menagerie position servo cannot give a monotonic, repeatable squeeze force, replace it in `scene.py` with a force-controlled tendon `motor` (ctrl = squeeze force in N) and document the change.
    - Wrist F/T tare (subtract the hand load measured before contact) and a first slip detector (load drop on F/T z, or finger-object relative velocity). Compare the F/T detector against privileged ground truth (object z relative to the fingers).
    - `docs/FEASIBILITY.md`: success heat map per class (grip force × close speed), squeeze-force calibration plot, slip-detector precision/recall, and the final class ranges.
  - Acceptance:
    - Each class has a success window inside the actuator range, and **no single grip force succeeds on all five classes** (the windows conflict, e.g. `fragile_soft` vs `heavy_slippery`).
    - Fast closing breaks `fragile_soft` more often than slow closing at the same target force (the force *profile* matters, not only the final value).
    - The F/T slip detector reaches ≥ 0.9 precision and ≥ 0.8 recall against ground truth on the spike data.
    - If a criterion fails: adjust class ranges or the gripper actuator in this task (the benchmark is not frozen yet). Record what you changed and why.
  - Verify: `uv run python scripts/spike_grip.py --quick` (< 5 min), then Read the heat-map PNG and quote the table in `docs/FEASIBILITY.md`.
  - Budget: ~60 min. The full sweep uses `--workers` and finishes in ≤ 9 min per command.
  - depends: T04

## Phase B — System 1: environment, skill, safety, baseline

- [ ] **T06 — Sensors, events, and episode recorder**
  - Goal: everything the memory later embeds, logged at control rate.
  - Deliverables:
    - `vfa/sim/sensors.py`: tared and low-pass-filtered wrist wrench, finger contact flags and normal forces, gripper width, EE pose/velocity. Promote the T05 slip detector here.
    - Event detector: `CONTACT_ONSET`, `GRASP_CLOSED`, `SLIP`, `DROP`, `BREAK`, `LIFTED`, `PLACED`, `SAFETY_STOP`, with timestamps.
    - `vfa/sim/recorder.py`: `EpisodeRecorder` storing per-tick arrays (t, phase, EE pose/vel, wrench, finger forces, width, object pose — privileged), events, the skill parameters, and the outcome. Save as `.npz` plus a `.jsonl` summary line.
    - `vfa/viz/force_plot.py`: `plot_force_profile(episode, path)` with phase bands and event markers.
  - Acceptance:
    - Tests: recorder round-trips through save/load exactly; the event detector reports `CONTACT_ONSET` and `GRASP_CLOSED` on a scripted grasp, and `DROP` when a test opens the gripper mid-lift.
    - A force-profile PNG renders for one recorded episode (Read it).
  - Verify: `uv run pytest -q tests/test_sensors.py tests/test_recorder.py`
  - Budget: ~45 min. Tests < 20 s.
  - depends: T05

- [ ] **T07 — Gymnasium environment (reach, lift, pick-and-place)**
  - Goal: a clean, standard interface around the scene.
  - Deliverables:
    - `vfa/sim/env.py`: `PandaVFAEnv(task, control="osc"|"torque", object_class=None, render_mode=None|"rgb_array")`.
    - `osc` action: `Box(4)` = EE delta xyz (clipped) + normalized grip-force command. `torque` action: raw 7 joint torques + gripper.
    - Observation `Dict`: joint pos/vel, EE pose/vel, tared wrench (6), gripper width, finger contact flags, observed cube pose and size. **Hidden properties never appear in the observation**; they go in `info["privileged"]` for logging only.
    - Rewards, termination, and truncation per task, as in `docs/ARCHITECTURE.md` §3. `info` carries events and phase.
    - `render()` returns an RGB array through `mujoco.Renderer`.
    - Register `vfa/PandaReach-v0`, `vfa/PandaLift-v0`, `vfa/PandaPickPlace-v0`. The env owns an `EpisodeRecorder`.
  - Acceptance:
    - `gymnasium.utils.env_checker.check_env` passes for all three ids.
    - Same seed → identical observation sequence for 50 random-action steps.
    - A scripted OSC policy solves `PandaReach-v0` in ≥ 9 of 10 seeds.
  - Verify: `uv run pytest -q tests/test_env.py`
  - Budget: ~45 min. Tests < 30 s.
  - depends: T06

- [ ] **T08 — Parameterized grasp-and-lift skill with touch-and-heft probe (System 1)**
  - Goal: the fast reactive executor that System 2 will tune.
  - Deliverables:
    - `vfa/skills/params.py`: `SkillParams` (θ) — `grip_force`, `close_speed`, `lift_accel`, `grasp_depth`, `approach_speed` — with bounds, `to_vector()`/`from_vector()`, and a JSON form.
    - `vfa/skills/pick.py`: `GraspLiftSkill`, a phase state machine: `APPROACH → DESCEND → PROBE (gentle close + 1–2 cm heft) → SQUEEZE → LIFT → HOLD`. It drives the env's OSC action space at control rate.
    - After `PROBE` the skill emits a `PROBE_DONE` escalation with the probe force features and **waits for a θ update** through a callback. This is the mid-attempt System 1 → System 2 handoff.
    - `vfa/skills/reflexes.py`: slip reflex — on `SLIP`, raise the grip force by a bounded step, at most N times, then escalate `SLIP_UNRECOVERED`.
    - `run_attempt(env, skill, theta, on_escalation) -> AttemptResult` (success, failure mode, recorder episode).
  - Acceptance:
    - With θ hand-picked from `docs/FEASIBILITY.md`, the skill lifts each of the five classes in ≥ 8 of 10 seeds.
    - With a deliberately wrong θ it fails with the correct failure mode (`fragile_soft` + high force → `BREAK`; `heavy_slippery` + low force → `SLIP`/`DROP`).
    - The `PROBE_DONE` callback fires exactly once per attempt and can change θ for the remaining phases.
  - Verify: `uv run pytest -q tests/test_skill.py` (mark the 5×10 sweep `slow` and run it once with `-m slow`).
  - Budget: ~60 min. Slow sweep < 8 min.
  - depends: T07

- [ ] **T09 — Safety supervisor and emergency stop**
  - Goal: a hard override above both systems.
  - Deliverables:
    - `vfa/control/safety.py`: `SafetySupervisor` checked every control tick: wrist force/torque limits, EE workspace box, EE speed cap, joint-limit margin, sustained torque saturation.
    - On violation: switch to `Emergency` mode, hold the current pose under gravity compensation, freeze the gripper, end the attempt with `SAFETY_STOP` and a reason. Log on the `[SAFETY]` channel.
    - Integrate it into `run_attempt` and the env step.
  - Acceptance:
    - Fault-injection tests: driving the EE into the table, commanding a target outside the workspace, and applying a large external force each trigger a stop within 50 ms of sim time with the right reason.
    - Normal T08 attempts never trigger it (zero false positives across the T08 test seeds).
  - Verify: `uv run pytest -q tests/test_safety.py tests/test_skill.py`
  - Budget: ~40 min. Tests < 30 s.
  - depends: T08

- [ ] **T10 — Baselines, difficulty calibration, and benchmark freeze**
  - Goal: fix the yardstick *before* memory exists, so memory cannot be graded on a test tuned for it.
  - Deliverables:
    - Measure wall time per attempt first (median and p95, one core). Size the profiles from that number: `quick` (1 seed, 15 situations) must finish in < 3 min with workers, and every full-benchmark chunk (one condition × one seed) must finish in ≤ 9 min. Default target: 60 situations per seed (12 per class); shrink only if the measured throughput forces it. Record attempts/minute in `docs/RESULTS.md`.
    - `configs/benchmark.toml`: class mix, situations per seed, evaluation seeds (3) separate from calibration seeds, `max_attempts = 5`, and the `quick` and `calib` profiles.
    - `vfa/experiments/baselines.py`:
      - `naive` — one fixed θ, a sensible engineer's default stated in the docstring.
      - `best_fixed` — the single θ with the highest success on the **calibration** seeds (grid search, chunked with `--workers`). This is the strongest non-adaptive baseline. Save it to `configs/best_fixed.toml`.
      - `random_retry` — `naive` first, then random θ within bounds on each retry, same `max_attempts` as memory.
    - `vfa/experiments/benchmark.py`: chunked and resumable per `CLAUDE.md` → *Runtime limits* (`--condition`, `--profile`, `--seed`, `--workers`, `--save-episodes DIR`), plus an `aggregate` subcommand that writes `metrics.json` (success@1, success@k, mean attempts-to-success, break rate, drop rate, peak-force excess, force jerk).
    - `scripts/make_fixtures.py`: runs the `calib` profile with `naive` and `random_retry` and commits a compact episode set to `tests/fixtures/calib_episodes.npz` (≤ 5 MB, all five classes, successes and failures, hidden class labels included for test scoring only). T11 and T13 build on this fixture.
    - Write `configs/benchmark.sha256` (one hash over `configs/benchmark.toml` + `configs/objects.toml` + `configs/best_fixed.toml`) and `tests/test_benchmark_frozen.py` (the hash must match).
    - Create `docs/RESULTS.md` with the three baselines' measured numbers.
  - Acceptance:
    - `best_fixed` success@1 ≤ 75 % — the problem needs adaptation. If it is higher, widen the class conflict in `configs/objects.toml` in this task, re-measure, and record the change.
    - `random_retry` success@5 is reported. It is the bar for memory's attempts-to-success.
    - The frozen-hash test passes.
  - Verify: `uv run python -m vfa.experiments.benchmark --condition naive --profile quick`, then the full run for all three baselines (one command per condition, each ≤ 9 min with `--workers 8`), then `aggregate`, then `uv run pytest -q tests/test_benchmark_frozen.py`.
  - Budget: ~75 min.
  - depends: T09

## Phase C — RuVector-inspired memory

- [ ] **T11 — Force / trajectory / context embeddings**
  - Goal: turn an episode into vectors where "similar physics" means "nearby".
  - Deliverables:
    - `vfa/memory/embeddings.py`, per `docs/ARCHITECTURE.md` §5:
      - `force_features(episode)` — resample tared wrench and finger forces per phase to fixed lengths, plus key moments (contact onset time, peak force, closure-force slope = stiffness, heft load, slip count).
      - `trajectory_features(episode)` — EE path relative to the object, resampled, plus a velocity profile.
      - `context_features(obs0)` — observed cube pose and size, initial joint config.
      - Retrieval **keys** (built from what exists at query time): `key_pre` (context only), `key_probe` (context + probe force), `key_retry` (context + previous attempt's full force profile). Also `full_embedding` (everything + outcome).
    - Per-group standardization (mean/std fitted on `tests/fixtures/calib_episodes.npz`, saved to `configs/embedding_stats.npz` and committed), per-group weights, and L2 normalization of every vector.
  - Acceptance:
    - Tests: deterministic; all vectors have unit norm; fixed dimensions are documented and asserted.
    - Sanity on `tests/fixtures/calib_episodes.npz`: same-class `key_probe` pairs are more similar than cross-class pairs (mean cosine gap ≥ 0.1). Context-only keys show a smaller gap — print both numbers.
  - Verify: `uv run pytest -q tests/test_embeddings.py`
  - Budget: ~45 min. Tests < 30 s.
  - depends: T10

- [ ] **T12 — MemorySystem: working, episodic, semantic tiers + pluggable backend**
  - Goal: the vector memory, shaped so the real RuVector crate can replace it.
  - Deliverables:
    - `vfa/memory/backend.py`: `MemoryBackend` Protocol — `add(id, vector, payload)`, `search(vector, k, filter=None) -> list[Hit]`, `update(id, payload_patch)`, `save(path)`, `load(path)`. `NumpyBackend` implements cosine (dot on unit vectors) and Euclidean.
    - `vfa/memory/memory_system.py`, mirroring `ruvector-robotics::cognitive::memory_system`:
      - `WorkingMemory(max_size)` — bounded buffer for the current attempt; evicts the least-important item.
      - `EpisodicMemory` — every attempt (`key_*` vectors, full embedding, θ, outcome, failure mode, situation id); `recall_similar(key, k, success=None)`.
      - `SemanticMemory` — named concept vectors (skill prototypes from T13); `find_similar(query, k)`.
      - `MemorySystem` facade, plus **reinforcement**: each record has a `utility` score. Raise it when a retrieved record led to success, lower it when it led to failure. Retrieval ranks by `similarity × f(utility)`.
    - Persistence to `runs/<id>/memory/` and reload.
  - Acceptance:
    - Tests: top-k ordering equals brute force; filters work; utility updates change ranking as specified; save → load → identical search results; the Rust-parity rule holds (stored vectors are unit norm, so dot == cosine).
  - Verify: `uv run pytest -q tests/test_memory.py`
  - Budget: ~45 min. Tests < 15 s.
  - depends: T11

- [ ] **T13 — SkillLibrary: consolidate successes into skill prototypes**
  - Goal: semantic memory — compress many episodes into a few reusable skills.
  - Deliverables:
    - `vfa/memory/skill_learning.py`, mirroring `ruvector-robotics::cognitive::skill_learning`: `Demonstration`, `Skill`, `SkillLibrary` with `learn_from_demonstration(name, demos)` (average the EE trajectories and θ; `confidence = 1 − 1/(n+1)`), `execute_skill`, `improve_skill(name, feedback)` (clamped to [0, 1]).
    - `consolidate(memory)`: cluster successful episodes on `key_probe` (k-means in numpy, or greedy threshold clustering), learn one skill per cluster, store its centroid in `SemanticMemory`. Skill names are generated (e.g. `skill_03`). Clustering never reads class labels.
  - Acceptance:
    - Tests: averaging and confidence math matches the Rust semantics; on the successes in `tests/fixtures/calib_episodes.npz` the clusters reach purity ≥ 0.7 against hidden class labels (labels used only to score the test).
  - Verify: `uv run pytest -q tests/test_skill_learning.py`
  - Budget: ~40 min. Tests < 20 s.
  - depends: T12

## Phase D — Dual-process cognition

- [ ] **T14 — CognitiveCore: System 1 / System 2 loop with explicit handoffs**
  - Goal: make the dual-process structure visible in code and logs.
  - Deliverables:
    - `vfa/cognition/cognitive_core.py`, mirroring `ruvector-robotics::cognitive::cognitive_core`: `CognitiveMode {REACTIVE, DELIBERATIVE, EMERGENCY}`, `CognitiveState {IDLE, PERCEIVING, THINKING, ACTING, LEARNING}`, `Percept`, `Decision`, `Outcome`, and `CognitiveCore` with `perceive / think / act / learn`.
    - System 1 = skill + reflexes + OSC at control rate (REACTIVE). System 2 = memory query and θ selection (DELIBERATIVE) at three triggers: attempt start, `PROBE_DONE`, and an escalation or failure. The safety supervisor forces EMERGENCY.
    - A structured handoff log per attempt (`runs/<id>/handoffs.jsonl`): mode changes, trigger, retrieved ids, chosen θ, latency. Console lines use `[S1]`, `[S2]`, `[SAFETY]`, `[MEM]`.
    - System 2 here uses a stub policy (returns the default θ). T15 plugs in retrieval.
  - Acceptance:
    - Tests: a scripted attempt produces the exact mode sequence `DELIBERATIVE → REACTIVE → DELIBERATIVE(probe) → REACTIVE → LEARNING`; a slip escalation produces a System 2 entry; a safety fault produces `EMERGENCY` and ends the attempt.
    - System 2 never runs inside the control tick. Assert that S2 calls happen only at the three triggers.
  - Verify: `uv run pytest -q tests/test_cognitive_core.py`
  - Budget: ~45 min. Tests < 30 s.
  - depends: T13

- [ ] **T15 — Retrieval-augmented adaptation + retry-until-success**
  - Goal: the core loop of the brief — retrieve what worked, bias toward it, retry until success, store everything.
  - Deliverables:
    - `vfa/cognition/system2.py`: `RetrievalPolicy`:
      - Query `EpisodicMemory` with the key for the trigger (`key_pre`, `key_probe`, `key_retry`). Get k success neighbors and k failure neighbors.
      - Estimate a **feasible θ window** per dimension: similarity-weighted successes give the center; failures push bounds away (a `SLIP` at force f raises the lower bound above f; a `BREAK` at force f lowers the upper bound below f).
      - Exploration noise shrinks with neighbor similarity and skill confidence. Fall back to `SemanticMemory` skills when episodic hits are weak, and to the default θ when memory is empty.
    - `vfa/cognition/retry.py`: `solve_situation(env, core, max_attempts)` — reset the arm and object (same hidden properties) after each failure, store every attempt, reinforce the utility of the records that were used, and run consolidation every N situations.
    - The `memory_full` and `memory_ctx_only` conditions plug into the T10 benchmark runner (`ctx_only` uses only context features in every key — the force ablation).
  - Acceptance:
    - Synthetic test (no physics): on a 1-D toy problem with a hidden window, `RetrievalPolicy` reaches success in fewer mean attempts than random search over 200 trials.
    - Quick-profile physics run: `memory_full` success@1 on the last half of the situations is higher than on the first half (it learns), and higher than `naive` (print both).
  - Verify: `uv run pytest -q tests/test_system2.py tests/test_retry.py && uv run python -m vfa.experiments.benchmark --condition memory_full --profile quick`
  - Budget: ~60 min.
  - depends: T14

## Phase E — Demo, experiments, documentation

- [ ] **T16 — Demo CLI: baseline vs memory, visible improvement across retries**
  - Goal: the one command a newcomer runs.
  - Deliverables:
    - `uv run vfa demo [--quick] [--seed N] [--video]` in `vfa/cli.py` + `vfa/experiments/demo.py`:
      1. Baseline phase: `naive` and `random_retry` on the same situations.
      2. Memory phase: start with empty memory, run `memory_full` with retries.
      3. Print a per-situation table (class hidden until the end, attempts used, failure modes) and a summary table (success@1, attempts-to-success, break rate).
    - Logs show System 1/System 2 handoffs and retrieval hits (`[MEM] hit ep_0042 sim=0.93 outcome=success θ.grip=6.1N`).
    - Writes `runs/demo-<timestamp>/` with metrics, episodes, and memory.
  - Acceptance:
    - `vfa demo --quick` finishes in < 3 min and exits 0.
    - Its summary shows memory beating `naive` on success@1 over the second half of the situations. Quote the numbers.
  - Verify: `uv run vfa demo --quick`
  - Budget: ~40 min.
  - depends: T15

- [ ] **T17 — Visualization: force profiles, retrieval hits, learning curves, video, HTML report**
  - Goal: make "force is the language of manipulation" visible.
  - Deliverables:
    - `vfa/viz/plots.py`:
      - Force-profile overlay: failed attempt vs the retrieved successful neighbors vs the next attempt, with phases marked.
      - Retrieval-hit view: query vs top-k neighbors (similarity, outcome, θ).
      - Learning curves: rolling success@1 and attempts-to-success over situation index, per condition.
      - Embedding map: 2-D PCA of `key_probe` colored by hidden class (diagnostic only).
    - `vfa/viz/video.py` (extra `video`): offscreen MP4/GIF of one attempt with a force overlay.
    - `vfa/viz/report.py`: a self-contained `runs/<id>/report.html` (images inlined as base64).
    - `vfa demo` writes the report at the end.
  - Acceptance:
    - Running the quick demo produces all four plot types and `report.html`. Read each PNG and confirm it shows what its title claims.
  - Verify: `uv run vfa demo --quick` and list `runs/<latest>/`.
  - Budget: ~45 min.
  - depends: T16

- [ ] **T18 — Full experiments and ablations → docs/RESULTS.md**
  - Goal: honest, reproducible evidence that memory improves movement.
  - Deliverables:
    - `vfa/experiments/run_all.py`: all conditions (`naive`, `best_fixed`, `random_retry`, `memory_ctx_only`, `memory_full`) × 3 evaluation seeds on the frozen benchmark. Plus a **held-out** run: freeze memory after the warm-up seed, then evaluate on new situations with no further writes.
    - `docs/RESULTS.md`: a table with mean ± std across seeds, the learning-curve plots, force-profile examples, the force ablation, and a plain-language analysis that includes failure cases.
  - Acceptance (targets):
    - `memory_full` success@1 over situations 21–60 ≥ `best_fixed` success@1 + 15 percentage points (mean of 3 seeds).
    - `memory_full` mean attempts-to-success < `random_retry`.
    - `memory_full` mean peak-force excess or force jerk is lower than `random_retry` (smoother, gentler force).
    - Report the force ablation (`memory_full` vs `memory_ctx_only`) with its measured gap. It is not a gate.
    - On a miss, follow `CLAUDE.md` → *Missed numeric targets*.
  - Verify: `uv run python -m vfa.experiments.run_all --workers 8`, re-run until it reports every chunk done (each invocation ≤ 9 min, resumable), then `uv run python -m vfa.experiments.run_all aggregate`, then quote the `docs/RESULTS.md` table.
  - Budget: ~60 min + run time.
  - depends: T17

- [ ] **T19 — Pick-and-place extension with force-aware placing**
  - Goal: the third task from the brief, with a contact-rich place phase.
  - Deliverables:
    - Extend the skill: `TRANSPORT → PLACE_DESCEND → CONTACT (F/T unload detection) → RELEASE → RETREAT`. New θ dims: `place_speed`, `release_height`.
    - Place failure modes: `PLACE_BREAK` (fragile impact force), `TOPPLE` / `MISS` (outside the target tolerance).
    - Memory works on the pick-and-place keys (embeddings get place-phase force features). Add a `pick_place` benchmark profile **as a new file** `configs/benchmark_pick_place.toml` with its own hash. Do not edit the frozen lift benchmark.
    - Extend `vfa demo --task pick_place` and add a pick-and-place section to `docs/RESULTS.md`.
  - Acceptance:
    - Hand-picked θ places each class in ≥ 7 of 10 seeds.
    - `memory_full` beats `random_retry` on attempts-to-success on the pick-and-place quick profile. Quote the numbers.
  - Verify: `uv run pytest -q tests/test_pick_place.py && uv run vfa demo --quick --task pick_place`
  - Budget: ~60 min.
  - depends: T18

- [ ] **T20 — Final docs: README, architecture, RuVector mapping, extension guide**
  - Goal: a newcomer can install, run, understand, and extend the project in 15 minutes.
  - Deliverables:
    - `README.md`: what this is, a GIF/PNG, setup (`uv sync`; pip fallback `pip install -r requirements.txt`), `vfa demo`, how memory improves performance (with the measured numbers from `docs/RESULTS.md` only), macOS note about `mjpython` for the interactive viewer, and the project layout.
    - Update `docs/ARCHITECTURE.md` to match the code as built (diagrams, data flow, every decision that changed).
    - `docs/RUVECTOR_MAPPING.md`: a type-by-type table (Python class ↔ `ruvector-robotics` Rust type), semantic differences (dot vs cosine, Episode fields, side tables for θ), and the drop-in plan.
    - `docs/EXTENDING.md`: add an object class, add a skill parameter, add a memory backend, add a task.
  - Acceptance:
    - Follow the README from a fresh clone in a temp dir (`git clone`, `uv sync`, `uv run vfa demo --quick`). It works. Paste the output tail.
    - Every number in the README appears in `docs/RESULTS.md`.
  - Verify: the fresh-clone run above plus `uv run pytest -m "not slow" -q`.
  - Budget: ~45 min.
  - depends: T19

## Phase F — Stretch: residual learning and the real RuVector backend

- [ ] **T21 — Residual System 1 policy (numpy evolution strategies)**
  - Goal: the "simple RL / residual learning" layer from the brief — learn a small correction on top of the impedance controller.
  - Deliverables:
    - `vfa/skills/residual.py`: a linear residual policy that maps tared F/T features, slip signal, and phase one-hot to Δgrip-force and Δz-velocity during `SQUEEZE`, `LIFT`, and `HOLD`. Bounded output. Off by default.
    - `vfa/experiments/train_residual.py`: OpenAI-style ES (antithetic sampling, numpy only) on the calibration seeds. Reward = success − λ·force jerk − μ·peak-force excess. Saves weights to `runs/<id>/residual.npz`.
    - A `memory_full+residual` condition in the benchmark.
  - Acceptance:
    - Training reward improves monotonically (smoothed) and the curve is saved.
    - Target: `memory_full+residual` force jerk ≥ 20 % lower than `memory_full` with success@1 not lower by more than 3 points. On a miss, follow the CLAUDE.md rule.
  - Verify: `uv run python -m vfa.experiments.train_residual --quick --workers 8` (≤ 9 min; full training is chunked and resumable by ES generation), then the benchmark quick profile for both conditions.
  - Budget: ~60 min.
  - depends: T20

- [ ] **T22 — Real RuVector backend: PyO3 binding over `ruvector-robotics`**
  - Goal: prove the drop-in hooks — run the same memory through the actual Rust crate.
  - Deliverables:
    - `rust/ruvector_vfa_py/`: a maturin + PyO3 crate. It depends on `ruvector-robotics` by git (`github.com/ruvnet/ruvector`, pinned `rev = "3e8429739fc0a3878b553a87bb91e7f66d8c0cbd"`, the latest upstream commit touching `crates/ruvector-robotics` when this plan was written; confirm it with `git ls-remote` or pick a newer one). Expose `EpisodicMemory` (`store`, `recall_similar`), `SemanticMemory` (`store`, `find_similar`), `SkillLibrary` (`learn_from_demonstration`, `improve_skill`), and `CognitiveCore` (mode/state transitions).
    - `vfa/memory/ruvector_backend.py`: `RuVectorBackend(MemoryBackend)`. Rust `Episode` has no id/success fields: store the retrieval key as the single percept, θ JSON + record id in `actions`, and the signed outcome in `reward`. Keep a Python side table for payloads.
    - Optional extra `ruvector` (`maturin`) and a `--backend ruvector` flag on the benchmark and the demo. Tests skip when the extension is not built.
  - Acceptance:
    - Parity test: for the same unit-norm vectors, `RuVectorBackend` and `NumpyBackend` return the same top-k ids in the same order on 1 000 random queries.
    - `vfa demo --quick --backend ruvector` runs end-to-end, and its summary matches the numpy backend for the same seed.
  - Verify: `uv run maturin develop -m rust/ruvector_vfa_py/Cargo.toml && uv run pytest -q tests/test_ruvector_backend.py && uv run vfa demo --quick --backend ruvector`
  - Budget: ~60 min.
  - depends: T20
