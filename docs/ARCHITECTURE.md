# Architecture — ruvector-vfa

Status: **design (pre-implementation)**. Each ROADMAP task implements one part. T20 updates this file to match the code as built. Section numbers are referenced from `ROADMAP.md`; keep them stable.

---

## 1. Thesis

> "Force is the language of manipulation." — the idea behind Eka Robotics' Vision-Force-Action (VFA) approach.

A camera tells the robot *where* a cube is. It does not tell the robot how heavy, slippery, or fragile the cube is. Those properties show up only in **force**: the load on the wrist when the cube leaves the table, the micro-slip when the grip is too weak, the stiffness felt while the fingers close.

This project tests one claim in simulation:

**A robot that stores force-rich experiences in a vector memory, and retrieves similar ones before and during each attempt, succeeds sooner and grips more gently than a robot that does not.**

The memory and cognition layers mirror the `ruvector-robotics` Rust crate (`MemorySystem`, `CognitiveCore`, `SkillLearning`). The Python implementation is a stand-in with the same shapes, so the real crate can replace it (T22).

---

## 2. System overview

```
                         ┌──────────────────────────────────────────────┐
                         │            CognitiveCore (cognition/)         │
                         │                                              │
   triggers:             │   SYSTEM 2 — DELIBERATIVE (event-driven)      │
   attempt start ───────►│   RetrievalPolicy: query memory → θ window    │
   PROBE_DONE ──────────►│   → choose / adapt skill params θ             │
   escalation/failure ──►│   learn(): store attempt, reinforce utility   │
                         │            ▲                    │ θ           │
                         │            │ percepts, events   ▼             │
                         │   SYSTEM 1 — REACTIVE (control rate)          │
                         │   GraspLiftSkill phase machine + slip reflex  │
                         │   → OperationalSpaceController (impedance)    │
                         └────────────┬───────────────────────▲─────────┘
                                      │ torques, grip         │ wrench, contacts,
           SafetySupervisor ──────────┤ (EMERGENCY override)  │ EE pose, width
           (control/safety.py)        ▼                       │
                         ┌──────────────────────────────────────────────┐
                         │  MuJoCo: Panda + table + cube (sim/)          │
                         │  Gymnasium env · sensors · EpisodeRecorder    │
                         └──────────────────────────────────────────────┘
                                      │ episode (force + trajectory + outcome)
                                      ▼
                         ┌──────────────────────────────────────────────┐
                         │  MemorySystem (memory/)                       │
                         │  Working · Episodic · Semantic (SkillLibrary) │
                         │  MemoryBackend: NumpyBackend | RuVectorBackend│
                         └──────────────────────────────────────────────┘
```

**Rates.** Physics `dt = 2 ms` (500 Hz). The OSC torque law, the skill phase machine, the reflexes, and the safety check run at `control_hz` (100–500 Hz; torques are held across physics substeps). T03 picks the highest `control_hz` that keeps simulation ≥ 10× faster than real time on one core, and records it here. System 2 runs only at its three triggers, never inside a control tick. The Gymnasium `osc` action interface steps at 50 Hz for RL users.

**Throughput budget.** Benchmarks run thousands of attempts. One shell command may run at most 10 minutes, so long runs are chunked per condition × seed, resumable, and parallel across processes (`--workers`). Refer to `CLAUDE.md` → *Runtime limits*.

---

## 3. Simulation and tasks

**Scene (T02).** The pristine MuJoCo Menagerie Panda (vendored at commit `feadf76d42f8a2162426f7d226a3b539556b3bf5`, Apache-2.0) loads into `mujoco.MjSpec`. Code then edits the spec: torque `motor` actuators replace the arm's position servos (limits 87/87/87/87/12/12/12 N·m), a table, a free-joint cube, a place target, an `ee` site, and a wrist F/T site with `force` + `torque` sensors are added. The vendored XML is never edited.

As built (`vfa/sim/scene.py`):

- **Frames.** The table top is at world `z = 0` and the robot base sits on it at the origin, so the world frame is the robot base frame. The floor is at `z = −0.75`.
- **Control vector.** `ctrl[0:7]` = arm joint torques (N·m, `motor1..7`); `ctrl[7]` = the menagerie gripper servo (`actuator8`, 0 = closed, 255 = open).
- **Sites.** `ee` = fingertip center (Franka TCP, 0.1034 m from the flange along the hand z axis). `wrist_ft` = the link7 → hand flange. `place_target` lies on the table. `left_pad` / `right_pad` cover all five fingertip pad boxes for the `left_touch` / `right_touch` sensors.
- **Touch vs. cube force.** A touch sensor counts a contact on its finger when MuJoCo's ray test (contact point along the contact normal) hits the pad site, whatever the other object is. `finger_cube_forces()` counts every finger↔cube contact and nothing else. Measured in T02 at the end state of the matching test (the tests assert the relation, not the exact values):
  - face-on pad grasp of the cube: both read the same force, to 1e-9;
  - open fingertips pushed 1 cm into the table by an impedance law: touch ≈ 2.08–2.10 N per finger, helper 0;
  - a 12 cm cube pushed up into the fingertips with 15 N, after 0.12 s: touch ≈ 2.6 N per finger, helper ≈ 6.9–7.7 N per finger, because most fingertip-end contacts are on the finger mesh outside the pad site.

  Cube-specific logic (T04's break rule, grasp checks) must use `finger_cube_forces()`. Treat touch as a pad-contact signal, not a force measurement.
- **Wrist F/T sign.** The reading is the force that link7 applies to the hand subtree, in the hand frame. At `home` the hand z axis points down, so the hand's weight reads as `F ≈ (0, 0, −7.46) N` (0.76 kg × g).
- **Per-episode hooks.** `set_cube_mass`, `set_cube_size`, `set_cube_friction`, `set_cube_solref`, and `place_cube` edit a compiled model in place. A test proves that every numeric model array then equals a fresh compile, so episode loops never recompile. A recompile is cheap anyway: the first `build_scene()` takes ≈ 0.33 s (mesh processing), later ones ≈ 0.025 s (MuJoCo caches the meshes). The hooks reject a cube size outside 5 mm–30 cm, a `solref` time constant below 2 × timestep (MuJoCo would clamp it silently), and a footprint off the table or within 0.16 m of the base axis.
- **Hot loops.** Resolving names (`scene_ids`) costs more than one physics step. Resolve the ids once and pass `ids` to `gravity_compensation`, `read_wrist_ft`, `read_finger_touch`, and `finger_cube_forces`.

**Gripper.** The menagerie gripper is a position servo on a tendon (±100 N force range). T05 measures whether it can deliver a repeatable squeeze force. If not, it becomes a force-controlled tendon `motor` (ctrl = squeeze force in N). T02 measurement: the servo's stiffness is only 100 N/m on the tendon, so fully closed (ctrl 0) on the 5 cm cube it squeezes with ≈ 1.25 N per finger. That is far below the §4 windows (`heavy_slippery` needs ≈ 17 N), so T05 should expect to replace it.

**Contact parameters.** MuJoCo combines two geoms' friction by taking the **maximum** unless one geom has a higher `priority`. The cube geom gets `priority=1`, so *its* friction and `solref` govern finger contacts. Without this, high-friction finger pads hide the cube's friction and the "slippery" class does not exist. Use `condim=4` on the cube so torsional slip is modeled.

**Control (T03).** Operational-space impedance control: `τ = Jᵀ(Kp·e − Kd·ẋ) + qfrc_bias + Nᵀ·τ_posture`, clipped to actuator limits. The end-effector orientation is top-down by default.

**Gymnasium env (T07).** `PandaVFAEnv(task, control, object_class, render_mode)`, registered as `vfa/PandaReach-v0`, `vfa/PandaLift-v0`, `vfa/PandaPickPlace-v0`. Hidden object properties are never in the observation. They appear in `info["privileged"]` for logging and scoring only.

| Task | Success | Failure (terminate) | Truncate | Reward (shaped) |
|---|---|---|---|---|
| Reach | EE within 1 cm of target for 0.2 s | `SAFETY_STOP` | 4 s | −‖ee − target‖ |
| Lift | cube ≥ 10 cm above table, held 1 s, not broken | `BREAK`, `DROP` after lift-off, `SAFETY_STOP` | 10 s | approach distance → grasp bonus → height progress → success bonus |
| Pick-and-place | cube within 2 cm of target, released, upright, not broken | `BREAK`, `PLACE_BREAK`, `DROP`, `TOPPLE`, `SAFETY_STOP` | 15 s | lift terms + transport distance + place bonus |

---

## 4. Learning problem design — why memory and force must matter

If one fixed set of skill parameters works for every object, memory is pointless. The benchmark therefore draws objects from **hidden classes with conflicting force requirements**. All cubes look identical (same color; sizes overlap across classes), so vision alone cannot identify the class.

Initial class table (T04 implements it; **T05 validates and may adjust it**; T10 freezes it in `configs/objects.toml`):

| Class | Mass (kg) | Friction μ | Contact softness | Break force (N) | What goes wrong with a wrong θ |
|---|---|---|---|---|---|
| `light_rigid` | 0.05–0.15 | 0.8–1.0 | stiff | none | almost nothing — the easy class |
| `heavy_rigid` | 0.6–1.0 | 0.7–1.0 | stiff | none | slips if grip is low or lift is fast |
| `slippery` | 0.15–0.35 | 0.15–0.30 | stiff | none | needs high grip for its mass |
| `fragile_soft` | 0.05–0.20 | 0.6–0.9 | soft | 6–12 | breaks if grip is high **or** closing is fast (force overshoot) |
| `heavy_slippery` | 0.5–0.8 | 0.25–0.40 | stiff | none | needs the highest grip — conflicts with `fragile_soft` |

Rough required grip with two finger contacts and lift acceleration `a`: `F_min ≈ m(g + a) / (2μ)`. Example: `heavy_slippery` needs ≈ 17 N; `fragile_soft` breaks at 6–12 N. **No single grip force satisfies both.** T05 proves this in physics; T10 proves it statistically (`best_fixed` ≤ 75 % success@1).

**Skill parameters θ** (`SkillParams`, T08): `grip_force`, `close_speed`, `lift_accel`, `grasp_depth`, `approach_speed`. T19 adds `place_speed`, `release_height`.

**Touch-and-heft probe.** Every attempt starts its grasp with a gentle probe that is safe for every class: close slowly to a low force, then raise the cube 1–2 cm. The probe's force signature carries the hidden properties:

- closure-force slope → contact stiffness → fragility;
- wrist load during the heft → mass;
- whether and when the cube slips under the probe grip → friction.

System 2 reads the probe (`PROBE_DONE`) and sets θ for the rest of the attempt. If the probe proves too noisy (T05/T11 decide), the fallback still works: between retries, the failed attempt's full force profile is the retrieval key (`key_retry`).

**Failure modes** (event names are shared by recorder, env, and memory): `MISS` (no contact at expected closure), `SLIP`, `DROP`, `BREAK`, `TIMEOUT`, `SAFETY_STOP`; pick-and-place adds `PLACE_BREAK`, `TOPPLE`.

---

## 5. Memory

### 5.1 Embeddings and retrieval keys (T11)

The memory stores **what was available at each decision point**, not only the final episode. Before an attempt only the context exists; after the probe, the probe forces exist; at a retry, the whole failed attempt exists. Each key type has its own index (and, in the Rust backend, its own `EpisodicMemory` instance).

| Vector | Built from | Used when |
|---|---|---|
| `key_pre` | context: observed cube pose and size, initial joint config | attempt start |
| `key_probe` | context + probe force features | after `PROBE_DONE` |
| `key_retry` | context + previous attempt's full force profile + its failure mode one-hot | retry after a failure |
| `full_embedding` | context + force + trajectory + outcome | consolidation, analysis |

Feature groups:

- **Force** — per phase, resample the tared wrist wrench (Fz, ‖Fxy‖, ‖τ‖) and the summed finger normal force to a fixed length (default 16 samples). Add key moments: contact-onset time, peak force, closure-force slope, heft load, slip-event count.
- **Trajectory** — EE path relative to the object (16 points × 3) and a speed profile (16).
- **Context** — observed cube position (3), size (1), initial joint positions (7).

Each group is standardized with statistics fitted on the committed calibration fixture `tests/fixtures/calib_episodes.npz` (saved to `configs/embedding_stats.npz`), multiplied by a group weight, concatenated, and **L2-normalized**. Unit vectors make dot product equal to cosine similarity, which matches the Rust crate's dot-product recall (§10).

### 5.2 Tiers (T12, T13)

| Tier | Holds | Lifetime |
|---|---|---|
| `WorkingMemory` | the current attempt's percepts, probe features, retrieved hits | one attempt; bounded; evicts least important |
| `EpisodicMemory` | every attempt: keys, full embedding, θ, outcome, failure mode, situation id, `utility` | whole run; persisted |
| `SemanticMemory` | skill prototypes: centroid vector → named `Skill` (mean trajectory, mean θ, confidence) | consolidated every N situations |

### 5.3 Reinforcement and consolidation

- Every episodic record has `utility ∈ [0, 1]` (initial 0.5). When System 2 uses a record and the attempt succeeds, the record's utility rises; on failure it falls. Ranking score: `similarity × (0.5 + utility)`.
- `SkillLibrary.consolidate()` clusters successful episodes on `key_probe` (no class labels), learns one `Skill` per cluster with `learn_from_demonstration` (averaged trajectory and θ, `confidence = 1 − 1/(n+1)`), and stores the centroid in `SemanticMemory`. `improve_skill(name, feedback)` adjusts confidence after each use (clamped to [0, 1]), exactly as in the Rust crate.

### 5.4 Backend seam

`MemoryBackend` (Protocol): `add`, `search`, `update`, `save`, `load`. `NumpyBackend` is the default (brute force, cosine or Euclidean; fine for thousands of records). `RuVectorBackend` (T22) runs the same calls through the Rust crate via PyO3. A later backend can use `ruvector-core` HNSW for scale without touching cognition code.

---

## 6. Dual-process loop (T14)

`CognitiveCore` follows the crate's perceive → think → act → learn cycle and its three modes.

| Mode | Who acts | Runs when |
|---|---|---|
| `REACTIVE` (System 1) | `GraspLiftSkill` + slip reflex + OSC | every control tick (`control_hz`) between System 2 decisions |
| `DELIBERATIVE` (System 2) | `RetrievalPolicy` | attempt start · `PROBE_DONE` · escalation (`SLIP_UNRECOVERED`, `MISS`) · after a failure |
| `EMERGENCY` | `SafetySupervisor` | any safety violation; overrides both |

Typical attempt (logged to `runs/<id>/handoffs.jsonl` and the console):

```
[S2] attempt 1  trigger=start      key_pre   hits=3 (best sim 0.71)  θ.grip=8.0N
[S1] APPROACH → DESCEND → PROBE
[S1] escalate PROBE_DONE  heft=2.9N slope=410N/m slip=0
[S2] trigger=probe       key_probe hits=5 (best sim 0.93, success) θ.grip=6.1N close=0.02m/s
[S1] SQUEEZE → LIFT → HOLD
[MEM] store ep_0117 success  reinforce ep_0042 +0.1
```

A failed attempt adds `[S2] trigger=failure key_retry …` and a retry with an adapted θ.

---

## 7. Retrieval-augmented adaptation and retry (T15)

```
solve_situation(situation):
    for attempt in 1..max_attempts:
        θ = S2.decide(key_pre)                      # trigger: attempt start
        run System 1 until PROBE_DONE
        θ = S2.decide(key_probe, prior=θ)           # trigger: probe
        result = run System 1 to the end (reflexes may escalate to S2)
        memory.store(result); memory.reinforce(used_records, result.success)
        if result.success: return attempt
        key_retry = embed(result)                   # the failure is now the best evidence
    return FAILED

S2.decide(key):
    S, F = top-k successes, top-k failures by similarity × (0.5 + utility)
    center = similarity-weighted mean of θ over S        (default θ if S is empty)
    for each failure f in F (weighted by similarity):
        SLIP / DROP at grip g  → lower bound ≥ g · (1 + margin)
        BREAK at grip g        → upper bound ≤ g · (1 − margin)
        BREAK at close speed v → close-speed upper bound ≤ v · (1 − margin)
    θ = clip(center, bounds); if S is empty: θ = midpoint of bounds
    θ += noise with σ = σ0 · (1 − best_similarity · skill_confidence)
```

On a retry of the *same* situation, the failed attempt has similarity ≈ 1, so its bound dominates — the window narrows like a bisection guided by force evidence. Across situations, the memory transfers what worked for objects that *felt* similar during the probe.

---

## 8. Safety (T09)

`SafetySupervisor` checks every control tick: wrist force/torque norm limits, an EE workspace box, an EE speed cap, joint-limit margins, and sustained torque saturation. A violation switches `CognitiveCore` to `EMERGENCY`, holds the current pose under gravity compensation, freezes the gripper, and ends the attempt with `SAFETY_STOP` and a reason. Safety stops are failures that memory stores like any other.

---

## 9. Experiments and metrics (T10, T15, T18)

**Conditions**

| Condition | Description |
|---|---|
| `naive` | one fixed default θ; retries repeat it |
| `best_fixed` | the single θ with the best calibration-seed success (strongest non-adaptive baseline) |
| `random_retry` | `naive` first, then random θ within bounds; same retry budget as memory |
| `memory_ctx_only` | full retrieval policy, but every key uses context features only — **the force ablation** |
| `memory_full` | full retrieval policy with force features |
| `memory_full+residual` | plus the learned residual (T21) |

**Metrics**

- success@1 — first-attempt success rate (over situations, after a warm-up window where stated).
- success@k — success within `max_attempts`.
- attempts-to-success — mean attempts per situation; an unsolved situation counts as `max_attempts + 1`.
- break rate, drop rate — per attempt.
- peak-force excess — peak finger normal force minus the privileged `F_min` estimate (lower = gentler).
- force jerk — RMS time derivative of the filtered finger normal force during `SQUEEZE` + `LIFT` (lower = smoother).

**Protocol.** Calibration seeds (for `best_fixed`, embedding statistics, residual training) are disjoint from evaluation seeds. The benchmark (`configs/benchmark.toml` + `configs/objects.toml` + `configs/best_fixed.toml`) is frozen by hash in T10, before any memory code exists. Profile sizes come from the attempts/minute measured in T10. Results are mean ± std over 3 evaluation seeds. A held-out run freezes memory after one warm-up seed and evaluates on new situations with no writes.

---

## 10. Mapping to `ruvector-robotics`

Source read: `github.com/ruvnet/ruvector`, `crates/ruvector-robotics/src/cognitive/` (local clone at `~/code/utilities/ruvector`).

| Python (`vfa`) | Rust (`ruvector_robotics::cognitive`) | Notes |
|---|---|---|
| `memory.WorkingMemory` | `memory_system::WorkingMemory`, `MemoryItem` | same rule: evict the least-important item when full |
| `memory.EpisodicMemory` | `memory_system::EpisodicMemory`, `Episode { percepts, actions, reward, timestamp }` | Rust `recall_similar` flattens `percepts` and uses a **raw dot product** → we store one unit-norm key as the single percept. One instance per key type. Rust `Episode` has no id/success: θ JSON + record id go in `actions`, signed outcome in `reward`, payloads in a Python side table |
| `memory.SemanticMemory` | `memory_system::SemanticMemory` | name → embedding; `find_similar` is also dot product |
| `memory.SkillLibrary`, `Skill`, `Demonstration` | `skill_learning::{SkillLibrary, Skill, Demonstration}` | same averaging and `confidence = 1 − 1/(n+1)`; `improve_skill` clamps to [0, 1]. Rust `Skill.trajectory` is `Vec<[f64; 3]>`; θ lives in a side table |
| `cognition.CognitiveCore`, `CognitiveMode`, `CognitiveState` | `cognitive_core::{CognitiveCore, CognitiveMode::{Reactive, Deliberative, Emergency}, CognitiveState}` | same modes and states. Rust `think()` picks the most confident percept; our System 2 replaces it with `RetrievalPolicy` |
| `cognition.Percept`, `Decision`, `Outcome` | same names | `Outcome { success, reward, description }` |
| `control.SafetySupervisor` | `CognitiveMode::Emergency` (priority 255 in `think`) | |
| `memory.backend.MemoryBackend` | — (adapter seam) | `NumpyBackend` now; `RuVectorBackend` in T22 |

**Drop-in path (T22).** A small maturin + PyO3 crate (`rust/ruvector_vfa_py/`) depends on `ruvector-robotics` by git and exposes `EpisodicMemory`, `SemanticMemory`, `SkillLibrary`, and `CognitiveCore`. `RuVectorBackend` implements `MemoryBackend` on top of it. A parity test proves both backends return the same top-k on unit-norm vectors.

**Candidate extensions** (crate modules not used yet): `behavior_tree` (skill phases as a tree), `decision_engine` (utility-based θ selection), `world_model`, `perception::scene_graph`, and `bridge::indexing` (spatial index).

---

## 11. Package layout (target)

```
src/vfa/
  cli.py                  # `vfa` entry point (demo, benchmark, scene-check)
  sim/                    # scene.py (MjSpec), objects.py, sensors.py, recorder.py, env.py
  control/                # osc.py, gripper.py, safety.py
  skills/                 # params.py, pick.py, reflexes.py, residual.py (T21)
  memory/                 # embeddings.py, backend.py, memory_system.py, skill_learning.py, ruvector_backend.py (T22)
  cognition/              # cognitive_core.py, system2.py, retry.py
  experiments/            # baselines.py, benchmark.py, demo.py, run_all.py, train_residual.py
  viz/                    # force_plot.py, plots.py, video.py, report.py
assets/menagerie/franka_emika_panda/   # vendored, pristine, Apache-2.0
configs/                  # objects.toml, benchmark.toml, best_fixed.toml, benchmark.sha256, embedding_stats.npz
scripts/                  # spike_grip.py and one-off tools
rust/ruvector_vfa_py/     # T22
tests/                    # fixtures/calib_episodes.npz (committed, ≤ 5 MB)
docs/                     # BRIEF, ARCHITECTURE, FEASIBILITY, RESULTS, RUVECTOR_MAPPING, EXTENDING
runs/                     # git-ignored artifacts
```

---

## 12. Decision log

| # | Decision | Why |
|---|---|---|
| D1 | Vendor the pristine Menagerie Panda at a pinned SHA; build the scene with `MjSpec` | offline and reproducible; license stays clean; no XML fork to maintain |
| D2 | Torque actuators + operational-space impedance control | position servos hide contact force; compliance is the point of a force-aware arm |
| D3 | Hidden object classes with conflicting force windows + touch-and-heft probe | without them memory cannot beat a fixed θ, and force cannot beat vision |
| D4 | Feasibility spike (T05) before skills and memory | the premise must hold in physics before anything is built on it |
| D5 | Freeze the benchmark by hash before memory exists (T10) | prevents tuning the test to the method |
| D6 | Separate retrieval keys per decision point | at query time only part of the episode exists |
| D7 | L2-normalize every stored vector | dot product equals cosine → parity with the Rust crate's dot-product recall |
| D8 | Cube geom `priority=1`, `condim=4` | MuJoCo uses max friction otherwise, which would erase the slippery classes |
| D9 | numpy-only default backend; Rust binding last | runs on any laptop; the seam is proven before the dependency is added |
| D10 | uv + Python 3.12, TOML configs | system Python 3.15 has no MuJoCo wheels; TOML is stdlib |
