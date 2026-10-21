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

**Rates.** Physics `dt = 2 ms` (500 Hz). The OSC torque law, the skill phase machine, the reflexes, and the safety check run at `control_hz` (100–500 Hz; torques are held across physics substeps). T03 picked the highest `control_hz` that keeps simulation ≥ 10× faster than real time on one core: **`control_hz = 500`** (`vfa.control.osc.CONTROL_HZ`), one control tick per physics step. Measured with `uv run python -m vfa.control.throughput` (a 10 s scripted grasp-and-lift under OSC + gripper, best of 5, Apple M3 Max, load average ≈ 4.5): 0.43 s wall at 500 Hz (23×), 0.29 s at 250 Hz (35×), 0.20 s at 100 Hz (51×); CPU time equals wall time (one core). Machine load moves these numbers a lot: at load ≈ 9.6 the same CLI measured 0.73 s at 500 Hz and the pytest `perf` test 0.84 s (12×). Per tick at 500 Hz (≈ 88 µs, instrumented at load ≈ 4.5): physics (`mj_step1` + `mj_step2`, with contacts) ≈ 29 µs, `OperationalSpaceController.compute` ≈ 37 µs, `advance_reference` ≈ 6 µs, `set_target` with the scripted path ≈ 14 µs, `Gripper.apply` ≈ 2 µs; `control_step`'s own checks add ≈ 4 µs on top. The default 10 s episode is the benchmark; in longer runs the cube slips out of the 1 N grip, so they mostly time a contact-free hold. **Headroom for T06.** Naive sensor reads cost ≈ 72 µs per tick in the grasp (32 active contacts): an extra `mj_forward` ≈ 33 µs, `finger_cube_forces` ≈ 36 µs, wrist F/T + touch + ee velocity ≈ 2.4 µs. That is ≈ 0.36 s per 10 s episode: it fits at low load (≈ 0.8 s) but misses 10× under heavy load. T06 should skip the extra `mj_forward` (after `control_step`, the contacts, constraint forces, and `sensordata` describe the start of the last physics step, 2 ms old), then re-measure. If 500 Hz still misses 10×, drop to 250 Hz. `control_hz` below 100 Hz is rejected (`MIN_CONTROL_HZ`) to keep a stability margin. Measured at the 8 box corners after 2 s: at the default gains 50 Hz still ends ≤ 3.0 mm off, but 25 Hz misses 6 of 8 corners (up to 45 mm / 20°). At 4× the default stiffness 100 Hz ends ≤ 0.3 mm off, while 50 Hz saturates the motors and misses one corner (12 mm / 5.1°). System 2 runs only at its three triggers, never inside a control tick. The Gymnasium `osc` action interface steps at 50 Hz for RL users.

**Throughput budget.** Benchmarks run thousands of attempts. One shell command may run at most 10 minutes, so long runs are chunked per condition × seed, resumable, and parallel across processes (`--workers`). Refer to `CLAUDE.md` → *Runtime limits*.

---

## 3. Simulation and tasks

**Scene (T02).** The pristine MuJoCo Menagerie Panda (vendored at commit `feadf76d42f8a2162426f7d226a3b539556b3bf5`, Apache-2.0) loads into `mujoco.MjSpec`. Code then edits the spec: torque `motor` actuators replace the arm's position servos (limits 87/87/87/87/12/12/12 N·m), a table, a free-joint cube, a place target, an `ee` site, and a wrist F/T site with `force` + `torque` sensors are added. The vendored XML is never edited.

As built (`vfa/sim/scene.py`):

- **Frames.** The table top is at world `z = 0` and the robot base sits on it at the origin, so the world frame is the robot base frame. The floor is at `z = −0.75`.
- **Control vector.** `ctrl[0:7]` = arm joint torques (N·m, `motor1..7`); `ctrl[7]` = the gripper (`actuator8`): since T05 a force motor on the `split` tendon, ctrl = tendon force in N (±100 N, positive opens, each finger gets half). `SceneConfig(gripper="servo")` keeps the menagerie servo (0 = closed, 255 = open). Callers use `Gripper`, never `ctrl[7]`.
- **Sites.** `ee` = fingertip center (Franka TCP, 0.1034 m from the flange along the hand z axis). `wrist_ft` = the link7 → hand flange. `place_target` lies on the table. `left_pad` / `right_pad` cover all five fingertip pad boxes for the `left_touch` / `right_touch` sensors.
- **Touch vs. cube force.** A touch sensor counts a contact on its finger when MuJoCo's ray test (contact point along the contact normal) hits the pad site, whatever the other object is. `finger_cube_forces()` counts every finger↔cube contact and nothing else. Measured in T02 at the end state of the matching test (the tests assert the relation, not the exact values):
  - face-on pad grasp of the cube: both read the same force, to 1e-9;
  - open fingertips pushed 1 cm into the table by an impedance law: touch ≈ 2.08–2.10 N per finger, helper 0;
  - a 12 cm cube pushed up into the fingertips with 15 N, after 0.12 s: touch ≈ 2.6 N per finger, helper ≈ 6.9–7.7 N per finger, because most fingertip-end contacts are on the finger mesh outside the pad site.

  Cube-specific logic (T04's break rule, grasp checks) must use `finger_cube_forces()`. Treat touch as a pad-contact signal, not a force measurement.
- **Wrist F/T sign.** The reading is the force that link7 applies to the hand subtree, in the hand frame. At `home` the hand z axis points down, so the hand's weight reads as `F ≈ (0, 0, −7.46) N` (0.76 kg × g).
- **Per-episode hooks.** `set_cube_mass`, `set_cube_size`, `set_cube_friction`, `set_cube_solref`, and `place_cube` edit a compiled model in place. A test proves that every numeric model array then equals a fresh compile, so episode loops never recompile. A recompile is cheap anyway: the first `build_scene()` takes ≈ 0.33 s (mesh processing), later ones ≈ 0.025 s (MuJoCo caches the meshes). The hooks reject a cube size outside 5 mm–30 cm, a `solref` time constant below 2 × timestep (MuJoCo would clamp it silently), and a footprint off the table or within 0.16 m of the base axis.
- **Hot loops.** Resolving names (`scene_ids`) costs more than one physics step. Resolve the ids once and pass `ids` to `gravity_compensation`, `read_wrist_ft`, `read_finger_touch`, and `finger_cube_forces`.

**Gripper.** The menagerie gripper is a position servo on a tendon (±100 N force range). T02 measurement: the servo's stiffness is only 100 N/m on the tendon, so fully closed (ctrl 0) on the 5 cm cube it squeezes with ≈ 1.25 N per finger. **T05 replaced it** with a force-controlled tendon `motor` (ctrl = tendon force in N; `Gripper.squeeze` maps a per-finger squeeze onto it, so callers never write `ctrl` themselves): the servo is repeatable but flat at 25·width N per finger (1.02–1.49 N on 4.2–6.0 cm cubes), and the motor lands every command from 0.25 to 32 N to ≤ 0.001 N (`docs/FEASIBILITY.md` §3).

As built in T03 (history: the T05 bullets below supersede the closing behavior, the jam, and the creep): `Gripper` has `open()`, `close(width)` (opening in m, 0–0.08), `squeeze(force_n)`, and `width()` readback. **`force_n` is per finger**: the normal force each pad presses on the object, the `F_min` of §4. Every control tick `apply()` picks a desired tendon force `F*` and solves MuJoCo's actuator line `F = g·ctrl + b0 + b1·L + b2·L̇` for `ctrl` at the current tendon length `L` (`L = c·width` with the tendon coupling `c = 0.5`, read from the model; the class requires `gear = 1`). Position commands use `F* = kp·(c·w* − L) − kd·L̇` with the servo's own 100 N/m and 10 N·s/m; for the menagerie servo that is exactly its native command `ctrl = 255·w*/0.08`. A squeeze uses `F* = −force_n/c − kd·L̇`, so the fingers keep the servo damping while they close. With the menagerie servo the ctrl floor caps the squeeze at `25·width` N per finger (1.25 N on a 5 cm cube); `saturated` and `max_squeeze_force()` report it. Measured in a face-on grasp: `finger_cube_forces` equals the request to < 1e-7 (relative; the tests check 1e-3) for 0.25–1.2 N. Nothing limits the closing speed: a squeeze from wide open overshoots on impact, then settles on the request. Measured on a floating cube at 500 Hz: servo 1.96–2.07× (3 cm cube at 0.5 N, 5 and 7 cm at 1 N); force motor 1.6–2.0× (5 cm cube at 6–20 N, 3 cm cube at 6–10 N), but at 15–20 N on a 3 cm cube the fingers punch through the soft contact (peak ≈ 1330–1380 N, width down to −2.6 mm) before they settle. Closing to 4 mm clearance by position first still leaves 1.37–1.71× (2–20 N): a 4 N squeeze peaks at 6.23 N and a 6 N squeeze at 9.05 N, inside `fragile_soft`'s 6–12 N break range (§4). T05 owns the closing profile (a force ramp or a closing-speed limit, and contact stiffness). With a force motor, a squeeze of ≥ 5 N that closes on *nothing* drives the light fingers through their 0 joint limit and into each other, and they jam (measured: −11 mm at 5 N, −49 mm at 20 N, still there after `open()`); the servo cannot do this. A motor swap in T05 needs an end stop or a closing-speed limit. The T03 throughput episode also shows soft-contact creep: in a 1 N grip (μ = 1, twice the friction the 0.1 kg cube needs) the cube slides down ≈ 1.2 mm/s, from 9.7 cm lifted at 6.5 s to 9.3 cm at 10 s. T05 owns grip-force and contact tuning. The class reads the gain, bias, and ranges from the model, so T05 can swap in a force `motor` without changing the force mapping. A test does that swap (±100 N motor): squeezes of 2–45 N land to < 1e-8 (tests check 1e-3), and the position commands emulate the servo. `command` is read-only; a command whose `apply()` raises does not become active.

As built in T05 (`vfa/control/gripper.py`, measured in `docs/FEASIBILITY.md`):

- **Closing speed.** `squeeze(f, speed=v)` (the libfranka `grasp(width, speed, force)` idea; `v` in 0.001–0.5 m/s, the rate of change of the opening, default 0.5 m/s) closes the fingers at `v` and then squeezes with `f`. A width target moves toward 0 at `v`; a PD law on the tendon (`TRACK_KP` 1000 N/m, `TRACK_KD` 30 N·s/m: critically damped for the 0.23 kg tendon inertia of two 15 g fingers with 0.1 kg joint armature, and stable at 100 Hz) pulls the fingers along, and `F* = max(−f/c, F_track)`, so the actuator never asks for more than `f`. A new command starts bumplessly: the target is placed where its force equals the force the actuator applies now, so the fingers accelerate smoothly to `v` (never above it; half of `v` after 24 ms, 95 % after 78 ms) and a new squeeze continues from the current force without a step or a dip (except a switch to a slower speed while the fingers still move faster, below). When the moving fingers hit the object the squeeze steps by `c²·TRACK_KD·v` = 7.5·v N per finger (the damping on the target velocity, capped at `f`); then the target runs ahead and the squeeze rises at `TRACK_KP·c²·v` = 250·v N/s to `f`, which it holds with no damping in that branch. A weaker request applies at once. The target never leads by more than the full squeeze needs (anti-windup). The bound's force is the squeeze while the fingers close slower than `v` (its speed term is capped at `c·v`, so fingers pushed open move the bound only one-for-one), so a weak squeeze still closes at `v` and a stronger one ramps on from a hold without a step (review round 3: before, 0.5 N at 0.5 m/s settled at 0.07 m/s). The target advances at most 10 ms per `apply` and restarts at the fingers when `reset_to_home` moves time back. Re-sending the same command every tick (T07's per-step grip action) changes nothing; a changed speed-limited squeeze keeps the moving target and its clock (shifted so the force stays continuous), so a command that changes every tick still closes and ramps (round 3: it used to stall at 0 N). After a gap of more than 10.5 ms between `apply` calls (or a reset) the target restarts at the fingers bumplessly, changed command or not; the restarted target sits no further open than a hold would place it, and at the fingers if ctrl froze on an opening force (the end-stop brake). Call `apply` every control tick at ≥ 100 Hz, the control floor the gains are sized for: slower calls stall the fingers or make the speed servo unstable (several m/s, bouncing open), so `apply` raises `RuntimeError` when two calls in a row each come after a gap (`control_step` checks only controllers with a `control_hz`, and `Gripper` has none). The speed servo never pulls the fingers open while a fingertip pad's touch sensor (one physics step old) reads a contact, so switching a fast probe to a slower squeeze at contact does not let go (round 4: it pulled the fingers off soft cubes for 3–14 ticks). It is not bumpless either: the motor force can drop to 0 while the slower target catches up (48 probed switches: up to 0.22 s at 0 N, contact force down to 0.017 N on the softest cube, never lost). A command outlives a reset (send `open()`). The speed limits the approach only: after an object is lost the fingers close under the held squeeze (20–50 N: up to 1.9–4.3 m/s at 500 Hz, 2.0–5.0 m/s averaged over a 10 ms tick at 100 Hz). The end stop and the finger joint limits stop them (at 50 N they overrun to −9 mm at 500 Hz, −30 mm at 100 Hz), and `open()` reopens to 8 cm. A brake that dropped the target's lead on overspeed halved the 500 Hz speeds but pulled fingers off soft cubes when the pad sensor missed a first contact, so it is not used. **`speed=None` is the T03 law, for tests only**: it hits a soft cube at several m/s (30 N on a 5 cm cube with time constant 0.04 ended at −50 mm width with 145 N per finger, and `open()` did not free it). **The speed sets the force profile:** on `fragile_soft` cubes (30 per cell) the median peak summed force over `2f` is 1.000 at 0.02 m/s and 1.267 / 1.199 / 1.147 / 1.103 at 0.5 m/s for `f` = 2 / 2.5 / 3 / 3.5 N, and the fast close breaks more cubes (63/180 vs 43/180; per cube 19 vs 0, sign test p = 1.9e-6). The impact barely depends on the request, so it weighs most at small forces on stiff cubes: 3.13× (1 N) and 1.41× (3 N) on `light_rigid` at 0.5 m/s. A current-controlled motor has no velocity feedback, so the force branch has no damping: the T03 servo damping would cut the 2 / 3 N overshoots to 1.100 / 1.065 and hide most of the effect. A slow speed with a large force is a slow grasp (30 N at 0.02 m/s ramps for 6 s), so T08 must bound θ accordingly.
- **End stop.** In a force mode the closing force is capped below 3 cm (`END_STOP_WIDTH + END_STOP_BAND`), falling linearly to 0 at 5 mm; inside that band a brake (`TRACK_KD`) slows fingers closing faster than 0.5 m/s scaled the same way. The full force needs an object at least 3 cm wide (a 45 N squeeze compressed a soft 4.2 cm cube to 2.7 cm and was capped at 44.2 N, `saturated`). The scene also excludes finger-finger contacts: with the end stop alone a 50 N squeeze without a speed still jams the finger meshes into each other (−47 mm at 500 and 100 Hz, still there after `open()`; a test pins it). With both, a squeeze on nothing (5, 10, 20, 50 N; with or without a speed; 100 and 500 Hz) comes to rest and `open()` reopens to 8 cm. The stop only removes the closing force, so with a speed of 0.5 m/s the fingers stop a little inside it (3.4 mm at 500 Hz; down to 1.6 mm at 100 Hz, then 5 mm), never below 0. Without a speed (T03 law) they reach 1.5–8.0 m/s and overrun past 0 for a moment (−8 mm at 20 N, −28 mm at 50 N at 500 Hz; −25 mm at 50 N at 100 Hz); below the stop nothing pushes them back, so they rest at 0.5–5 mm.
- **Paths need feed-forward.** Over 35 cubes a 12 cm, 1.0 s minimum-jerk descent without `vel=` drifted 7.2 mm sideways (median; max 7.9 mm). An earlier rate-limited target drifted 13.6 mm, enough to land a fingertip on a 5.9 cm cube. With `vel=`: 1.8 mm (max 2.0 mm). T08's approach, descent, and lift must feed the path velocity forward.
- **Held mass from the wrist.** The vertical wrist force over the wrist's vertical proper acceleration (`mj_objectAcceleration` on the wrist site; an accelerometer in a real F/T sensor reads the same, +g at rest) is the mass below the wrist; minus the hand's 0.760 kg (the tare) it is the held cube's mass: over 1114 held cubes the low-passed estimate stayed within 0.40 % of its running peak, and lifts up to 4.71 m/s² peak raised no false alarm. T05's slip detector fires when it stays below 0.9 × its peak for 6 ms (precision 0.991, recall 0.996 on 1939 noise-free attempts; 0.919 / 1.000 on held-out cubes; 0.773 / 1.000 with white noise of 0.05 N on F_z and 0.02 m/s² on a_z; 0.870 / 0.930 on the 62 attempts whose cube slid 1–50 mm; `docs/FEASIBILITY.md` §5). It is also the heft probe's mass reading. T06 owns the sensor: add it to the scene, model its noise, and re-size the filter, threshold, and debounce.

**Contact parameters.** MuJoCo combines two geoms' friction by taking the **maximum** unless one geom has a higher `priority`. The cube geom gets `priority=1`, so *its* friction and `solref` govern finger contacts. Without this, high-friction finger pads hide the cube's friction and the "slippery" class does not exist. Use `condim=4` on the cube so torsional slip is modeled. **Since T05 the scene uses elliptic friction cones with `impratio = 10`** (D11; MuJoCo's advice for grasping). With the default pyramidal cones and impratio 1, a held cube slid 2.9–9.4 mm in the grip over a 2 s lift and hold at 1.1× the grip it needs (1.5–5.5 mm at 2×; one cube per class); now 0.12–0.47 mm at 1.1×, while 0.9× still drops at lift-off. So `F_min = m(g + a)/(2μ)` is a sharp threshold. The T03 throughput episode costs no more (0.470–0.482 s pyramidal vs 0.458–0.469 s elliptic per 10 s episode, measured on a loaded machine) and its 1 N grasp no longer creeps (9.87 cm at 10 s, vs 9.25 cm).

**Control (T03).** Operational-space impedance control: `τ = Jᵀ(Kp·e − Kd·ẋ) + qfrc_bias + Nᵀ·τ_posture`, clipped to actuator limits. The end-effector orientation is top-down by default.

As built (`vfa/control/osc.py`, `OperationalSpaceController` + `OSCGains`):

- **Law.** `τ = Jᵀ·(K·e + D·(v_ref − ẋ)) + qfrc_bias + Nᵀ·τ_posture`, clipped to the motor limits (`saturated` flags a clip; `at_joint_limit` flags an arm joint within 0.01 rad of its range end, a stall the torque limits can miss; a non-finite torque or arm dynamics raises `FloatingPointError` instead of reaching the motors). `e` is the `ee` site's position error and axis-angle orientation error (shortest arc) to the **reference pose**, world frame. `J` comes from `mj_jacSite`; `M` is the arm block of `mj_fullM`.
- **Gains.** `K` is a physical stiffness: default 1500 N/m and 60 N·m/rad, per axis if needed. At rest an external force `F` at the ee deflects it by `F/K`; the speed limit never caps the spring (measured along +y after a 2–3 s push: 10 N → 6.6667 mm at 1500 N/m and 33.326 mm at 300 N/m, F/K = 33.333, the slowest mode still settling; 70 N → 46.667 mm at 1500 N/m; 30 N → 100.000 mm at 300 N/m). Only the motor torque limits cap it: a 70 N push along +x saturates joint 6 (12 N·m) and deflects 45.8 mm with a 3° tilt. The menagerie joint damping (1 N·m·s/rad) is not compensated; it adds damping only. **Damping is modal:** one 6×6 `eigh` per tick, `K½·Λ⁻¹·K½ = V·Ω²·Vᵀ` with `Λ = (J·M⁻¹·Jᵀ)⁻¹`, gives the mode frequencies `Ω`, `Λ = K½·V·Ω⁻²·Vᵀ·K½`, and `D = K½·V·(2ζ/Ω)·Vᵀ·K½`. Every mode of `Λ·ẍ + D·ẋ + K·x = 0` then has the damping ratio ζ (default 1) in every configuration (double diagonalization, Albu-Schäffer et al. 2003). Explicit `pos_damping` / `rot_damping` replace that block with a diagonal and drop its coupling. History: per-axis damping from the diagonal of Λ went unstable at 100–125 Hz at the far-high corners (2 of 8 corners end ≈ 22 mm off after 2 s). Per-axis damping from the apparent mass `1/(Λ⁻¹)_ii` was stable, but a review found it underdamped the coupled modes (effective ζ 0.41–0.96 per axis over 30 poses): a long move from a non-home pose, (0.579, −0.019, 0.15) → (0.389, −0.033, 0.237), overshot 12.7 mm (13.4 mm at 100 Hz). Modal damping is stable at 100–500 Hz, and that move overshoots 0.00 mm. The price: heavy modes settle slowly (at the far-high corners the slowest mode has ω ≈ 3.5 rad/s 2 s into the reach, ≈ 2.8 rad/s once the elbow settles), so a reach to a box corner is still ≈ 3 mm / 2.4° off after 2 s. The fastest mode at the default gains has ω ≈ 33 rad/s, so `2ζω·dt` ≤ 0.67 at 100 Hz.
- **Posture.** `τ_posture = 10·(q_home − q) − 2·q̇` (N·m) through the dynamically consistent projector `Nᵀ = I − Jᵀ·Λ·J·M⁻¹`. It moves the elbow without moving the ee (measured: 0.59 rad of joint motion, ≤ 0.35 mm ee deviation; the test bound is 2 mm).
- **Reference pose and speed limit.** `set_target` sets the goal; the spring pulls toward a reference pose that `apply()` moves toward the goal at most 0.5 m/s and 1.5 rad/s per control tick (`max_lin_speed`, `max_ang_speed`; `math.inf` disables them). `v_ref` is always the goal's own feed-forward velocity (`vel=`, clipped to the limits), never the reference's chase velocity. A fixed goal has `v_ref = 0`, so while the reference chases it the damper resists the motion, the ee trails the reference, and with ζ = 1 it arrives without overshoot. On the goal, the reference follows it exactly. Measured over the 8 box corners, a floor point, and 10 random targets at 500 and 100 Hz: peak ee speed 0.511 m/s for the 0.5 m/s limit, overshoot along the path 0.00 mm, no finger–table contact, no torque saturation. A yaw ±π/2 turn in place peaks at 1.500 rad/s for the 1.5 rad/s limit; combined with a move to a box corner the hand turns at up to 1.75 rad/s. With `vel=` the ee can pass the limit (0.52–0.53 m/s on a 0.3 m minimum-jerk path in 0.6–1.0 s). With no acceleration term that path overshoots 2.4–2.9 mm when it takes 1.0–1.2 s (≤ 0.2 mm in 0.6–0.9 s); omit `vel=` for point-to-point moves. A goal that jumps while it carries a velocity still steps the torque: keep `vel=` paths continuous. History: a first design scaled the spring (Khatib's velocity saturation); a review found it capped the static restoring force at `v_max·D` (≈ 67 N at 1500 N/m). A second design fed the chase velocity forward as `v_ref`; a review found every long move overshot by ≈ v/(e·ω) ≈ 1 cm and the fingertips (≈ 9.8 mm below the ee) hit the table at z = 0.02. A third set `v_ref = 0` only while chasing; a review found that a feed-forward path faster than the limit then stepped a joint torque by ≈ 40 N·m in one tick when the reference caught up (now ≤ 0.8 N·m per tick at 500 Hz on 0.3 m paths in 0.6–1.2 s, tested).
- **Handoffs (T08, T09).** `qfrc_bias` covers the arm and the hand, not a held object, so a payload sags the ee by `m·g/K` (6.5 mm for 1 kg at 1500 N/m, from the measured `F/K` law); T08's heft probe can read it, and its lifts must allow for it. `hold_current` while the ee moves overshoots the held point by ≈ 2 cm per m/s (measured over 6 moves: 4.1–5.2 mm at ≈ 0.25 m/s, 6.8–10.1 mm at 0.44–0.50 m/s, then < 0.01 mm), so T09's emergency hold must allow for that. After a stall on an unreachable goal the reference sits on that goal; call `hold_current` before the next `set_target`, or the reference walks back at the speed limit.
- **Orientation.** `TOP_DOWN_ROT`: the hand z axis points down and the fingers close along world x (the menagerie `home` orientation). `set_target(pos, yaw=…)` turns it about world z; `rot=` sets any orientation. Joint 7 (±2.8973 rad) bounds the yaw, asymmetrically: at yaw +π/2 joint 7 has ≈ 0.05 rad left at some corners, at −π/2 ≈ 1.6 rad. `set_target(yaw=…)` accepts `|yaw| ≤ π/2 + 0.05` (`GRASP_YAW_MARGIN`): every box corner is reached there (worst 4.0 mm, 2.9° after 2 s, joint 7 ≥ 0.013 rad from its limit). Yaw 2.0 (possible only through `rot=`) runs joint 7 into its limit at the four y = −0.3 corners (up to 16 mm / 13° off after 2 s). The reference turns the shortest way, so `set_target` also rejects a yaw more than π from the reference's yaw: that turn passes yaw ±π and stalls at the joint-7 limit. A review found the case: an external twist left the hand at yaw −1.61, and a target of +1.56 ended 34 mm / 26° off after 4 s (`at_joint_limit` set for 1223 of 2000 ticks). `wrap_grasp_yaw(yaw)` maps a grasp yaw into [−π/2, π/2) using the parallel gripper's half-turn symmetry; `wrap_grasp_yaw(yaw, near=controller.ee_yaw())` picks the equivalent yaw in the band closest to the current one (−1.58 in that case, a 0.03 rad turn). A turn of θ takes θ / 1.5 s at the default limit (2.1 s for ≈ π).
- **Reach.** `WORKSPACE_LOW/HIGH` (x 0.30–0.65, y ±0.30, z 0.02–0.45 m; z 0.02 is a 4 cm cube's center) is the box the reach tests sample. From `home`, the 8 box corners end ≤ 3.1 mm and ≤ 2.4° from the target after 2 s at 500, 250, and 100 Hz, and 10 random targets ≤ 0.14 mm and ≤ 0.09° (acceptance: 1 cm, 5°). Slow tests chain 30 random targets with yaw in ±π/2 for two seeds; each move gets 2 s plus its turn time.
- **Loop.** `control_step(model, data, controllers, substeps)` runs `mj_step1` (kinematics, `M`, `qfrc_bias` at the current state), calls each controller's `apply()`, then `mj_step2`, and holds the commands for the remaining substeps. The controller's `control_hz` sets the reference step, so `control_step` rejects a controller whose `control_hz` does not match the tick rate (a 500 Hz controller ticked at 100 Hz by hand ended 134 mm from the (0.65, 0.3, 0.45) corner after 2 s). It also copies `controllers` into a tuple (a generator works) and rejects a controller listed twice (its reference would advance twice). `control_hz` and `gains` are read-only (`set_gains` swaps gains). It rejects two controllers that write the same actuator, for controllers that list their `actuators` (the OSC and the `Gripper` do). MuJoCo hides bad numbers (non-finite, or beyond its 1e10 limit) behind a logged warning (also appended to `MUJOCO_LOG.TXT` in the working directory, git-ignored): a bad `qpos`, `qvel`, or `qacc` resets `data` to the model defaults, and a bad `ctrl` stays in `ctrl` while every actuator applies zero force. `control_step` reads those four warning counters at entry, after `mj_step1`, and after the physics steps (which end with `mj_checkPos` / `mj_checkVel`), raises `FloatingPointError` if one is set, and zeroes them when it raises (a reset leaves the counter that fired at 1, so comparing counts missed a second divergence). Values below 1e10 that are still absurd are T09's job. `substeps_for(model, control_hz)` rejects a rate below 100 Hz or one that does not divide 500 Hz into 1–1000 substeps. After a tick, `data`'s kinematics still describe the state before the last physics step (≈ 1 mm behind at 0.5 m/s); run `mj_forward` before reading poses.
- **Singularities.** `Λ⁻¹` gets a 1e-9 Tikhonov term (its smallest eigenvalue is ≥ 0.01 at reach endpoints over the box with `|yaw| ≤ π/2`, so the effect is < 1e-7). At an exact singularity (arm straight up, `q = 0`) the torques stay finite and inside the limits, and the arm recovers to box targets. From other near-singular starts it can stall: with the arm stretched and leaning (q2 = q6 = 0.3, elbow at its limit) it stopped 125 mm short of (0.5, 0, 0.3) after 4 s. The T09 safety supervisor must keep the arm away from such poses.

**Gymnasium env (T07).** `PandaVFAEnv(task, control, object_class, render_mode)`, registered as `vfa/PandaReach-v0`, `vfa/PandaLift-v0`, `vfa/PandaPickPlace-v0`. Hidden object properties are never in the observation. They appear in `info["privileged"]` for logging and scoring only.

| Task | Success | Failure (terminate) | Truncate | Reward (shaped) |
|---|---|---|---|---|
| Reach | EE within 1 cm of target for 0.2 s | `SAFETY_STOP` | 4 s | −‖ee − target‖ |
| Lift | cube ≥ 10 cm above table, held 1 s, not broken | `BREAK`, `DROP` after lift-off, `SAFETY_STOP` | 10 s | approach distance → grasp bonus → height progress → success bonus |
| Pick-and-place | cube within 2 cm of target, released, upright, not broken | `BREAK`, `PLACE_BREAK`, `DROP`, `TOPPLE`, `SAFETY_STOP` | 15 s | lift terms + transport distance + place bonus |

---

## 4. Learning problem design — why memory and force must matter

If one fixed set of skill parameters works for every object, memory is pointless. The benchmark therefore draws objects from **hidden classes with conflicting force requirements**. All cubes look identical (same color; sizes overlap across classes), so vision alone cannot identify the class.

Initial class table (T04 implements it; **T05 validated it in physics without changes**, `docs/FEASIBILITY.md`; T10 freezes it in `configs/objects.toml`):

| Class | Mass (kg) | Friction μ | Contact softness | Break force (N, summed over both fingers) | What goes wrong with a wrong θ |
|---|---|---|---|---|---|
| `light_rigid` | 0.05–0.15 | 0.8–1.0 | stiff | none | almost nothing — the easy class |
| `heavy_rigid` | 0.6–1.0 | 0.7–1.0 | stiff | none | slips if grip is low or lift is fast |
| `slippery` | 0.15–0.35 | 0.15–0.30 | stiff | none | needs high grip for its mass |
| `fragile_soft` | 0.05–0.20 | 0.6–0.9 | soft | 6–12 | breaks if grip is high **or** closing is fast (force overshoot) |
| `heavy_slippery` | 0.5–0.8 | 0.25–0.40 | stiff | none | needs the highest grip — conflicts with `fragile_soft` |

Rough required grip with two finger contacts and lift acceleration `a`: `F_min ≈ m(g + a) / (2μ)` per finger, so the summed grip is `2·F_min ≈ m(g + a)/μ`. Example: `heavy_slippery` needs up to ≈ 17 N per finger, and even its easiest cube needs a summed 12.3 N at `a = 0` (0.5 kg, μ 0.40); `fragile_soft` breaks at a summed 6–12 N (3–6 N per finger). **No single grip force satisfies both.** T05 measured it in physics: the success windows (≥ 80 % of 5 cubes, same at every closing speed) are `light_rigid` 1–30 N, `heavy_rigid` 8–30 N, `slippery` 6–30 N, `fragile_soft` 1–3 N, `heavy_slippery` 16–30 N per finger. The 5-cube edges are coarse: 6 N holds only 55 % of all `slippery` cubes (every one from 12.1 N), and 1 N holds 62 % of `fragile_soft`. T10 proves it statistically (`best_fixed` ≤ 75 % success@1). T05 handoff: the best single (force, speed) pooled over the five classes already reaches 80 % (16 N; it fails only `fragile_soft`), so T10 must widen the conflict.

As built in T04 (`vfa/sim/objects.py`, ranges in `configs/objects.toml`):

- **Classes.** `load_object_config()` reads the file into `ObjectClass`es in file order. Every section and key is required and unknown keys are errors, so the file alone defines the objects (T10 hashes it). The loader also rejects ranges that cannot fit the default scene: a pose box that puts the largest cube off the table or into the base keep-out (the four corners and the box point nearest the base), and a softness time constant below 2 × timestep. `sample_object(cls, rng)` draws uniformly in a fixed order, visible fields first (size, x, y, yaw), then hidden (mass, μ, time constant, break force); a golden test pins it. So a seed fixes the cube, and one seed gives every class the same visible cube. Hidden fields (`HIDDEN_FIELDS`): class name, mass, friction triple, `solref`, break force (`None` = unbreakable). Visible fields (`VISIBLE_FIELDS`): size and the pose `x, y, yaw`. Every class shares one size range (4–6 cm), so size carries no class information. The pose range is x 0.45–0.60, y ±0.15, yaw ±π/4 (a cube looks the same after a quarter turn). `ObjectConfig` pickles, so `--workers` processes can receive it.
- **Softness.** `stiff` = `solref` time constant 0.015–0.025 (around the scene default 0.02), `soft` = 0.04–0.08; dampratio 1. Static sink per side, floating 5 cm cube of 0.1 kg, force motor, 8 N per finger, after 1 s:

  | time constant | μ 0.2 | μ 0.6 | μ 1.0 | elliptic cone (any μ) |
  |---|---|---|---|---|
  | 0.02 | 0.004 mm | 0.047 mm | 0.185 mm | 0.265 mm |
  | 0.04 | 0.016 mm | 0.182 mm | 0.527 mm | 0.658 mm |
  | 0.08 | 0.068 mm | 0.521 mm | 1.392 mm | 2.004 mm |

  The sink also falls with the cube mass (MuJoCo scales contact stiffness by the effective mass). Same setup, 0.05 / 0.1 / 0.2 / 1.0 kg: 0.304 / 0.185 / 0.117 / 0.059 mm at 0.02 and μ 1.0 (elliptic: 0.411 / 0.265 / 0.172 / 0.089 mm); 0.299 / 0.182 / 0.115 / 0.058 mm at 0.04 and μ 0.6 (elliptic: 0.931 / 0.658 / 0.504 / 0.315 mm). The squeeze overshoot (peak summed force as the monitor reads it, after `control_step`, over the summed command `2f`; from 4 mm total clearance, `close(size + 0.004)`; μ 1, 0.1 kg, `f` = 2, 4, 8 N per finger) is 1.36–1.67× at 0.02 and 1.04–1.11× at 0.04–0.08, on 4 and 6 cm cubes.
- **T05 handoff (contact).** T05's answers: (1) elliptic cones removed the μ coupling of the resting sink (it now depends on the softness alone, below); (2) with a closing speed ≤ 0.5 m/s the median overshoot on `fragile_soft` is at most 1.27× for 2–5 N in the speed study (1.275 in the map's 2 N cell, 1.66× at 1 N, where the impact weighs more), and a fragile cube breaks long before a punch-through force; (3) the closing speed now shapes the force profile (§3); (4) the scene gripper is a force motor. The T04 measurements: (1) The static contact stiffness depends on μ and on the mass as much as on `solref`. With the scene's pyramidal friction cones a "stiff" cube can feel softer than a "soft" one: 0.275 vs 0.182 mm per side for (0.025, μ 1.0) vs (0.04, μ 0.6) at 0.1 kg, and 0.304 vs 0.115 mm for a `light_rigid`-like (0.05 kg, 0.02, μ 1.0) vs a `fragile_soft`-like cube (0.2 kg, 0.04, μ 0.6). Elliptic cones (`opt.cone`) remove only the μ coupling (table above); the mass coupling stays. The "closure-force slope → stiffness → fragility" probe signal therefore needs `solref` calibrated per (μ, mass), or a probe that reads the slope together with the heft load. (2) During the impact on a soft cube the 15 g fingers sink far deeper than the static sink before they spring back: ≈ 6.4 mm per side at 8 N per finger on a 5–6 cm, 0.1 kg, μ 0.6–0.9 cube at 0.04, and 13.1 mm on a 4 cm, 0.1 kg, μ 0.6 cube at 0.08. At 17 N per finger (`heavy_slippery` grip) on that 4 cm cube they pass through it for every tested mass and μ (0.05–0.2 kg, μ 0.6–0.9: minimum width −2.6 to −3.1 mm, peak summed force 7.8–49× the summed command). (3) On the first contact tick the summed force already reaches 0.81–0.93× the command for a squeeze to 1.5× the threshold, and 0.74–0.91× for 0.6× (`fragile_soft` seeds 0–5), so the rule cannot wait for a ramp. (4) The menagerie servo (static cap 25·width N per finger, §3) cannot break a `fragile_soft` cube: a full close from open on a floating soft cube (4–6 cm, 0.04–0.08, 0.05–0.2 kg, μ 0.6–0.9) peaks at 1.97–3.19 N summed (24 runs), below the 6 N minimum. On stiff cubes the closing impact reaches 3.45–8.48 N (4–6 cm, 0.015–0.025, 0.05–1.0 kg, 16 runs). The break tests use a test-local force motor, as in T03.
- **`apply_object(model, data, spec)`** writes size, mass, friction, and `solref` through the T02 hooks, paints the cube `OBJECT_RGBA` (one color for every class), and `place_cube`s it at its pose. It validates the footprint and the `solref` time constant (following the model's `refsafe` setting) first, so a rejected spec changes nothing. Mass must be 0.01–100 kg: MuJoCo scales contact stiffness by the effective mass, so under a 2 N press (1 s; default 5 cm cube, `solref` 0.02, μ 1) a 1 g cube sinks 6.7 mm and a 0.1 g cube falls through the table, without an error; a 10 g cube sinks 0.8 mm. ≈ 1.0 ms per call. A test proves that after three classes in a row every numeric model array equals a fresh compile of the last one.
- **Break rule.** `BreakMonitor.update()` (call it after every `control_step`) sums `finger_cube_forces` (left + right). The first tick the sum exceeds `break_force` sets `broken` (sticky), records `break_time` and `break_load`, logs `[S1] BREAK`, and paints the cube `BROKEN_RGBA` (red); the caller ends the attempt with `BREAK`. `apply_object` restores the color. The forces describe the start of the last physics step, so `break_time` (`data.time`) is up to one step late. With `substeps > 1` the monitor sees only the last substep of each tick. The threshold is on the **sum**, so T03's per-finger squeeze peaks double: on T03's stiff 5 cm, 0.1 kg cube (`solref` 0.02, μ 1, floating, force motor, 4 mm total clearance) a 4 N squeeze (T03: 6.23 N peak per finger, read after `mj_forward`) peaks at 12.02 N summed as the monitor reads it (12.41 N after `mj_forward`).
- **Perception.** `observe_cube(model, data, noise, rng)` returns the true center and yaw (wrapped into [−π/4, π/4)) plus Gaussian noise (`[perception]`, σ = 0 in benchmark v1); the size is exact. The cube's `solref` and μ also govern its table contact, so a settled cube sinks into the table depending on the class: with T05's elliptic cones 0.08–0.14 mm for stiff and 0.46–0.75 mm for soft cubes (seeds 0–2; T04's pyramidal cones: 0.002–0.41 mm). A height less than 5 mm below the rest height `size/2` (`REST_SINK_BAND`) therefore reads as `size/2`; a lifted cube, or one that fell off the table, reads its true height. After a 0.5 s settle, one seed's cube observes the same for all five classes (x, y within 1 µm, z exact).

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
  control/                # osc.py, gripper.py, throughput.py (T03 timing CLI), safety.py
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
| D11 | Elliptic friction cones, `impratio = 10` (T05) | pyramidal cones let a held cube creep 2.9–9.4 mm at 1.1× the needed grip; elliptic cones make `F_min` a sharp hold/drop threshold, at no extra cost |
| D12 | Force-motor gripper with a closing speed (default 0.5 m/s) and an end stop; no finger-finger contacts (T05) | the servo caps the squeeze at 25·width N; the speed shapes the force profile that breaks fragile cubes; the uncapped T03 law drives the fingers through soft cubes; the stop and the exclusion keep a squeeze on nothing from jamming |
