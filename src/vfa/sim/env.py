"""Gymnasium environment around the Panda scene: reach, lift, pick-and-place (T07).

``PandaVFAEnv(task, control, object_class, render_mode)`` wraps the T02–T06 stack
(scene, object classes, OSC, gripper, sensors, events, recorder) in the standard
``reset`` / ``step`` / ``render`` interface (ARCHITECTURE §3). Importing this
module registers ``vfa/PandaReach-v0``, ``vfa/PandaLift-v0``, and
``vfa/PandaPickPlace-v0`` (``import vfa`` alone does not, to keep Gymnasium's
import off every CLI call); ``gymnasium.make("vfa.sim.env:vfa/PandaLift-v0")``
imports it on demand.

**Rates.** Physics and the control law run at ``CONTROL_HZ`` (500 Hz); one
``step`` runs ``CONTROL_HZ / step_hz`` control ticks (default ``step_hz`` 50 Hz,
10 ticks). Every tick runs the break rule, the sensors, the event detector, and
the recorder, so events and the recorded episode keep the full 500 Hz. T08
drives the env at control rate with ``step_hz=500``.

**Actions** (``Box(-1, 1)``, float32):

- ``control="osc"``: ``Box(4)``. ``a[:3]`` moves the ee target by
  ``a[:3]·MAX_EE_SPEED/step_hz`` [m], its length capped at
  ``MAX_EE_SPEED/step_hz`` (1 cm per 50 Hz step: 0.5 m/s in any direction, the
  OSC's own limit, so the OSC reference follows the target exactly). The target
  accumulates (it is the impedance spring's anchor, not an offset from the ee,
  so a held payload does not sink step after step) and is clipped to the
  target box ``TARGET_LOW``–``TARGET_HIGH`` (the OSC workspace with its near
  face at x = 0.33, T09). Inside the step it moves on a
  straight line, tick by tick, with the step's velocity fed forward (``vel=``):
  T05 measured 7.2 mm of sideways drift on a descent without it. The hand stays
  top-down, turned to the grasp yaw: 0 for reach, the observed cube yaw for
  lift and pick-and-place (the reference turns there at 1.5 rad/s in the first
  steps). ``a[3]`` is the grip: ``a[3] > 0`` squeezes each finger with
  ``a[3]·GRIP_FORCE_MAX`` N at the closing speed ``close_speed`` (default
  0.1 m/s; refer to ``Gripper.squeeze``), ``a[3] <= 0`` opens.
- ``control="torque"``: ``Box(8)``, raw actuator commands with no gravity
  compensation: ``ctrl[0:7] = a[0:7]·ARM_TORQUE_LIMITS`` [N·m] and the gripper
  ``ctrl[7] = a[7]·GRIPPER_FORCE_LIMIT`` (the tendon force [N]; **positive
  opens**, MuJoCo's sign).

**Observation** (``Dict``, float64 unless noted). Only observable signals:
``joint_pos``, ``joint_vel`` (arm, 7 each, the state after the step);
``ee_pos``, ``ee_quat`` (w, x, y, z), ``ee_vel`` (linear, angular); ``wrench``
(the tared, low-passed wrist wrench, world axes); ``load`` (the held load [N]);
``finger_force`` (the two pad forces [N]); ``finger_contact`` (``MultiBinary(2)``,
int8); ``gripper_width``; ``cube_pos``, ``cube_yaw``, ``cube_size`` (the camera
snapshot taken at reset); ``goal`` (the reach target, the lift goal 10 cm above
the cube, or the place target at the cube's rest height); and with
``control="osc"`` ``ee_target`` (the accumulated target, so the observation is
Markov). The sensor readings describe the start of the last physics step
(T06). The cube pose is a snapshot: vision localizes the cube before the grasp,
and the cube's motion in the grip (which depends on its hidden mass, friction,
and softness) stays out of the observation. Hidden properties never appear in
the observation; ``info["privileged"]`` carries them for logging and scoring.

**Info.** ``phase``; ``events``: the observable events of the step
(``CONTACT_ONSET``, ``GRASP_CLOSED``, ``SLIP``, ``DROP``, ``SAFETY_STOP``);
``result`` and ``is_success`` once the episode ends; ``t`` (since the reset
ended) and ``sim_time`` (the clock of event times and of the recorder);
``privileged``: the hidden object properties, the true cube pose, ``broken``,
and the ground-truth events ``BREAK``, ``LIFTED``, ``PLACED``
(``PRIVILEGED_EVENTS``: the true cube height and break force are not
observable). The recorder keeps every event.

**Rewards** read observable signals only, so they leak nothing the observation
does not (the success and failure checks read ground truth and end the episode).
Reach: ``-‖ee − goal‖``. Lift: an approach term ``1 − tanh(10·‖ee − cube‖)``
(1 once grasped), ``GRASP_BONUS`` while both pads touch under a squeeze, the
ee's rise since the grasp closed over ``LIFT_HEIGHT`` (clipped to [0, 1]) while
grasped, and ``SUCCESS_BONUS`` on success. Pick-and-place takes the best rise
of the grasp instead and adds a transport term ``1 − tanh(10·‖ee_xy −
goal_xy‖)`` during a carry; both stay through the set-down and the release, so
placing loses no reward. The step a ``SAFETY_STOP`` ends costs
``SAFETY_PENALTY`` plus, while the reward is negative (reach), that reward for
every step left to the time limit: tripping the safety supervisor never scores
above holding the stop pose until the time limit (the penalty does not price a
braking path; no task here trains on these rewards). A ``BREAK`` in the same
tick as a stop outranks it (the result is ``BREAK``, or ``PLACE_BREAK`` in a
pick-and-place release phase; ``emergency`` keeps the stop).

**Termination** (ARCHITECTURE §3 table). Reach: success when the ee stays
within 1 cm of the goal for 0.2 s. Lift: success when the cube's center stays ≥
10 cm above its rest height for 1 s with a pad in contact; failure on
``BREAK``, and on ``DROP`` after a lift-off: ``LIFTED`` came in this carry, or
the cube has not rested on the table since its last ``LIFTED``. A grip that
slips before the cube clears 2 cm runs on (``TIMEOUT`` unless the policy
recovers; T08 ends its own attempts). Pick-and-place: ``PLACED`` (resting flat
within 2 cm of the target, unbroken, no pad contact) is not success on its own
(T06). It is judged once, from ground truth at the cube's last touchdown: the
first table contact after an airborne spell (more than ``REST_SINK_BAND`` up)
that began with both fingers pinching it. While both fingers keep pinching it
without a break, the touchdown moves with it (a heavy cube that grazed the
table early and was dragged the rest of the way, T19); it stays where it is
once the pinch breaks. If the cube left the fingers more
than ``RELEASE_HEIGHT`` above the table before that touchdown, the result is
``DROP``; if the touchdown lies within ``PLACE_TOUCHDOWN_RADIUS`` of the
target, ``SUCCESS``; else the cube was pushed there: no success, and the
episode runs on (``PLACED`` comes once, so it cannot succeed). A nudge after
the touchdown cannot change the verdict. Pick-and-place also fails on ``BREAK``
(``PLACE_BREAK`` in or after a release phase, until a new carry starts; its break
monitor also reads the table's normal force on the cube, T19's impact rule),
``DROP`` as in lift, and ``TOPPLE`` (in or after a release phase: resting, no
pad contact, no face within 10° of flat, for 0.2 s). Every task ends on
``SAFETY_STOP``: one reported between steps (T09, through ``events.report``)
ends the episode before the next action runs. Truncation: 4 / 10 / 15 s of sim
time after the reset, result ``TIMEOUT``. The checks run every control tick,
and the step stops at the first terminal tick.

**Carries** (T06 handoff: a policy without a skill phase machine). With
``auto_carry`` (default) the env runs a small state machine on observable
signals: ``GRASP`` (after ``GRASP_CLOSED``) → ``CARRY`` (``watch_slip()``) once
both pads touch under a squeeze and the ee has risen ``CARRY_RISE`` above its
lowest point since the grasp closed. In a carry, at most ``RELEASE_HEIGHT``
above that point, a lowering (osc: the commanded target moves down, so payload
sag is not one; torque: the ee sits ``CARRY_RISE`` below the carry's highest
point and the held load stayed below ``SETDOWN_LOAD`` of the carry's reference
load, minus ``SETDOWN_NOISE_K`` σ of the estimate's noise, for ``SETDOWN_S``;
the reference is the highest level the load held for ``SETDOWN_WINDOW_S``, so
neither a load spike nor a noise dip fakes one, T19) is a set-down,
``PLACE``, and an open is a release, ``RELEASE``; both call ``unwatch_slip()``,
since the table taking the load is not a drop (T06). Higher up the carry stays
watched, so an open there makes the cube's fall a ``DROP``.
From ``PLACE`` a raise with both pads touching (osc: commanded; torque:
``CARRY_RISE`` above the set-down's lowest point) starts a new carry and an
open releases; a squeeze in ``RELEASE`` goes back to ``GRASP``, so a later
carry is watched again. ``DROP`` and the end of the grasp (both pads off for
0.1 s) also end a carry. A skill (T08) passes ``auto_carry=False``, calls the
sensor suite itself, and names its phases with ``set_phase``.

**Phases** (``info["phase"]``, the recorder's phase column): ``SETTLE`` during
the reset; ``REACH``; or ``APPROACH`` → ``GRASP`` (in contact) → ``CARRY`` →
``PLACE`` → ``RELEASE`` → ``APPROACH`` again once the grasp ends. ``PLACE`` and
``RELEASE`` are the event detector's release phases.

**Limits.** The T06 slip detector reads a jolt as a slip: a one-step sideways
target jolt with a ``light_rigid`` cube held 12 cm up fired ``SLIP`` from 0.15
m/s on for one cube (none at 0.05–0.1 m/s; other cubes and directions from
0.3–0.4 m/s or not up to 0.5 m/s), never on ``heavy_rigid`` up to 0.5 m/s.
``SLIP`` never ends an episode; smooth policies avoid it (the tests ramp at 1
m/s²). Until the robot first touches the cube the observation is bit-identical
across classes: MuJoCo's solver warmstart is off until then (ARCHITECTURE D14).
With it, its warmstart choice weighs the whole constraint problem, the cube's
table contacts included, and the moving arm's readings differed between classes
by round-off (about 1e-13 N in the wrench over 400 random steps). Contact-free
ticks cost the same either way (≈ 116 µs); in scripted lifts of the five
classes warmstart saves a quarter on average (128 vs 168 µs per tick; the
saving depends on the class), so it comes back at the first touch. Whether an
observable ``DROP`` ends the episode depends on ground truth (did the cube
clear 2 cm?), as the §3 table defines it. A ``reset()`` before the episode ends
drops its recording.

**Safety** (T09, ``vfa.control.safety``). With ``safety`` set (the default
``DEFAULT_SAFETY``), a ``SafetySupervisor`` checks every control tick after the
reset: the commanded ee target (osc) and the measured ee against the workspace
box, the tared wrist force and torque, the ee speed, the arm joint margins, and
sustained torque saturation (osc: the OSC clipped its law; torque: an arm
command at its motor limit). On the first violation the env engages the
emergency mode: the OSC holds the current pose (in torque control the arm
motors get the gravity compensation torques), the gripper keeps its last
command, ``emergency`` holds the ``SafetyViolation``, and ``SAFETY_STOP`` (info:
its reason and values) ends the episode in that tick. The observation's
``ee_target`` stays the last commanded target (inside its box); the hold is
``osc.target``. ``safety=None`` turns the
supervisor off (random raw torques, for example, trip it within 1–5 steps at
50 Hz).

**Reset.** ``reset(seed, options)`` puts the arm at ``START_ARM_QPOS`` (the ee at
``START_POS``, top-down, at rest, inside the workspace), draws the cube
(``options["object"]`` an ``ObjectSpec``, else ``options["object_class"]`` or
the constructor's ``object_class``, else a class drawn uniformly), takes the
camera snapshot (the cube sits exactly at its sampled pose: its
class-dependent sink into the table comes later), then holds the pose for
``SETTLE_S`` and tares the wrist over its last ``TARE_S`` (T06 handoff: before
the tare the wrench is NaN). Everything random comes from the env's
``np_random``, one independent stream each for the object, the reach goal,
the sensor noise, and the camera noise: same seed, same episode, whatever ran
before. The env owns an ``EpisodeRecorder`` per episode (every
``record_stride``-th tick, from the settle on); the finished ``Episode`` is
``env.unwrapped.episode`` after the last step.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from vfa.control.gripper import MAX_CLOSE_SPEED, MIN_CLOSE_SPEED, Gripper
from vfa.control.osc import (
    CONTROL_HZ,
    WORKSPACE_HIGH,
    WORKSPACE_LOW,
    OperationalSpaceController,
    control_step,
    substeps_for,
)
from vfa.control.safety import SafetyLimits, SafetySupervisor, SafetyViolation
from vfa.sim import objects as OB
from vfa.sim import scene as S
from vfa.sim.recorder import RESULTS, Episode, EpisodeRecorder
from vfa.sim.sensors import (
    Event,
    EventConfig,
    EventDetector,
    EventRecord,
    PrivilegedState,
    SensorConfig,
    SensorFrame,
    SensorSuite,
    read_privileged,
)

TASKS: tuple[str, ...] = ("reach", "lift", "pick_place")
CONTROLS: tuple[str, ...] = ("osc", "torque")
ENV_IDS: dict[str, str] = {
    "vfa/PandaReach-v0": "reach",
    "vfa/PandaLift-v0": "lift",
    "vfa/PandaPickPlace-v0": "pick_place",
}

STEP_HZ = 50.0  # default env step rate (ARCHITECTURE §2)
TIME_LIMIT_S: dict[str, float] = {"reach": 4.0, "lift": 10.0, "pick_place": 15.0}
SETTLE_S = 0.2  # reset: hold home this long; its last TARE_S is the tare window
TARE_S = 0.1
MAX_EE_SPEED = 0.5  # ee target speed at |a[:3]| = 1 [m/s] (the OSC's own speed limit)
# The ee target stays in the OSC workspace (T03: every corner reached to ≤ 3.1 mm).
# Its floor (0.02 m, a 4 cm cube's center) keeps the fingertips about 1 cm above the
# table at rest; a full-speed descent into a far corner overshoots it about 1 cm and
# touches. Above it the far corners stall: (0.65, ±0.3) ends 10.6 mm short at the home
# height z = 0.521 and 17.7 mm short at z = 0.55 (3 s), so the episode does not start
# at home. Its near face is x = 0.33, not the workspace's 0.30 (T09): around (0.30, 0,
# 0.08-0.18) joint 4 comes within the safety supervisor's 0.02 rad margin of its
# range end (it reaches the end itself without the supervisor), and at 0.32 a
# full-speed rise reversed into a descent still did (the ee lags 2 cm past the face);
# at 0.33 it kept >= 0.049 rad in a review's reversal sweep.
TARGET_LOW = np.array([0.33, WORKSPACE_LOW[1], WORKSPACE_LOW[2]])
TARGET_HIGH = np.array(WORKSPACE_HIGH, dtype=float)
# Start pose: the arm joints that hold the ee at START_POS, top-down, under the OSC
# (the rest state after a 12 s OSC move from home; it drifts 1.5e-7 mm in 1 s).
START_POS = np.array([0.55, 0.0, 0.40])
START_ARM_QPOS: tuple[float, ...] = (0.0, -0.03493, 0.0, -1.922671, 0.0, 1.887741, -0.785398)
# Reach goals: the target box with the floor at 10 cm, clear of any cube.
REACH_LOW = np.array([TARGET_LOW[0], TARGET_LOW[1], 0.10])
REACH_HIGH = np.array(WORKSPACE_HIGH, dtype=float)
REACH_TOLERANCE = 0.01  # m
REACH_HOLD_S = 0.2
LIFT_HEIGHT = 0.10  # the cube's center above its rest height [m]
LIFT_HOLD_S = 1.0
GRIP_FORCE_MAX = 30.0  # squeeze per finger at a[3] = 1 [N]: the top of the T05 map
DEFAULT_CLOSE_SPEED = 0.1  # m/s (scripts/record_episode.py)
CARRY_RISE = 0.003  # ee rise since the grasp closed that starts a carry [m]
# An open, or a lowering, at most this far above the grasp height is a release or a
# set-down [m]; a cube that left the fingers higher than this above the table fell.
RELEASE_HEIGHT = 0.02
# A place counts only if the cube was carried down near the target: its last
# touchdown (the first table contact after an airborne spell that began with both
# fingers pinching it, more than REST_SINK_BAND up) lies within this of it.
PLACE_TOUCHDOWN_RADIUS = 0.04  # m
# Torque mode: a set-down also needs the table to take the load: the held load stays
# below this fraction of the carry's reference load for SETDOWN_S. The reference is the
# highest level the load held for SETDOWN_WINDOW_S in this carry, so a spike (a fast lift
# start slipped a heavy cube in the grip: 1.4-2.8x the static load) cannot set it (T19);
# payload sag lowers the ee 3+ mm but keeps the load.
# With sensor noise the load must also stay SETDOWN_NOISE_K σ of the filtered estimate
# below that line (the slip detector's own margin): at 5 % below a 0.5 N reference, a
# noise dip of two ticks set a cube down 1.6 cm in the air in 3 of 10 carries (T19).
SETDOWN_LOAD = 0.95
SETDOWN_S = 0.004  # two control ticks: shorter than the slip detector's 6 ms debounce
SETDOWN_WINDOW_S = 0.1
SETDOWN_NOISE_K = 4.0
# MuJoCo's solver warmstart, off until the robot first touches the cube (refer to reset).
_WARMSTART = int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
# Ground-truth events: for scoring and logs, reported under info["privileged"].
PRIVILEGED_EVENTS = frozenset({Event.BREAK, Event.LIFTED, Event.PLACED})
TOPPLE_REST_S = 0.2
GRASP_BONUS = 0.25
DEFAULT_SAFETY = SafetyLimits()  # the env's safety limits unless safety= says otherwise
SUCCESS_BONUS = 10.0
SAFETY_PENALTY = 10.0  # subtracted when a SAFETY_STOP ends the episode (module doc)
RENDER_SIZE = (480, 640)  # (height, width) of render()
_EPS_T = 1e-9


class PandaVFAEnv(gym.Env):
    """Panda reach / lift / pick-and-place with force sensing (refer to the module doc)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": int(STEP_HZ)}

    def __init__(
        self,
        task: str = "lift",
        control: str = "osc",
        object_class: str | None = None,
        render_mode: str | None = None,
        *,
        step_hz: float = STEP_HZ,
        close_speed: float = DEFAULT_CLOSE_SPEED,
        object_config: OB.ObjectConfig | None = None,
        sensor_config: SensorConfig | None = None,
        event_config: EventConfig | None = None,
        auto_carry: bool = True,
        record: bool = True,
        record_stride: int = 1,
        render_size: tuple[int, int] = RENDER_SIZE,
        safety: SafetyLimits | None = DEFAULT_SAFETY,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        if control not in CONTROLS:
            raise ValueError(f"control must be one of {CONTROLS}, got {control!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be None or 'rgb_array', got {render_mode!r}")
        self.task = task
        self.control = control
        self.render_mode = render_mode
        self.object_config = object_config if object_config is not None else OB.load_object_config()
        if not isinstance(self.object_config, OB.ObjectConfig):
            raise TypeError("object_config must be an ObjectConfig")
        if object_class is not None and object_class not in self.object_config.names:
            raise ValueError(
                f"object_class must be None or one of {self.object_config.names}, "
                f"got {object_class!r}"
            )
        self.object_class = object_class
        self.sensor_config = sensor_config if sensor_config is not None else SensorConfig()
        self.event_config = event_config if event_config is not None else EventConfig()
        self.auto_carry = bool(auto_carry)
        self.record = bool(record)
        if isinstance(record_stride, bool) or not isinstance(record_stride, int):
            raise ValueError(f"record_stride must be an int >= 1, got {record_stride!r}")
        if record_stride < 1:
            raise ValueError(f"record_stride must be an int >= 1, got {record_stride!r}")
        self.record_stride = record_stride
        self.close_speed = close_speed
        if safety is not None and not isinstance(safety, SafetyLimits):
            raise TypeError(f"safety must be a SafetyLimits or None, got {safety!r}")
        self.safety = safety

        self.model = S.build_scene()
        self.data = mujoco.MjData(self.model)
        self.ids = S.scene_ids(self.model)
        self.dt = 1.0 / CONTROL_HZ
        if substeps_for(self.model, CONTROL_HZ) != 1:
            raise ValueError("the env runs one physics step per control tick")
        ticks = CONTROL_HZ / float(step_hz)
        if not (math.isfinite(ticks) and ticks >= 1 and abs(ticks - round(ticks)) < 1e-9):
            raise ValueError(
                f"step_hz must divide the {CONTROL_HZ} Hz control rate, got {step_hz!r}"
            )
        self.ticks_per_step = round(ticks)
        self.step_hz = CONTROL_HZ / self.ticks_per_step
        self.metadata = {**self.metadata, "render_fps": int(round(self.step_hz))}
        self.max_step = MAX_EE_SPEED / self.step_hz
        self.time_limit = TIME_LIMIT_S[task]
        self._arm_limits = np.abs(self.model.actuator_ctrlrange[self.ids.arm_actuators, 1]).copy()
        self._grip_limit = float(abs(self.model.actuator_ctrlrange[self.ids.gripper_actuator, 1]))
        h, w = render_size
        if not (isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0):
            raise ValueError(f"render_size must be two ints > 0, got {render_size!r}")
        offw, offh = S.SceneConfig().offscreen_size
        if w > offw or h > offh:
            raise ValueError(f"render_size must fit the {offw}×{offh} offscreen buffer")
        self.render_size = (h, w)
        self._renderer: mujoco.Renderer | None = None

        n_act = 4 if control == "osc" else 8
        self.action_space = spaces.Box(-1.0, 1.0, (n_act,), np.float32)
        self.observation_space = spaces.Dict(self._obs_spaces())

        # Episode state (set by reset).
        self.osc: OperationalSpaceController | None = None
        self.gripper: Gripper | None = None
        self.monitor: OB.BreakMonitor | None = None
        self.supervisor: SafetySupervisor | None = None
        self.emergency: SafetyViolation | None = None
        self.sensors: SensorSuite | None = None
        self.events: EventDetector | None = None
        self.recorder: EpisodeRecorder | None = None
        self.episode: Episode | None = None
        self.object_spec: OB.ObjectSpec | None = None
        self.observed: OB.ObservedPose | None = None
        self.goal = np.zeros(3)
        self.ee_target = np.zeros(3)
        self.grasp_yaw = 0.0
        self.result: str | None = None
        self._frame: SensorFrame | None = None
        self._priv: PrivilegedState | None = None
        self._phase_override: str | None = None
        self._auto_phase = "SETTLE"
        self._settling = False
        self._done = True

    # ------------------------------------------------------------------ spaces

    def _obs_spaces(self) -> dict[str, spaces.Space]:
        # The physical signals are unbounded: random raw torques measured wrist forces
        # of 5 kN, joint speeds of 33 rad/s, and fingers 5 cm past their joint limits.
        def box(n: int, low: object = -np.inf, high: object = np.inf) -> spaces.Box:
            return spaces.Box(np.broadcast_to(low, n), np.broadcast_to(high, n), (n,), np.float64)

        quarter = 0.25 * math.pi
        obs = {
            "joint_pos": box(7),
            "joint_vel": box(7),
            "ee_pos": box(3),
            "ee_quat": box(4, -1.0, 1.0),
            "ee_vel": box(6),
            "wrench": box(6),
            "load": box(1),
            "finger_force": box(2),
            "finger_contact": spaces.MultiBinary(2),
            "gripper_width": box(1),
            "cube_pos": box(3),
            "cube_yaw": box(1, -quarter, quarter),
            "cube_size": box(1, *S.CUBE_SIZE_RANGE),
            "goal": box(3),
        }
        if self.control == "osc":
            obs["ee_target"] = box(3, TARGET_LOW, TARGET_HIGH)
        return obs

    # --------------------------------------------------------------- settings

    @property
    def close_speed(self) -> float:
        """Closing speed [m/s] of the grip squeeze (``control="osc"``); T08 sets θ's."""
        return self._close_speed

    @close_speed.setter
    def close_speed(self, value: float) -> None:
        speed = S._real("close_speed", value)
        if not MIN_CLOSE_SPEED <= speed <= MAX_CLOSE_SPEED:
            raise ValueError(
                f"close_speed must be in [{MIN_CLOSE_SPEED}, {MAX_CLOSE_SPEED}] m/s, got {value!r}"
            )
        self._close_speed = speed

    def set_phase(self, phase: str | None) -> None:
        """Name the phase for the recorder and the event detector (a skill's state).

        ``None`` returns to the env's own phase labels. A reset clears it.
        """
        if phase is not None and (not isinstance(phase, str) or not phase):
            raise ValueError(f"phase must be None or a non-empty str, got {phase!r}")
        self._phase_override = None if phase is None else str.__str__(phase)

    @property
    def phase(self) -> str:
        """The current phase label."""
        return self._phase_override or self._auto_phase

    # ------------------------------------------------------------------ reset

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})
        unknown = set(options) - {"object", "object_class"}
        if unknown:
            raise ValueError(f"unknown reset options {sorted(unknown)}")
        # One independent stream per use, drawn first: the class and the hidden draws
        # (a fragile class draws one more number) never shift the goal or the noise.
        object_rng, goal_rng, sensor_rng, camera_rng = (
            np.random.default_rng(int(s)) for s in self.np_random.integers(2**63, size=4)
        )
        spec = options.get("object")
        if spec is not None:
            if not isinstance(spec, OB.ObjectSpec):
                raise TypeError(f"options['object'] must be an ObjectSpec, got {spec!r}")
        else:
            name = options.get("object_class", self.object_class)
            names = self.object_config.names
            if name is None:
                name = names[int(object_rng.integers(len(names)))]
            elif name not in names:
                raise ValueError(f"object_class must be one of {names}, got {name!r}")
            spec = OB.sample_object(self.object_config.classes[name], object_rng)
        goal = goal_rng.uniform(REACH_LOW, REACH_HIGH) if self.task == "reach" else None

        model, data, ids = self.model, self.data, self.ids
        S.reset_to_home(model, data)
        data.qpos[ids.arm_qpos] = START_ARM_QPOS
        OB.apply_object(model, data, spec)  # ends with mj_forward
        self.object_spec = spec
        self.osc = OperationalSpaceController(model, data, ids=ids)
        self.gripper = Gripper(model, data, ids)
        self.gripper.open()
        self.monitor = OB.BreakMonitor(model, data, spec, ids, impact=self.task == "pick_place")
        self.sensors = SensorSuite(model, data, ids, self.sensor_config, rng=sensor_rng)
        self.events = EventDetector(self.event_config)
        self.supervisor = None
        if self.safety is not None:
            self.supervisor = SafetySupervisor(model, data, ids, self.safety, self.dt)
        self.emergency = None
        self.episode = None
        self.result = None
        self._phase_override = None
        self._auto_phase = "SETTLE"
        self._grasp_z: float | None = None  # lowest ee z since GRASP_CLOSED
        self._best_rise = 0.0  # highest ee rise above _grasp_z in this grasp
        self._placing = False  # a set-down: lowered near the table under a squeeze
        self._released = False  # opened near the table (until a squeeze again)
        self._place_started = False  # a release phase (PLACE, RELEASE) was entered
        self._lifted_carry: int | None = None  # the carry (watch_id) of the last LIFTED
        self._contact_rise = 0.0  # true cube rise at its last finger contact [m]
        self._target_vz = 0.0  # the step's commanded target z speed (osc) [m/s]
        self._carry_peak = 0.0  # highest ee rise in the current carry (torque set-down)
        self._place_low = 0.0  # lowest ee rise in the current set-down (torque re-lift)
        self._carry_load = (
            0.0  # the carry's reference load (torque set-down, refer to SETDOWN_LOAD)
        )
        self._load_window: deque[float] = deque()  # the carry's loads of the last SETDOWN_WINDOW_S
        self._unloaded = 0  # ticks in a row with the load under the set-down line
        self._airborne = False  # a pinched cube left the table (above REST_SINK_BAND) ...
        self._touchdown_xy: np.ndarray | None = None  # ... and came down here,
        self._touchdown_release = 0.0  # ... having left the fingers this high [m]
        self._dragged = False  # pinched without a break since the touchdown: it follows
        self._place_judged = False  # PLACED was scored
        self._delivered = 0  # events in self.events.events already reported
        # No solver warmstart until the robot first touches the cube (module doc, D14).
        self.model.opt.disableflags |= _WARMSTART
        self._untouched = True
        self._near_since: float | None = None
        self._high_since: float | None = None
        self._tilted_since: float | None = None
        self._squeezing = False
        hidden = spec.hidden()
        self._hidden = hidden
        noise = self.sensor_config.noise
        meta = {
            "task": self.task,
            "control": self.control,
            "class": spec.class_name,
            "seed": seed,
            "control_hz": CONTROL_HZ,
            "step_hz": self.step_hz,
            "noise": {
                "force": noise.force,
                "torque": noise.torque,
                "accel": noise.accel,
                "touch": noise.touch,
            },
            "size": spec.size,
            "hidden": hidden,
        }
        self.recorder = (
            EpisodeRecorder(meta=meta, stride=self.record_stride) if self.record else None
        )

        # The camera snapshot comes before the settle: the cube sits exactly at its
        # sampled pose, so its class-dependent sink cannot move what the camera reports.
        seen = OB.observe_cube(model, data, self.object_config.perception, camera_rng, ids)
        self.observed = seen
        # Settle at home, then tare the wrist with the hand at rest (T06 handoff).
        settle = round(SETTLE_S / self.dt)
        tare = round(TARE_S / self.dt)
        self._settling = True
        for k in range(settle):
            if k == settle - 1 - tare:
                self.sensors.begin_tare()
            elif k == settle - 1:
                self.sensors.end_tare()  # the last settle tick reads a tared frame
            self._tick_physics(self.osc, self.gripper)
            self._sense()
        self._settling = False

        if self.task == "reach":
            self.goal = np.asarray(goal, dtype=float)
            self.grasp_yaw = 0.0
        elif self.task == "lift":
            self.goal = np.array([seen.x, seen.y, seen.z + LIFT_HEIGHT])
            self.grasp_yaw = seen.yaw
        else:
            place = model.site_pos[ids.place_site]
            self.goal = np.array([place[0], place[1], 0.5 * seen.size])
            self.grasp_yaw = seen.yaw
        self.ee_target = np.array(self.osc.target.pos, dtype=float)
        self.osc.set_target(self.ee_target, yaw=self.grasp_yaw)
        if self.recorder is not None:
            self.recorder.context.update(
                {
                    "cube": {
                        "x": seen.x,
                        "y": seen.y,
                        "z": seen.z,
                        "yaw": seen.yaw,
                        "size": seen.size,
                    },
                    "arm_qpos": data.qpos[ids.arm_qpos].tolist(),
                    "goal": self.goal.tolist(),
                }
            )
        self._auto_phase = "REACH" if self.task == "reach" else "APPROACH"
        self._t0 = float(data.time)
        self._done = False
        return self._obs(), self._info([])

    # ------------------------------------------------------------------- step

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("the episode is over (or never started): call reset()")
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}, got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"action must be finite, got {a!r}")
        a = np.clip(a, -1.0, 1.0)

        # An event reported between steps (T09's SAFETY_STOP) ends the episode before
        # the new action runs.
        new_events = self._collect_events()
        stops = [e for e in new_events if e.name is Event.SAFETY_STOP]
        if stops:
            if self.emergency is None:  # reported from outside: engage the hold too
                self._engage(_external_violation(stops[0]))
            self.result = "SAFETY_STOP"
            self._done = True
            self._finish()
            # No tick ran: the step earns only the stop's cost.
            return self._obs(), -self._stop_cost(), True, False, self._info(new_events)

        n = self.ticks_per_step
        if self.control == "osc":
            start = self.ee_target.copy()
            delta = a[:3] * self.max_step
            norm = float(np.linalg.norm(delta))
            if norm > self.max_step:  # the speed limit holds in every direction
                delta *= self.max_step / norm
            end = np.clip(start + delta, TARGET_LOW, TARGET_HIGH)
            vel = (end - start) * self.step_hz
            self._target_vz = float(vel[2])
            self._squeezing = bool(a[3] > 0.0)
            if self._squeezing:
                self.gripper.squeeze(float(a[3]) * GRIP_FORCE_MAX, speed=self.close_speed)
            else:
                self.gripper.open()
            controllers = (self.osc, self.gripper)
        else:
            ctrl = self.data.ctrl
            ctrl[self.ids.arm_actuators] = a[:7] * self._arm_limits
            ctrl[self.ids.gripper_actuator] = a[7] * self._grip_limit
            self._squeezing = bool(a[7] < 0.0)  # a negative tendon force closes
            controllers = ()

        terminated = False
        for k in range(1, n + 1):
            if self.control == "osc":
                self.ee_target = end if k == n else start + (end - start) * (k / n)
                self.osc.set_target(self.ee_target, yaw=self.grasp_yaw, vel=(*vel, 0.0, 0.0, 0.0))
            self._tick_physics(*controllers)
            events = self._sense()
            new_events.extend(events)
            result = self._check_terminal(events)
            if result is not None:
                self.result = result
                terminated = True
                break
        truncated = False
        if not terminated and self.elapsed >= self.time_limit - _EPS_T:
            self.result = "TIMEOUT"
            truncated = True
        reward = self._reward()
        if terminated and self.result == "SUCCESS" and self.task != "reach":
            reward += SUCCESS_BONUS
        elif terminated and self.result == "SAFETY_STOP":
            reward -= self._stop_cost()
        if terminated or truncated:
            self._done = True
            self._finish()
        return self._obs(), float(reward), terminated, truncated, self._info(new_events)

    def end_episode(self, result: str) -> None:
        """End the episode now with ``result`` (one of ``recorder.RESULTS``).

        For an agent that stops on its own: T08's skill ends an attempt after a
        ``DROP`` the env does not end on (no lift-off yet), a ``MISS``, or a hold
        that never succeeded. The recorder finishes as at a termination
        (``episode``), and ``step`` raises until the next ``reset``.
        """
        if self._done:
            raise RuntimeError("the episode is over (or never started): call reset()")
        if result not in RESULTS:
            raise ValueError(f"result must be one of {RESULTS}, got {result!r}")
        # Events reported since the last step reach the recorder; a pending safety
        # stop outranks the caller's reason.
        stops = [e for e in self._collect_events() if e.name is Event.SAFETY_STOP]
        if stops:
            if self.emergency is None:
                self._engage(_external_violation(stops[0]))
            result = "SAFETY_STOP"
        self.result = result
        self._done = True
        self._finish()

    @property
    def elapsed(self) -> float:
        """Sim time since the end of the reset [s]."""
        return float(self.data.time) - self._t0

    # ------------------------------------------------------------- one tick

    def _tick_physics(self, *controllers: object) -> None:
        control_step(self.model, self.data, controllers, 1)
        self.monitor.update()

    def _sense(self) -> list[EventRecord]:
        """Read the sensors, run the carry logic and the event detector, record the tick."""
        frame = self.sensors.read()
        priv = read_privileged(self.model, self.data, self.ids, self.monitor)
        if self.task != "reach" and not self._settling:
            self._track_grasp(frame)
            if self.auto_carry:
                self._carry_logic(frame)
            self._auto_phase = self._next_phase()
        if self.phase in self.event_config.release_phases:
            self._place_started = True
        elif self.sensors.slip.watching:
            self._place_started = False  # a new carry: a break now is not a place break
        self.events.update(frame, self.phase, priv)
        if self.supervisor is not None and not self._settling and self.emergency is None:
            self._supervise(frame)
        events = self._collect_events()  # with any reported from outside (T09)
        for e in events:
            if e.name is Event.LIFTED:
                self._lifted_carry = frame.watch_id if frame.watching else None
        if self.auto_carry and self.sensors.slip.watching:
            if any(e.name is Event.DROP for e in events):
                self.sensors.unwatch_slip()  # the carry is over
        forces = priv.cube_finger_force
        cube_rise = float(priv.cube_pos[2]) - priv.rest_z
        if float(np.sum(forces)) > 0.0:
            self._contact_rise = cube_rise
        pinched = bool(forces[0] > 0.0 and forces[1] > 0.0)
        contacts = self._cube_contacts()
        if contacts & 1:  # on the table
            if self._airborne:
                self._airborne = False
                self._touchdown_xy = priv.cube_pos[:2].copy()
                self._touchdown_release = self._contact_rise
                self._dragged = pinched
            elif self._dragged and pinched:  # still pinched since the touchdown: it
                self._touchdown_xy = priv.cube_pos[:2].copy()  # comes down here (T19)
        elif cube_rise > OB.REST_SINK_BAND and pinched:
            self._airborne = True  # carried: a push or a toss never starts a spell
        if not pinched:
            self._dragged = False  # the pinch broke: the touchdown stays where it is
        if self._untouched and contacts & 2:
            self.model.opt.disableflags &= ~_WARMSTART
            self._untouched = False
        if self.recorder is not None:
            self.recorder.record(frame, self.phase, priv)
        self._frame, self._priv = frame, priv
        return events

    def _supervise(self, frame: SensorFrame) -> None:
        """Run the safety checks; on a violation engage the emergency mode (module doc)."""
        arm = self.ids.arm_actuators
        if self.control == "osc":
            target = np.asarray(self.osc.target.pos)
            saturated = self.osc.saturated
        else:
            target = None
            saturated = bool(np.any(np.abs(self.data.ctrl[arm]) >= self._arm_limits))
        violation = self.supervisor.check(frame, target, saturated)
        if violation is not None:
            self._engage(violation)
            self.events.report(Event.SAFETY_STOP, frame.t, **violation.info())

    def _stop_cost(self) -> float:
        """What a safety stop costs: ``SAFETY_PENALTY``, plus the rest of the time limit
        at the current reward if it is negative (reach), so that ending an episode by a
        stop always scores below holding the stop pose until the time limit."""
        steps_left = max(0.0, (self.time_limit - self.elapsed) * self.step_hz)
        return SAFETY_PENALTY - min(float(self._reward()), 0.0) * steps_left

    def _engage(self, violation: SafetyViolation) -> None:
        """The emergency mode: hold the pose, keep the gripper's command (module doc)."""
        self.emergency = violation
        # Hold the pose the sensors just read (one physics step old, as hold_current
        # reads it after control_step). Torque control gets one write of the gravity
        # compensation torques; the episode ends in this tick.
        self.osc.hold_current()
        if self.control == "torque":
            arm = self.ids.arm_actuators
            self.data.ctrl[arm] = np.clip(
                self.data.qfrc_bias[self.ids.arm_dof], -self._arm_limits, self._arm_limits
            )

    def _cube_contacts(self) -> int:
        """The cube's contacts: bit 1 with the table, bit 2 with anything else (the robot)."""
        ncon = self.data.ncon
        if ncon == 0:
            return 0
        con = self.data.contact
        g1, g2 = con.geom1[:ncon], con.geom2[:ncon]
        cube, table = self.ids.cube_geom, self.ids.table_geom
        other = np.where(g1 == cube, g2, np.where(g2 == cube, g1, -1))
        other = other[other >= 0]
        return int(np.any(other == table)) | 2 * int(np.any(other != table))

    def _collect_events(self) -> list[EventRecord]:
        """The detector's events not reported yet (also those added with ``report``)."""
        new = self.events.events[self._delivered :]
        self._delivered = len(self.events.events)
        if new and self.recorder is not None:
            self.recorder.add_events(new)
        return list(new)

    def _track_grasp(self, frame: SensorFrame) -> None:
        """Keep the lowest ee height since ``GRASP_CLOSED`` (the grasp height)."""
        ev = self.events  # its state is one tick old here
        if ev.closed and ev.in_grasp:
            z = float(frame.ee_pos[2])
            self._grasp_z = z if self._grasp_z is None else min(self._grasp_z, z)
            self._best_rise = max(self._best_rise, z - self._grasp_z)
        elif not ev.in_grasp:
            self._grasp_z = None
            self._best_rise = 0.0
            self._placing = self._released = False

    def _carry_logic(self, frame: SensorFrame) -> None:
        """Start and end carries for a policy without a phase machine (module doc).

        States: GRASP (closed, not carrying) → CARRY (watched) → PLACE (lowered near
        the table under a squeeze) or RELEASE (opened near the table). A squeeze in
        RELEASE goes back to GRASP, so the next carry is watched again.
        """
        ev, suite = self.events, self.sensors
        if self._grasp_z is None:
            if suite.slip.watching and not ev.in_grasp:
                suite.unwatch_slip()  # the grasp ended: so did the carry
            return
        rise = float(frame.ee_pos[2]) - self._grasp_z
        both = bool(frame.contact[0] and frame.contact[1])
        if self.control == "osc":
            vz = self._target_vz  # the commanded motion: payload sag is not a move
            lowered, raised = vz < 0.0, vz > 0.0
        else:  # the measured motion, CARRY_RISE past the turning point (a hysteresis:
            # the ee rebounds at 5-6 mm/s when a set-down stops, ≤ 0.15 mm), and for a
            # set-down the table taking the load for SETDOWN_S (payload sag lowers the
            # ee, not the load)
            if suite.slip.watching:
                self._track_carry_load(float(frame.load))
                margin = SETDOWN_NOISE_K * suite.slip.sigma()  # 0 without sensor noise
                below = frame.load + margin < SETDOWN_LOAD * self._carry_load
                self._unloaded = self._unloaded + 1 if below else 0
            lowered = rise < self._carry_peak - CARRY_RISE and self._unloaded >= round(
                SETDOWN_S / self.dt
            )
            raised = rise > self._place_low + CARRY_RISE
        if suite.slip.watching:
            self._carry_peak = max(self._carry_peak, rise)
        elif self._placing:
            self._place_low = min(self._place_low, rise)
        if self._released:
            if self._squeezing:
                self._released = False  # squeezed again: back to GRASP
            return
        if suite.slip.watching:  # CARRY
            if rise <= RELEASE_HEIGHT and not self._squeezing:
                suite.unwatch_slip()  # opened near the table: a release, not a drop
                self._released = True
            elif rise <= RELEASE_HEIGHT and lowered:
                suite.unwatch_slip()  # lowered near the table: a set-down, not a drop
                self._placing = True
                self._place_low = rise
            return
        if self._placing:  # PLACE
            if not self._squeezing:
                self._placing = False
                self._released = True
            elif both and raised:
                suite.watch_slip()  # raised again: a new carry
                self._placing = False
                self._start_carry(rise)
            return
        if self._squeezing and both and rise >= CARRY_RISE:  # GRASP
            suite.watch_slip()
            self._start_carry(rise)

    def _start_carry(self, rise: float) -> None:
        self._carry_peak, self._carry_load = rise, 0.0
        self._load_window.clear()
        self._unloaded = 0

    def _track_carry_load(self, load: float) -> None:
        """Raise the carry's reference load to the lowest load of the last
        ``SETDOWN_WINDOW_S`` (a level the load held that long; refer to SETDOWN_LOAD).
        Until the carry is that old the reference is 0: a set-down then needs the
        table to push the load below 0, and an open near the table is a release."""
        window = self._load_window
        window.append(load)
        full = round(SETDOWN_WINDOW_S / self.dt)
        if len(window) > full:
            window.popleft()
        if len(window) == full:
            self._carry_load = max(self._carry_load, min(window))

    def _next_phase(self) -> str:
        if not self.events.in_grasp:
            return "APPROACH"
        if self._released:
            return "RELEASE"
        if self._placing:
            return "PLACE"
        if self.sensors.slip.watching:
            return "CARRY"
        return "GRASP"

    def _check_terminal(self, events: list[EventRecord]) -> str | None:
        """The result if this tick ends the episode (``events``: the tick's new events)."""
        ev, frame, priv = self.events, self._frame, self._priv
        if ev.broken and self.task != "reach":  # the outcome outranks a stop in its tick
            return "PLACE_BREAK" if self.task == "pick_place" and self._place_started else "BREAK"
        if any(e.name is Event.SAFETY_STOP for e in events):
            return "SAFETY_STOP"
        t = frame.t
        if self.task == "reach":
            near = float(np.linalg.norm(frame.ee_pos - self.goal)) <= REACH_TOLERANCE
            self._near_since = (
                (self._near_since if self._near_since is not None else t) if near else None
            )
            if near and t - self._near_since >= REACH_HOLD_S - _EPS_T:
                return "SUCCESS"
            return None
        dropped = any(e.name is Event.DROP for e in events)
        lifted = ev.lifted or self._lifted_carry == frame.watch_id  # in the air, or this carry's
        if dropped and frame.watch_id > 0 and lifted:
            return "DROP"  # a drop after a lift-off
        contact = bool(frame.contact[0] or frame.contact[1])
        if self.task == "lift":
            high = contact and float(priv.cube_pos[2]) - priv.rest_z >= LIFT_HEIGHT
            self._high_since = (
                (self._high_since if self._high_since is not None else t) if high else None
            )
            if high and t - self._high_since >= LIFT_HOLD_S - _EPS_T:
                return "SUCCESS"
            return None
        if ev.placed and not self._place_judged:
            # T06: PLACED alone is not success. The last touchdown decides: a cube that
            # left the fingers high before it fell; one that came down far from the
            # target and reached it along the table was pushed (no success, the episode
            # runs on; PLACED comes once).
            self._place_judged = True
            down = self._touchdown_xy
            if down is not None and self._touchdown_release > RELEASE_HEIGHT:
                return "DROP"
            if down is not None and np.hypot(*(down - priv.place_xy)) <= PLACE_TOUCHDOWN_RADIUS:
                return "SUCCESS"
        if self._place_started:
            speed = float(np.linalg.norm(priv.cube_vel[:3]))
            tilted = (
                not contact
                and speed < self.event_config.place_speed
                and not _flat(priv.cube_quat, self.event_config.place_tilt_deg)
            )
            self._tilted_since = (
                (self._tilted_since if self._tilted_since is not None else t) if tilted else None
            )
            if tilted and t - self._tilted_since >= TOPPLE_REST_S - _EPS_T:
                return "TOPPLE"
        return None

    # ------------------------------------------------------- reward, obs, info

    def _reward(self) -> float:
        frame = self._frame
        ee = frame.ee_pos
        if self.task == "reach":
            return -float(np.linalg.norm(ee - self.goal))
        both = bool(frame.contact[0] and frame.contact[1])
        grasped = both and self._squeezing and self.events.in_grasp
        # Pick-and-place keeps the lift and transport terms through the set-down and
        # the release, so lowering the cube onto the target does not lose reward.
        placing = self.task == "pick_place" and (self._placing or self._released)
        cube = np.array([self.observed.x, self.observed.y, self.observed.z])
        if not (grasped or placing):
            return 1.0 - math.tanh(10.0 * float(np.linalg.norm(ee - cube)))
        reward = 1.0 + GRASP_BONUS
        if self._grasp_z is not None:
            rise = self._best_rise if self.task == "pick_place" else float(ee[2]) - self._grasp_z
            reward += min(max(rise / LIFT_HEIGHT, 0.0), 1.0)
        if self.task == "pick_place" and (placing or frame.watching):
            reward += 1.0 - math.tanh(10.0 * float(np.linalg.norm(ee[:2] - self.goal[:2])))
        return reward

    def _obs(self) -> dict[str, np.ndarray]:
        frame, data, ids, seen = self._frame, self.data, self.ids, self.observed
        obs = {
            "joint_pos": data.qpos[ids.arm_qpos].astype(np.float64),
            "joint_vel": data.qvel[ids.arm_dof].astype(np.float64),
            "ee_pos": np.array(frame.ee_pos, dtype=np.float64),
            "ee_quat": np.array(frame.ee_quat, dtype=np.float64),
            "ee_vel": np.array(frame.ee_vel, dtype=np.float64),
            "wrench": np.array(frame.wrench, dtype=np.float64),
            "load": np.array([frame.load], dtype=np.float64),
            "finger_force": np.array(frame.finger_force, dtype=np.float64),
            "finger_contact": frame.contact.astype(np.int8),
            "gripper_width": np.array([frame.width], dtype=np.float64),
            "cube_pos": np.array([seen.x, seen.y, seen.z], dtype=np.float64),
            "cube_yaw": np.array([seen.yaw], dtype=np.float64),
            "cube_size": np.array([seen.size], dtype=np.float64),
            "goal": self.goal.astype(np.float64),
        }
        if self.control == "osc":
            obs["ee_target"] = self.ee_target.astype(np.float64)
        return obs

    def _info(self, events: list[EventRecord]) -> dict[str, Any]:
        priv = self._priv

        def dump(records: list[EventRecord]) -> list[dict]:
            return [{"name": e.name.value, "t": e.t, "info": dict(e.info)} for e in records]

        return {
            "t": self.elapsed,
            "sim_time": float(self.data.time),  # the clock of event times and the recorder
            "phase": self.phase,
            "events": dump([e for e in events if e.name not in PRIVILEGED_EVENTS]),
            "result": self.result,
            "is_success": self.result == "SUCCESS",
            "privileged": {
                "events": dump([e for e in events if e.name in PRIVILEGED_EVENTS]),
                "object": dict(self._hidden),
                "cube_pos": priv.cube_pos.tolist(),
                "cube_quat": priv.cube_quat.tolist(),
                "broken": priv.broken,
            },
        }

    def _finish(self) -> None:
        if self.recorder is None:
            return
        priv = self._priv
        self.episode = self.recorder.finish(
            self.result,
            task=self.task,
            rise=float(priv.cube_pos[2]) - priv.rest_z,
            peak_cube_force=self.monitor.peak_force,
            sim_s=self.elapsed,
        )

    # ----------------------------------------------------------------- render

    def render(self) -> np.ndarray | None:
        """An RGB image (H, W, 3, uint8) from the ``overview`` camera (``rgb_array`` mode)."""
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self.render_size)
        # No mj_forward: kinematics one physics step old are fine for a picture, and
        # rendering must not touch the state the next step starts from.
        self._renderer.update_scene(self.data, camera=S.OVERVIEW_CAMERA)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _external_violation(event: EventRecord) -> SafetyViolation:
    """A ``SafetyViolation`` from a ``SAFETY_STOP`` reported outside the supervisor."""
    info = dict(event.info)

    def real(key: str) -> float:
        value = info.pop(key, math.nan)
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return math.nan

    reason = str(info.pop("reason", "external"))
    value, limit = real("value"), real("limit")
    detail = str(info.pop("detail", ""))
    return SafetyViolation(reason, float(event.t), value, limit, detail, info)


def _flat(quat: np.ndarray, tilt_deg: float) -> bool:
    """True if one of the cube's face normals is within ``tilt_deg`` of world z."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return bool(np.max(np.abs(mat[6:9])) >= math.cos(math.radians(tilt_deg)))


def register_envs() -> None:
    """Register the three env ids with Gymnasium (idempotent)."""
    for env_id, task in ENV_IDS.items():
        if env_id not in gym.registry:
            gym.register(env_id, entry_point="vfa.sim.env:PandaVFAEnv", kwargs={"task": task})


register_envs()
