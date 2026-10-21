"""Hidden object classes: property ranges, per-episode sampling, the break rule, perception.

ARCHITECTURE §4: the right grip depends on properties a camera cannot see. Each
episode draws a cube from one hidden class (``configs/objects.toml``):

- **Hidden**: mass, friction (sliding μ), contact softness (MuJoCo ``solref``),
  and the break-force threshold. Also the class name itself.
- **Visible**: the edge length (the same range for every class) and the pose
  on the table. ``apply_object`` paints every cube the same color
  (``OBJECT_RGBA``), so vision cannot identify the class.

``sample_object(cls, rng)`` draws an ``ObjectSpec`` from an ``ObjectClass``.
``apply_object(model, data, spec)`` writes it through the T02 hooks
(``set_cube_size``, ``set_cube_mass``, ``set_cube_friction``,
``set_cube_solref``, ``place_cube``). It validates everything before the first
write, so a bad spec leaves the model untouched.

The cube geom has ``priority=1`` and ``condim=4`` (ARCHITECTURE D8, set in
``scene.py``). MuJoCo otherwise takes the larger friction of the two geoms in a
contact, and the finger pads (μ = 1) would hide a slippery cube's friction.

**Break rule** (``BreakMonitor``): when the summed finger-to-cube normal force
(``finger_cube_forces``, left + right) exceeds the spec's ``break_force``, the
cube is ``broken``. The monitor paints it red (``BROKEN_RGBA``), and the caller
ends the attempt as a failure (``BREAK``). The threshold is on the **sum**: a
face-on squeeze of ``f`` N per finger loads it with ``2f``.

**Perception** (``observe_cube``): the observed pose is the true pose plus
optional Gaussian noise (``PerceptionNoise``; σ = 0 in benchmark v1). The
observed height of a cube on the table is its rest height ``size / 2``: a
resting cube sinks into the table by a class-dependent fraction of a
millimeter (its ``solref`` and friction govern that contact too), and the
exact sink would reveal the class.
"""

from __future__ import annotations

import logging
import math
import tomllib
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import mujoco
import numpy as np

from vfa.sim.scene import (
    CUBE_SIZE_RANGE,
    REPO_ROOT,
    TABLE_TOP_Z,
    SceneConfig,
    SceneIds,
    _check_cube_xy,
    _cube_size,
    _friction_triple,
    _positive,
    _real,
    _refsafe,
    _solref_pair,
    _table_rect,
    finger_cube_forces,
    place_cube,
    scene_ids,
    set_cube_friction,
    set_cube_mass,
    set_cube_size,
    set_cube_solref,
)

logger = logging.getLogger(__name__)

OBJECTS_TOML = REPO_ROOT / "configs" / "objects.toml"
# Every cube renders in this color, whatever its class (the scene default).
OBJECT_RGBA: tuple[float, float, float, float] = SceneConfig().cube_rgba
# A broken cube turns this color.
BROKEN_RGBA: tuple[float, float, float, float] = (0.85, 0.1, 0.1, 1.0)
# A cube looks the same after a quarter turn about its up axis.
CUBE_SYMMETRY = 0.5 * math.pi
# observe_cube reads a cube less than this far below its rest height as resting [m].
REST_SINK_BAND = 0.005
# Allowed cube mass [kg]. MuJoCo scales contact stiffness by the effective mass,
# so far lighter cubes sink without an error: under a 2 N press a 1 g cube sinks
# 6.7 mm (past REST_SINK_BAND) and a 0.1 g cube falls through the table.
OBJECT_MASS_RANGE: tuple[float, float] = (0.01, 100.0)

Range = tuple[float, float]


# ------------------------------------------------------------------- data types


@dataclass(slots=True, frozen=True)
class PoseRange:
    """Where a cube may appear: center x, y [m] (world frame) and yaw [rad]."""

    x: Range
    y: Range
    yaw: Range

    def __post_init__(self) -> None:
        for name in ("x", "y", "yaw"):
            object.__setattr__(self, name, _range(f"pose.{name}", getattr(self, name)))


@dataclass(slots=True, frozen=True)
class PerceptionNoise:
    """Gaussian noise on the observed cube pose: position σ [m] and yaw σ [rad]."""

    pos_sigma: float = 0.0
    yaw_sigma: float = 0.0

    def __post_init__(self) -> None:
        for name in ("pos_sigma", "yaw_sigma"):
            value = _real(f"perception.{name}", getattr(self, name))
            if value < 0:
                raise ValueError(f"perception.{name} must be >= 0, got {value!r}")
            object.__setattr__(self, name, value)


@dataclass(slots=True, frozen=True)
class ObjectClass:
    """One hidden class: the ranges ``sample_object`` draws a cube from.

    ``friction`` is the sliding μ range. ``timeconst`` is the ``solref[0]``
    range of the ``softness`` preset (larger = softer). ``break_force`` is the
    range of the summed finger-force threshold [N], or ``None`` for a cube that
    cannot break. The contact constants (``dampratio`` = ``solref[1]``,
    ``torsional`` and ``rolling`` friction) and the pose range are shared by
    all classes of one config. ``configs/objects.toml`` sets every value; the
    class has no defaults of its own.
    """

    name: str
    size: Range
    mass: Range
    friction: Range
    softness: str
    timeconst: Range
    dampratio: float
    torsional: float
    rolling: float
    pose: PoseRange
    break_force: Range | None = None

    def __post_init__(self) -> None:
        def put(name: str, value: object) -> None:
            object.__setattr__(self, name, value)

        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"class name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.softness, str) or not self.softness:
            raise ValueError(f"{self.name}: softness must be a non-empty string")
        put("size", _range(f"{self.name}.size", self.size))
        lo, hi = CUBE_SIZE_RANGE
        if self.size[0] < lo or self.size[1] > hi:
            raise ValueError(f"{self.name}.size must lie in [{lo}, {hi}] m, got {self.size!r}")
        put("mass", _range(f"{self.name}.mass", self.mass, positive=True))
        _check_mass(f"{self.name}.mass", self.mass[0])
        _check_mass(f"{self.name}.mass", self.mass[1])
        put("friction", _range(f"{self.name}.friction", self.friction, positive=True))
        put("timeconst", _range(f"{self.name}.timeconst", self.timeconst, positive=True))
        if self.break_force is not None:
            put("break_force", _range(f"{self.name}.break_force", self.break_force, positive=True))
        put("dampratio", _positive(f"{self.name}.dampratio", self.dampratio))
        # Validates torsion >= 0 and roll >= 0 (the slide value is a placeholder).
        _, torsional, rolling = _friction_triple((1.0, self.torsional, self.rolling))
        put("torsional", torsional)
        put("rolling", rolling)
        if not isinstance(self.pose, PoseRange):
            raise ValueError(f"{self.name}.pose must be a PoseRange, got {self.pose!r}")

    @property
    def fragile(self) -> bool:
        """True if cubes of this class can break."""
        return self.break_force is not None


@dataclass(slots=True, frozen=True)
class ObjectSpec:
    """One concrete cube for one episode.

    Hidden fields (``HIDDEN_FIELDS``): ``class_name``, ``mass`` [kg],
    ``friction`` (MuJoCo's slide, torsion, roll), ``solref`` (timeconst,
    dampratio), ``break_force`` [N, summed over both fingers; ``None`` =
    unbreakable]. Visible fields (``VISIBLE_FIELDS``): ``size`` (edge length
    [m]), ``x``, ``y`` (center on the table [m]), ``yaw`` [rad].
    """

    class_name: str
    size: float
    mass: float
    friction: tuple[float, float, float]
    solref: tuple[float, float]
    break_force: float | None
    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        def put(name: str, value: object) -> None:
            object.__setattr__(self, name, value)

        if not isinstance(self.class_name, str) or not self.class_name:
            raise ValueError(f"class_name must be a non-empty string, got {self.class_name!r}")
        put("size", _cube_size(self.size))
        put("mass", _check_mass("mass", self.mass))
        put("friction", _friction_triple(self.friction))
        # The timestep check needs the model; apply_object repeats it there.
        put("solref", _solref_pair(self.solref, 0.0))
        if not self.solref[0] > 0:
            raise ValueError(f"solref must be (timeconst > 0, dampratio > 0), got {self.solref!r}")
        if self.break_force is not None:
            put("break_force", _positive("break_force", self.break_force))
        for name in ("x", "y", "yaw"):
            put(name, _real(name, getattr(self, name)))

    @property
    def fragile(self) -> bool:
        """True if this cube can break."""
        return self.break_force is not None

    def hidden(self) -> dict[str, object]:
        """The privileged properties (for logging and scoring, never for the policy)."""
        return {name: getattr(self, name) for name in HIDDEN_FIELDS}

    def visible(self) -> dict[str, object]:
        """The properties a camera can measure."""
        return {name: getattr(self, name) for name in VISIBLE_FIELDS}


HIDDEN_FIELDS: tuple[str, ...] = ("class_name", "mass", "friction", "solref", "break_force")
VISIBLE_FIELDS: tuple[str, ...] = ("size", "x", "y", "yaw")


@dataclass(slots=True, frozen=True)
class ObjectConfig:
    """``configs/objects.toml``: the classes (in file order) and the perception noise."""

    classes: Mapping[str, ObjectClass]
    perception: PerceptionNoise

    def __post_init__(self) -> None:
        classes = dict(self.classes)
        if not classes:
            raise ValueError("an object config needs at least one class")
        for name, cls in classes.items():
            if not isinstance(cls, ObjectClass) or cls.name != name:
                raise ValueError(f"class {name!r} must be an ObjectClass named {name!r}")
        if not isinstance(self.perception, PerceptionNoise):
            raise ValueError(f"perception must be a PerceptionNoise, got {self.perception!r}")
        object.__setattr__(self, "classes", MappingProxyType(classes))

    def __reduce__(self):  # a mappingproxy cannot be pickled; workers need the config
        return (ObjectConfig, (dict(self.classes), self.perception))

    # File order is part of the config (``names``), so equality and hash both use it.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObjectConfig):
            return NotImplemented
        key = (tuple(self.classes.items()), self.perception)
        return key == (tuple(other.classes.items()), other.perception)

    def __hash__(self) -> int:
        return hash((tuple(self.classes.items()), self.perception))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.classes)


@dataclass(slots=True, frozen=True)
class ObservedPose:
    """What a camera reports about the cube: center [m], yaw [rad], edge length [m].

    ``yaw`` is wrapped into [−π/4, π/4): a camera cannot tell a cube from the
    same cube turned a quarter turn. ``z`` of a cube resting on the table is
    its rest height ``size / 2`` (refer to ``observe_cube``).
    """

    x: float
    y: float
    z: float
    yaw: float
    size: float


# ------------------------------------------------------------------------ config


def load_object_config(path: str | Path | None = None) -> ObjectConfig:
    """Load and validate ``configs/objects.toml`` (or ``path``).

    Every section and key is required and unknown keys are errors, so the file
    alone defines the benchmark objects (T10 hashes it). The ranges must also
    fit the default scene (``SceneConfig()``): every pose in the pose box keeps
    the largest cube on the table and outside the base keep-out, and every
    softness time constant is at least 2 × timestep.
    """
    path = Path(path) if path is not None else OBJECTS_TOML
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    sections = {"pose", "contact", "softness", "perception", "classes"}
    _keys("objects.toml", raw, required=sections)

    pose_raw = _table("pose", raw["pose"])
    _keys("pose", pose_raw, required={"x", "y", "yaw"})
    pose = PoseRange(**pose_raw)

    contact = _table("contact", raw["contact"])
    _keys("contact", contact, required={"dampratio", "torsional", "rolling"})

    scene = SceneConfig()
    presets: dict[str, Range] = {}
    for name, value in _table("softness", raw["softness"]).items():
        presets[name] = _range(f"softness.{name}", value, positive=True)
        if presets[name][0] < 2.0 * scene.timestep:
            raise ValueError(
                f"softness.{name} time constant must be >= 2 × timestep ({2.0 * scene.timestep})"
            )

    perception_raw = _table("perception", raw["perception"])
    _keys("perception", perception_raw, required={"pos_sigma", "yaw_sigma"})

    classes: dict[str, ObjectClass] = {}
    for name, body in _table("classes", raw["classes"]).items():
        body = _table(f"classes.{name}", body)
        _keys(
            f"classes.{name}",
            body,
            required={"size", "mass", "friction", "softness"},
            optional={"break_force"},
        )
        preset = body["softness"]
        if not isinstance(preset, str) or preset not in presets:
            raise ValueError(
                f"classes.{name}.softness must be one of {sorted(presets)}, got {preset!r}"
            )
        cls = ObjectClass(
            name=name,
            size=body["size"],
            mass=body["mass"],
            friction=body["friction"],
            softness=preset,
            timeconst=presets[preset],
            pose=pose,
            break_force=body.get("break_force"),
            **contact,
        )
        _check_pose_box(cls, scene)
        classes[name] = cls
    return ObjectConfig(classes=classes, perception=PerceptionNoise(**perception_raw))


def _check_pose_box(cls: ObjectClass, scene: SceneConfig) -> None:
    """Reject a pose box where the class's largest cube can leave the table or hit the base."""
    half = 0.5 * cls.size[1]
    (x0, x1), (y0, y1) = cls.pose.x, cls.pose.y
    nearest = (min(max(0.0, x0), x1), min(max(0.0, y0), y1))  # box point closest to the base
    for xy in ((x0, y0), (x0, y1), (x1, y0), (x1, y1), nearest):
        _check_cube_xy(
            f"{cls.name} pose (x, y)", xy, half, scene.table_center_xy, scene.table_half_xy
        )


# ---------------------------------------------------------------------- sampling


def sample_object(cls: ObjectClass, rng: np.random.Generator) -> ObjectSpec:
    """Draw one cube of class ``cls``. Every range is sampled uniformly.

    The draw order is fixed: the visible fields first (size, x, y, yaw), then
    the hidden ones (mass, friction, timeconst, and the break force if the
    class has one). So the same seed gives the same cube, and classes that
    share the size and pose ranges get the same visible cube from the same seed.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")

    def draw(r: Range) -> float:
        return float(rng.uniform(r[0], r[1]))

    size = draw(cls.size)
    x = draw(cls.pose.x)
    y = draw(cls.pose.y)
    yaw = draw(cls.pose.yaw)
    mass = draw(cls.mass)
    slide = draw(cls.friction)
    timeconst = draw(cls.timeconst)
    break_force = draw(cls.break_force) if cls.break_force is not None else None
    return ObjectSpec(
        class_name=cls.name,
        size=size,
        mass=mass,
        friction=(slide, cls.torsional, cls.rolling),
        solref=(timeconst, cls.dampratio),
        break_force=break_force,
        x=x,
        y=y,
        yaw=yaw,
    )


# -------------------------------------------------------------------------- apply


def apply_object(model: mujoco.MjModel, data: mujoco.MjData, spec: ObjectSpec) -> None:
    """Write ``spec`` into the compiled model and put the cube at its pose.

    Sets size, mass (and inertia), friction, and ``solref`` through the T02
    hooks, restores the shared color ``OBJECT_RGBA`` (a cube broken in the last
    episode turns back), then ``place_cube`` puts it at rest at ``(x, y, yaw)``
    and runs ``mj_forward``. The arm state in ``data`` is untouched.

    Everything is validated first (the footprint on the table and the solref
    time constant against the model timestep), so a rejected spec changes
    nothing. Call it after ``reset_to_home``: a later reset moves the cube back
    to the keyframe position (with the new size's rest height).
    """
    if not isinstance(spec, ObjectSpec):
        raise TypeError(f"spec must be an ObjectSpec, got {type(spec).__name__}")
    ids = scene_ids(model)  # the hooks resolve their own ids too (≈ 1 ms per call in total)
    _check_cube_xy("cube (x, y)", (spec.x, spec.y), 0.5 * spec.size, *_table_rect(model, ids))
    _solref_pair(spec.solref, float(model.opt.timestep), refsafe=_refsafe(model))
    set_cube_size(model, spec.size)
    set_cube_mass(model, spec.mass)
    set_cube_friction(model, spec.friction)
    set_cube_solref(model, spec.solref)
    model.geom_rgba[ids.cube_geom] = OBJECT_RGBA
    place_cube(model, data, spec.x, spec.y, spec.yaw)


# ----------------------------------------------------------------- break rule


class BreakMonitor:
    """The break rule for one attempt with one cube.

    Call ``update()`` after every physics step (after ``control_step``, when
    the contacts and their forces describe the same state: the start of the
    last physics step, so ``break_time``, read from ``data.time``, is up to
    one step late). With ``substeps > 1`` it sees only the last substep of
    each tick. Once the
    summed finger-to-cube normal force exceeds ``spec.break_force``, ``broken``
    stays ``True``, the cube turns ``BROKEN_RGBA``, and the caller must end the
    attempt as a failure. A cube with ``break_force=None`` never breaks.
    ``apply_object`` restores the color for the next episode; make a new
    monitor per attempt.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        spec: ObjectSpec,
        ids: SceneIds | None = None,
    ) -> None:
        if not isinstance(spec, ObjectSpec):
            raise TypeError(f"spec must be an ObjectSpec, got {type(spec).__name__}")
        self.model = model
        self.data = data
        self.ids = ids if ids is not None else scene_ids(model)
        self.threshold: float | None = spec.break_force
        self.class_name = spec.class_name
        self.broken = False
        self.peak_force = 0.0  # largest summed finger force seen [N]
        self.break_time: float | None = None  # data.time when it broke [s]
        self.break_load: float | None = None  # summed force that broke it [N]

    def update(self) -> bool:
        """Read the finger forces and apply the rule. Returns ``broken``."""
        left, right = finger_cube_forces(self.model, self.data, self.ids)
        return self.check(left + right, float(self.data.time))

    def check(self, force_sum: float, time: float) -> bool:
        """Apply the rule to a summed finger normal force [N] at ``time`` [s]."""
        force_sum = _real("force_sum", force_sum)
        time = _real("time", time)
        if force_sum > self.peak_force:
            self.peak_force = force_sum
        if self.broken or self.threshold is None or not force_sum > self.threshold:
            return self.broken
        self.broken = True
        self.break_time = time
        self.break_load = force_sum
        self.model.geom_rgba[self.ids.cube_geom] = BROKEN_RGBA
        logger.info(
            "[S1] BREAK %s at t=%.3f s: finger force %.2f N > %.2f N",
            self.class_name,
            self.break_time,
            force_sum,
            self.threshold,
        )
        return True


# --------------------------------------------------------------------- perception


def observe_cube(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    noise: PerceptionNoise | None = None,
    rng: np.random.Generator | None = None,
    ids: SceneIds | None = None,
) -> ObservedPose:
    """The cube pose a camera reports: the true pose plus optional Gaussian noise.

    Reads the free-joint ``qpos`` (current, no ``mj_forward`` needed). The yaw
    is the heading of a horizontal cube axis about world z, wrapped into
    [−π/4, π/4). A height less than ``REST_SINK_BAND`` below the rest height
    ``size / 2`` reads as the rest height (before the noise): the settled sink
    into the table (seeds 0–2: 0.08–0.14 mm for stiff cubes, 0.46–0.75 mm
    for soft ones, with T05's elliptic friction cones) would otherwise reveal
    the class. A lifted cube, or one that fell
    off the table, reads its true height. With zero noise ``rng`` is not used
    (and may be ``None``). The edge length is reported exactly.
    """
    noise = noise if noise is not None else PerceptionNoise()
    ids = ids if ids is not None else scene_ids(model)
    q = ids.cube_qpos
    size = 2.0 * float(model.geom_size[ids.cube_geom, 0])
    pos = np.array(data.qpos[q : q + 3], dtype=float)
    rest = TABLE_TOP_Z + 0.5 * size
    if rest - REST_SINK_BAND < pos[2] < rest:
        pos[2] = rest
    yaw = _cube_yaw(np.asarray(data.qpos[q + 3 : q + 7], dtype=float))
    noisy = noise.pos_sigma > 0 or noise.yaw_sigma > 0
    if noisy:
        if not isinstance(rng, np.random.Generator):
            raise ValueError("perception noise > 0 needs a numpy Generator rng")
        pos = pos + rng.normal(0.0, noise.pos_sigma, 3)
        yaw = yaw + float(rng.normal(0.0, noise.yaw_sigma))
    return ObservedPose(
        x=float(pos[0]),
        y=float(pos[1]),
        z=float(pos[2]),
        yaw=wrap_cube_yaw(yaw),
        size=size,
    )


def wrap_cube_yaw(yaw: float) -> float:
    """Map a cube yaw into [−π/4, π/4) using its quarter-turn symmetry."""
    yaw = _real("yaw", yaw)
    half = 0.5 * CUBE_SYMMETRY
    wrapped = (yaw + half) % CUBE_SYMMETRY - half
    # Float rounding can land exactly on +π/4; keep the interval half-open.
    return wrapped if wrapped < half else -half


def _cube_yaw(quat: np.ndarray) -> float:
    """Heading [rad] about world z of a horizontal cube axis.

    For a cube resting on any face, the body axis closest to world z points up.
    The first of the other two body axes lies flat; its heading is the yaw.
    """
    norm = math.hypot(*(float(v) for v in quat))  # no overflow for huge entries
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"cube quaternion must be finite and non-zero, got {quat!r}")
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat / norm)
    rot = mat.reshape(3, 3)
    up = int(np.argmax(np.abs(rot[2])))
    axis = 0 if up != 0 else 1
    return math.atan2(float(rot[1, axis]), float(rot[0, axis]))


# -------------------------------------------------------------------- helpers


def _range(name: str, value: object, positive: bool = False) -> Range:
    """A [low, high] pair of finite reals with low <= high (and low > 0 if ``positive``)."""
    if isinstance(value, str | bytes) or not isinstance(value, list | tuple | np.ndarray):
        raise ValueError(f"{name} must be a [low, high] pair, got {value!r}")
    items = list(value)
    if len(items) != 2:
        raise ValueError(f"{name} must be a [low, high] pair, got {value!r}")
    lo, hi = (_real(name, v) for v in items)
    if lo > hi:
        raise ValueError(f"{name} needs low <= high, got {value!r}")
    if positive and lo <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return (lo, hi)


def _check_mass(name: str, value: object) -> float:
    mass = _real(name, value)
    lo, hi = OBJECT_MASS_RANGE
    if not lo <= mass <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}] kg, got {value!r}")
    return mass


def _table(name: str, value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a table, got {value!r}")
    return value


def _keys(
    name: str,
    table: dict,
    required: AbstractSet[str] = frozenset(),
    optional: AbstractSet[str] = frozenset(),
) -> None:
    missing = required - table.keys()
    unknown = table.keys() - required - optional
    if missing:
        raise ValueError(f"{name}: missing keys {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name}: unknown keys {sorted(unknown)}")
