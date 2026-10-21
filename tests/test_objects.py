"""T04: hidden object classes, per-seed sampling, ``apply_object``, the break rule, perception.

Covers the ROADMAP T04 acceptance list: the cube's friction reaches the
finger contacts during a grasp (``data.contact[i].friction``); sampling is
deterministic per seed; the applied mass, friction, and solref read back from
the model; the break rule fires when a squeeze passes a fragile cube's
threshold and not below it.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import fields, replace

import mujoco
import numpy as np
import pytest

from vfa.control import osc as O
from vfa.control.gripper import Gripper
from vfa.sim import objects as OB
from vfa.sim import scene as S
from vfa.sim import scene_check

CLASS_NAMES = ("light_rigid", "heavy_rigid", "slippery", "fragile_soft", "heavy_slippery")
G = 9.81
# sample_object(..., default_rng(2026)): light_rigid, then fragile_soft from the same stream.
# (size, x, y, yaw, mass, friction, timeconst, break_force)
GOLDEN_LIGHT = (
    0.04357869627350872,
    0.5459869748572732,
    -0.009819479656954466,
    -0.2034172540207866,
    0.08549173343096513,
    0.958103649170653,
    0.02405143836677174,
    None,
)
GOLDEN_FRAGILE = (
    0.043547063836460974,
    0.5479177204027698,
    -0.060509169793329216,
    0.7335023561662825,
    0.18797752408059176,
    0.7907612412433954,
    0.07010928388316347,
    9.090922177934111,
)


@pytest.fixture(scope="module")
def config() -> OB.ObjectConfig:
    return OB.load_object_config()


def _spec(config: OB.ObjectConfig, name: str, seed: int = 0, **changes) -> OB.ObjectSpec:
    spec = OB.sample_object(config.classes[name], np.random.default_rng(seed))
    return replace(spec, **changes) if changes else spec


def _motor_model() -> mujoco.MjModel:
    """The scene with actuator8 as a force motor (ctrl = tendon force, ±100 N): the T05 default."""
    return S.build_scene()


def _floating_grasp(model: mujoco.MjModel, spec: OB.ObjectSpec):
    """Apply ``spec``, then float the cube between the open fingers with gravity off."""
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    OB.apply_object(model, data, spec)
    model.opt.gravity[:] = 0.0
    q = ids.cube_qpos
    data.qpos[q : q + 3] = data.site_xpos[ids.ee_site]
    data.qpos[q + 3 : q + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    return ids, data, osc, Gripper(model, data, ids)


def _run(model, data, controllers, seconds: float) -> None:
    for _ in range(round(seconds * O.CONTROL_HZ)):
        O.control_step(model, data, controllers, 1)


def _finger_cube_contacts(data: mujoco.MjData, ids: S.SceneIds) -> list[int]:
    fingers = ids.left_finger_geoms | ids.right_finger_geoms
    out = []
    for i in range(data.ncon):
        g1, g2 = int(data.contact[i].geom1), int(data.contact[i].geom2)
        if (g1 == ids.cube_geom and g2 in fingers) or (g2 == ids.cube_geom and g1 in fingers):
            out.append(i)
    return out


# ------------------------------------------------------------------- the config


def test_committed_config_has_the_five_classes_in_order(config: OB.ObjectConfig) -> None:
    assert config.names == CLASS_NAMES
    assert config.perception == OB.PerceptionNoise(0.0, 0.0)  # benchmark v1: no noise


def test_committed_config_matches_the_architecture_table(config: OB.ObjectConfig) -> None:
    """ARCHITECTURE §4 initial table (T05 may change both together)."""
    table = {
        "light_rigid": ((0.05, 0.15), (0.8, 1.0), "stiff", None),
        "heavy_rigid": ((0.6, 1.0), (0.7, 1.0), "stiff", None),
        "slippery": ((0.15, 0.35), (0.15, 0.30), "stiff", None),
        "fragile_soft": ((0.05, 0.20), (0.6, 0.9), "soft", (6.0, 12.0)),
        "heavy_slippery": ((0.5, 0.8), (0.25, 0.40), "stiff", None),
    }
    for name, (mass, friction, softness, break_force) in table.items():
        cls = config.classes[name]
        assert (cls.mass, cls.friction, cls.softness, cls.break_force) == (
            mass,
            friction,
            softness,
            break_force,
        ), name
    stiff = config.classes["light_rigid"].timeconst
    soft = config.classes["fragile_soft"].timeconst
    assert soft[0] > stiff[1], "a soft contact must have a larger solref time constant"


def test_visible_size_carries_no_class_information(config: OB.ObjectConfig) -> None:
    sizes = {cls.size for cls in config.classes.values()}
    assert len(sizes) == 1, "every class must share one size range"
    ((lo, hi),) = sizes
    assert 0.0 < lo < hi < 0.08, "the cube must fit between the open fingers (8 cm)"


def test_class_windows_exist_and_conflict(config: OB.ObjectConfig) -> None:
    """§4 design invariant at lift acceleration 0: summed grip ≥ m·g/μ to hold the cube.

    Each fragile class has a window below its break force, and even the easiest
    heavy_slippery cube needs more summed grip than the strongest fragile_soft
    cube can take, so no single grip suits both.
    """
    for cls in config.classes.values():
        hardest = cls.mass[1] * G / cls.friction[0]
        if cls.fragile:
            assert hardest < cls.break_force[0], cls.name
    easiest_heavy_slippery = 0.5 * G / 0.40
    assert easiest_heavy_slippery == pytest.approx(
        config.classes["heavy_slippery"].mass[0] * G / config.classes["heavy_slippery"].friction[1]
    )
    assert easiest_heavy_slippery > config.classes["fragile_soft"].break_force[1]


def test_pose_range_keeps_every_cube_on_the_table_and_reachable(config: OB.ObjectConfig) -> None:
    model = S.build_scene()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    for cls in config.classes.values():
        pose = cls.pose
        assert O.WORKSPACE_LOW[0] <= pose.x[0] and pose.x[1] <= O.WORKSPACE_HIGH[0]
        assert O.WORKSPACE_LOW[1] <= pose.y[0] and pose.y[1] <= O.WORKSPACE_HIGH[1]
        assert pose.yaw == pytest.approx((-math.pi / 4, math.pi / 4), abs=1e-6)
        corner = OB.sample_object(cls, np.random.default_rng(0))
        for x in pose.x:
            for y in pose.y:
                OB.apply_object(model, data, replace(corner, size=cls.size[1], x=x, y=y))


def test_config_pickles_for_worker_processes(config: OB.ObjectConfig) -> None:
    clone = pickle.loads(pickle.dumps(config))
    assert clone == config and clone.names == config.names
    assert hash(clone) == hash(config)
    reordered = OB.ObjectConfig(
        classes=dict(reversed(list(config.classes.items()))), perception=config.perception
    )
    assert reordered != config  # file order is part of the config
    with pytest.raises(TypeError):
        clone.classes["x"] = clone.classes["slippery"]  # type: ignore[index]
    with pytest.raises(ValueError, match="PerceptionNoise"):
        OB.ObjectConfig(classes=dict(config.classes), perception={"pos_sigma": 0.0})  # type: ignore[arg-type]


_GOOD_TOML = """
[pose]
x = [0.45, 0.60]
y = [-0.15, 0.15]
yaw = [-0.5, 0.5]

[contact]
dampratio = 0.9
torsional = 0.01
rolling = 0.0002

[softness]
stiff = [0.015, 0.025]
soft = [0.04, 0.08]

[perception]
pos_sigma = 0.002
yaw_sigma = 0.01

[classes.a]
size = [0.04, 0.06]
mass = [0.1, 0.2]
friction = [0.5, 0.6]
softness = "stiff"

[classes.b]
size = [0.05, 0.05]
mass = [0.3, 0.3]
friction = [0.2, 0.3]
softness = "soft"
break_force = [5.0, 7.0]
"""


def test_loader_reads_every_section(tmp_path) -> None:
    path = tmp_path / "objects.toml"
    path.write_text(_GOOD_TOML)
    cfg = OB.load_object_config(path)
    assert cfg.names == ("a", "b")
    a, b = cfg.classes["a"], cfg.classes["b"]
    assert a.pose == OB.PoseRange((0.45, 0.6), (-0.15, 0.15), (-0.5, 0.5))
    assert (a.dampratio, a.torsional, a.rolling) == (0.9, 0.01, 0.0002)
    assert a.timeconst == (0.015, 0.025) and not a.fragile
    assert b.timeconst == (0.04, 0.08) and b.break_force == (5.0, 7.0) and b.fragile
    assert cfg.perception == OB.PerceptionNoise(0.002, 0.01)
    spec = OB.sample_object(b, np.random.default_rng(1))
    assert spec.size == 0.05 and spec.mass == 0.3  # a [v, v] range is a constant
    assert spec.friction[1:] == (0.01, 0.0002) and spec.solref[1] == 0.9
    with pytest.raises(TypeError):
        cfg.classes["c"] = a  # type: ignore[index]  # read-only mapping


@pytest.mark.parametrize(
    "old, new, match",
    [
        ("[pose]", "typo = 1\n[pose]", "unknown keys"),
        ("rolling = 0.0002", "rolling = 0.0002\nspin = 1", "unknown keys"),
        ('softness = "soft"', 'softness = "soft"\ncolor = "red"', "unknown keys"),
        ("mass = [0.1, 0.2]\n", "", "missing keys"),
        ('softness = "stiff"\n', 'softness = "squishy"\n', "softness must be one of"),
        ("mass = [0.1, 0.2]", "mass = [0.2, 0.1]", "low <= high"),
        ("mass = [0.1, 0.2]", "mass = [-0.1, 0.2]", "must be > 0"),
        ("mass = [0.1, 0.2]", "mass = [nan, 0.2]", "finite"),
        ("mass = [0.1, 0.2]", "mass = [0.1, inf]", "finite"),
        ("mass = [0.1, 0.2]", "mass = [true, 0.2]", "finite"),
        ("mass = [0.1, 0.2]", "mass = 0.1", "pair"),
        ("mass = [0.1, 0.2]", 'mass = "0.1"', "pair"),
        ("mass = [0.1, 0.2]", "mass = [0.1, 0.2, 0.3]", "pair"),
        ("friction = [0.5, 0.6]", "friction = [0.0, 0.6]", "must be > 0"),
        ("size = [0.04, 0.06]", "size = [0.001, 0.06]", "size must lie in"),
        ("break_force = [5.0, 7.0]", "break_force = [0.0, 7.0]", "must be > 0"),
        ("soft = [0.04, 0.08]", "soft = [0.0, 0.08]", "must be > 0"),
        ("dampratio = 0.9", "dampratio = 0.0", "must be > 0"),
        ("torsional = 0.01", "torsional = -0.01", "torsion >= 0"),
        ("pos_sigma = 0.002", "pos_sigma = -0.002", "must be >= 0"),
        ("yaw = [-0.5, 0.5]", "yaw = [0.5, -0.5]", "low <= high"),
        ("yaw = [-0.5, 0.5]\n", "", "missing keys"),
        ("dampratio = 0.9\n", "", "missing keys"),
        ("yaw_sigma = 0.01\n", "", "missing keys"),
        ("x = [0.45, 0.60]", "x = [0.0, 0.1]", "keep-out"),
        ("x = [0.45, 0.60]", "x = [0.45, 0.83]", "off the table"),
        # Every corner clears the base; the box point nearest to it does not.
        ("x = [0.45, 0.60]\ny = [-0.15, 0.15]", "x = [-0.2, 0.6]\ny = [-0.1, 0.1]", "keep-out"),
        ("stiff = [0.015, 0.025]", "stiff = [0.003, 0.025]", "2 × timestep"),
        ("mass = [0.1, 0.2]", "mass = [0.0001, 0.2]", "kg"),
        ("mass = [0.1, 0.2]", "mass = [0.1, 1000.0]", "kg"),
    ],
)
def test_loader_rejects_bad_configs(tmp_path, old: str, new: str, match: str) -> None:
    assert old in _GOOD_TOML
    path = tmp_path / "objects.toml"
    path.write_text(_GOOD_TOML.replace(old, new, 1))
    with pytest.raises(ValueError, match=match):
        OB.load_object_config(path)


_HEAD_TOML = _GOOD_TOML[: _GOOD_TOML.index("[classes.a]")]


@pytest.mark.parametrize(
    "text, match",
    [
        (_HEAD_TOML, "missing keys"),
        ("classes = 1\n" + _HEAD_TOML, r"\[classes\] must be a table"),
        (_HEAD_TOML + "[classes]\n", "at least one class"),
        ("", "missing keys"),
    ],
)
def test_loader_rejects_missing_or_empty_classes(tmp_path, text: str, match: str) -> None:
    path = tmp_path / "objects.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        OB.load_object_config(path)


# ---------------------------------------------------------------------- sampling


def test_sampling_is_deterministic_per_seed(config: OB.ObjectConfig) -> None:
    def draw(seed: int) -> list[OB.ObjectSpec]:
        rng = np.random.default_rng(seed)
        return [OB.sample_object(cls, rng) for cls in config.classes.values() for _ in range(3)]

    assert draw(7) == draw(7)
    assert draw(7) != draw(8)
    for a, b in zip(draw(7), draw(8), strict=True):
        assert a.class_name == b.class_name and a.mass != b.mass


def test_sampling_order_is_pinned(config: OB.ObjectConfig) -> None:
    """Golden values: a change to the draw order would silently change every seeded benchmark."""
    rng = np.random.default_rng(2026)
    light = OB.sample_object(config.classes["light_rigid"], rng)  # unbreakable first: an
    fragile = OB.sample_object(config.classes["fragile_soft"], rng)  # extra draw shifts this
    golden = {
        "fragile": (GOLDEN_FRAGILE, fragile),
        "light": (GOLDEN_LIGHT, light),
    }
    for name, (want, got) in golden.items():
        values = (got.size, got.x, got.y, got.yaw, got.mass, got.friction[0], got.solref[0])
        assert values == pytest.approx(want[:7], rel=1e-12, abs=0.0), name
        assert got.break_force == (pytest.approx(want[7], rel=1e-12) if want[7] else None), name


def test_one_seed_gives_every_class_the_same_visible_cube(config: OB.ObjectConfig) -> None:
    for seed in range(5):
        specs = [
            OB.sample_object(cls, np.random.default_rng(seed)) for cls in config.classes.values()
        ]
        assert all(s.visible() == specs[0].visible() for s in specs), seed
        assert len({s.mass for s in specs}) == len(specs)


def test_samples_stay_in_their_ranges_and_cover_them(config: OB.ObjectConfig) -> None:
    rng = np.random.default_rng(123)
    for cls in config.classes.values():
        specs = [OB.sample_object(cls, rng) for _ in range(2000)]
        columns = {
            "size": ([s.size for s in specs], cls.size),
            "mass": ([s.mass for s in specs], cls.mass),
            "friction": ([s.friction[0] for s in specs], cls.friction),
            "timeconst": ([s.solref[0] for s in specs], cls.timeconst),
            "x": ([s.x for s in specs], cls.pose.x),
            "y": ([s.y for s in specs], cls.pose.y),
            "yaw": ([s.yaw for s in specs], cls.pose.yaw),
        }
        if cls.fragile:
            columns["break_force"] = ([s.break_force for s in specs], cls.break_force)
        else:
            assert all(s.break_force is None and not s.fragile for s in specs)
        for name, (values, (lo, hi)) in columns.items():
            values = np.asarray(values)
            assert values.min() >= lo and values.max() <= hi, (cls.name, name)
            span = hi - lo
            assert values.min() < lo + 0.01 * span and values.max() > hi - 0.01 * span
            assert values.mean() == pytest.approx(0.5 * (lo + hi), abs=0.03 * span)
        assert all(s.class_name == cls.name for s in specs)
        assert all(s.friction[1:] == (cls.torsional, cls.rolling) for s in specs)
        assert all(s.solref[1] == cls.dampratio for s in specs)


@pytest.mark.parametrize("rng", [None, 0, np.random.RandomState(0), np.random])
def test_sampling_needs_a_generator(config: OB.ObjectConfig, rng) -> None:
    with pytest.raises(TypeError, match="Generator"):
        OB.sample_object(config.classes["light_rigid"], rng)


def test_spec_splits_hidden_and_visible_fields(config: OB.ObjectConfig) -> None:
    names = {f.name for f in fields(OB.ObjectSpec)}
    assert set(OB.HIDDEN_FIELDS) | set(OB.VISIBLE_FIELDS) == names
    assert not set(OB.HIDDEN_FIELDS) & set(OB.VISIBLE_FIELDS)
    assert {"mass", "friction", "solref", "break_force", "class_name"} == set(OB.HIDDEN_FIELDS)
    assert {"size", "x", "y", "yaw"} == set(OB.VISIBLE_FIELDS)
    spec = _spec(config, "fragile_soft")
    assert spec.hidden() == {
        "class_name": "fragile_soft",
        "mass": spec.mass,
        "friction": spec.friction,
        "solref": spec.solref,
        "break_force": spec.break_force,
    }
    assert spec.visible() == {"size": spec.size, "x": spec.x, "y": spec.y, "yaw": spec.yaw}
    assert not any("rgba" in n or "color" in n for n in names), "a cube color would leak"


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"class_name": ""}, "class_name"),
        ({"size": 0.001}, "cube size"),
        ({"mass": 0.0}, "mass"),
        ({"mass": math.nan}, "mass"),
        ({"mass": True}, "mass"),
        ({"mass": 0.009}, "kg"),
        ({"mass": 1e3}, "kg"),
        ({"friction": (0.0, 0.005, 0.0001)}, "friction"),
        ({"friction": 0.5}, "friction"),
        ({"solref": (-100.0, -10.0)}, "solref"),
        ({"solref": (0.02, 0.0)}, "solref"),
        ({"break_force": 0.0}, "break_force"),
        ({"break_force": math.inf}, "break_force"),
        ({"x": math.nan}, "x"),
        ({"yaw": "0"}, "yaw"),
    ],
)
def test_spec_rejects_bad_values(config: OB.ObjectConfig, changes, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _spec(config, "fragile_soft", **changes)


@pytest.mark.parametrize("mass", [0.01, 100.0])
def test_spec_accepts_the_mass_bounds(config: OB.ObjectConfig, mass: float) -> None:
    assert OB.OBJECT_MASS_RANGE == (0.01, 100.0)
    assert _spec(config, "heavy_rigid", mass=mass).mass == mass
    cls = replace(config.classes["heavy_rigid"], mass=(mass, mass))
    assert OB.sample_object(cls, np.random.default_rng(0)).mass == mass


def test_class_rejects_bad_values(config: OB.ObjectConfig) -> None:
    good = config.classes["fragile_soft"]
    for changes, match in (
        ({"name": ""}, "name"),
        ({"softness": ""}, "softness"),
        ({"timeconst": (0.0, 0.1)}, "must be > 0"),
        ({"size": (0.04, 0.5)}, "size must lie in"),
        ({"pose": None}, "PoseRange"),
        ({"rolling": math.nan}, "finite"),
    ):
        with pytest.raises(ValueError, match=match):
            replace(good, **changes)


# ------------------------------------------------------------------------- apply


def test_applied_properties_read_back_from_the_model(config: OB.ObjectConfig) -> None:
    model = S.build_scene()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    for seed, name in enumerate(CLASS_NAMES):
        spec = _spec(config, name, seed)
        OB.apply_object(model, data, spec)
        half = 0.5 * spec.size
        assert model.body_mass[ids.cube_body] == spec.mass
        np.testing.assert_array_equal(model.geom_friction[ids.cube_geom], spec.friction)
        np.testing.assert_array_equal(model.geom_solref[ids.cube_geom], spec.solref)
        np.testing.assert_allclose(model.geom_size[ids.cube_geom], [half] * 3, rtol=1e-12)
        np.testing.assert_allclose(
            model.body_inertia[ids.cube_body], [2.0 / 3.0 * spec.mass * half**2] * 3, rtol=1e-12
        )
        np.testing.assert_allclose(
            model.geom_rgba[ids.cube_geom], OB.OBJECT_RGBA, atol=1e-7
        )  # float32
        q = ids.cube_qpos
        np.testing.assert_allclose(data.qpos[q : q + 3], [spec.x, spec.y, half], atol=1e-12)
        np.testing.assert_allclose(
            data.qpos[q + 3 : q + 7],
            [math.cos(spec.yaw / 2), 0.0, 0.0, math.sin(spec.yaw / 2)],
            atol=1e-12,
        )
        assert not data.qvel[ids.cube_dof : ids.cube_dof + 6].any()
        observed = OB.observe_cube(model, data)
        assert observed.size == pytest.approx(spec.size, rel=1e-12)
        assert observed.yaw == pytest.approx(OB.wrap_cube_yaw(spec.yaw), abs=1e-12)


def test_apply_after_another_class_matches_a_fresh_compile(config: OB.ObjectConfig) -> None:
    """No property of the previous cube leaks into the next one."""
    edited = S.build_scene()
    data = mujoco.MjData(edited)
    S.reset_to_home(edited, data)
    final = _spec(config, "light_rigid", 3)
    for spec in (_spec(config, "heavy_slippery", 1), _spec(config, "fragile_soft", 2), final):
        OB.apply_object(edited, data, spec)
        OB.BreakMonitor(edited, data, spec).check(1e6, 0.0)  # break the fragile one
    fresh = S.build_scene(
        S.SceneConfig(
            cube_size=final.size,
            cube_mass=final.mass,
            cube_friction=final.friction,
            cube_solref=final.solref,
        )
    )
    compared = 0
    for name in dir(fresh):
        value = getattr(fresh, name, None)
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiub":
            np.testing.assert_allclose(
                getattr(edited, name), value, rtol=1e-9, atol=1e-12, err_msg=name
            )
            compared += 1
    assert compared > 200


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"x": 0.9}, "off the table"),
        ({"x": 0.1, "y": 0.0}, "keep-out"),
        ({"solref": (0.003, 1.0)}, "2 × timestep"),
    ],
)
def test_apply_rejects_a_bad_spec_without_side_effects(
    config: OB.ObjectConfig, changes, match: str
) -> None:
    model = S.build_scene()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    before_model = {n: getattr(model, n).copy() for n in ("body_mass", "geom_size", "geom_solref")}
    before_qpos = data.qpos.copy()
    spec = _spec(config, "heavy_rigid", size=0.06, **changes)
    with pytest.raises(ValueError, match=match):
        OB.apply_object(model, data, spec)
    for name, value in before_model.items():
        np.testing.assert_array_equal(getattr(model, name), value, err_msg=name)
    np.testing.assert_array_equal(data.qpos, before_qpos)


def test_apply_follows_the_models_refsafe_setting(config: OB.ObjectConfig) -> None:
    """Like ``set_cube_solref``: no 2 × timestep check when MuJoCo would not clamp."""
    model = S.build_scene()
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_REFSAFE
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    OB.apply_object(model, data, _spec(config, "heavy_rigid", solref=(0.003, 1.0)))
    assert model.geom_solref[S.scene_ids(model).cube_geom, 0] == 0.003


def test_apply_rejects_a_non_spec() -> None:
    model = S.build_scene()
    with pytest.raises(TypeError, match="ObjectSpec"):
        OB.apply_object(model, mujoco.MjData(model), {"mass": 1.0})  # type: ignore[arg-type]


def test_every_class_renders_the_same_color(config: OB.ObjectConfig) -> None:
    model = S.build_scene()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    colors = set()
    for name in CLASS_NAMES:
        OB.apply_object(model, data, _spec(config, name))
        colors.add(tuple(model.geom_rgba[ids.cube_geom]))
        assert model.geom_matid[ids.cube_geom] == -1  # no material overrides the color
    assert len(colors) == 1
    np.testing.assert_allclose(colors.pop(), OB.OBJECT_RGBA, atol=1e-7)  # float32


# ------------------------------------------------------- friction reaches the grasp


@pytest.mark.parametrize("mu", [0.2, 0.9])
def test_cube_friction_governs_the_finger_contacts_in_a_grasp(
    config: OB.ObjectConfig, mu: float
) -> None:
    """ROADMAP T04: changing the cube friction changes the finger-contact friction."""
    model = S.build_scene()
    spec = _spec(config, "slippery", size=0.05, friction=(mu, 0.005, 0.0001))
    ids, data, osc, gripper = _floating_grasp(model, spec)
    gripper.squeeze(1.0)
    _run(model, data, [osc, gripper], 0.6)
    contacts = _finger_cube_contacts(data, ids)
    assert len(contacts) >= 2
    pads = ids.left_finger_geoms | ids.right_finger_geoms
    assert all(model.geom_friction[g, 0] == 1.0 for g in pads)  # max() would hide mu
    for i in contacts:
        con = data.contact[i]
        assert con.dim == 4
        np.testing.assert_allclose(con.friction[:3], [mu, mu, 0.005])
        np.testing.assert_allclose(con.solref, spec.solref)


def test_without_priority_the_finger_pads_would_hide_the_cube_friction(
    config: OB.ObjectConfig,
) -> None:
    """Why D8 matters: with priority 0 MuJoCo takes max(cube, pad) = the pad's 1.0."""
    model = S.build_scene()
    model.geom_priority[S.scene_ids(model).cube_geom] = 0
    spec = _spec(config, "slippery", size=0.05, friction=(0.2, 0.005, 0.0001))
    ids, data, osc, gripper = _floating_grasp(model, spec)
    gripper.squeeze(1.0)
    _run(model, data, [osc, gripper], 0.6)
    contacts = _finger_cube_contacts(data, ids)
    assert contacts
    for i in contacts:
        assert data.contact[i].friction[0] == pytest.approx(1.0)


def test_low_friction_cube_slips_out_of_a_grip_that_holds_a_high_friction_one(
    config: OB.ObjectConfig,
) -> None:
    """The friction reaches the dynamics: 1 N per finger holds 0.1 kg at μ 0.9, not at μ 0.2."""
    drops = {}
    for mu in (0.2, 0.9):
        model = S.build_scene()
        spec = _spec(config, "slippery", size=0.05, mass=0.1, friction=(mu, 0.005, 0.0001))
        ids, data, osc, gripper = _floating_grasp(model, spec)
        gripper.squeeze(1.0)
        _run(model, data, [osc, gripper], 0.6)
        z0 = float(data.qpos[ids.cube_qpos + 2])
        model.opt.gravity[:] = (0.0, 0.0, -G)
        _run(model, data, [osc, gripper], 0.5)
        drops[mu] = z0 - float(data.qpos[ids.cube_qpos + 2])
    assert drops[0.9] < 0.002, drops  # held (soft-contact creep only)
    assert drops[0.2] > 0.05, drops  # 2·μ·F = 0.4 N < m·g = 0.98 N: it falls


# -------------------------------------------------------------------- break rule


def test_break_rule_logic_at_the_threshold(config: OB.ObjectConfig) -> None:
    model = S.build_scene()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    spec = _spec(config, "fragile_soft", break_force=8.0)
    monitor = OB.BreakMonitor(model, data, spec, ids)
    assert monitor.check(7.9, 0.1) is False
    assert monitor.check(8.0, 0.2) is False  # "exceeds": equal does not break
    assert not monitor.broken and monitor.break_time is None and monitor.break_load is None
    np.testing.assert_allclose(model.geom_rgba[ids.cube_geom], OB.OBJECT_RGBA, atol=1e-7)
    assert monitor.check(8.0 + 1e-9, 0.3) is True
    assert monitor.broken and monitor.break_time == 0.3 and monitor.break_load == 8.0 + 1e-9
    np.testing.assert_allclose(model.geom_rgba[ids.cube_geom], OB.BROKEN_RGBA, atol=1e-7)  # float32
    assert monitor.check(0.0, 0.4) is True  # sticky
    assert monitor.check(20.0, 0.5) is True
    assert monitor.break_time == 0.3 and monitor.break_load == 8.0 + 1e-9
    assert monitor.peak_force == 20.0
    for bad in (math.nan, math.inf, "1"):
        with pytest.raises(ValueError):
            monitor.check(bad, 0.6)  # type: ignore[arg-type]
    fresh = OB.BreakMonitor(model, data, spec, ids)
    with pytest.raises(ValueError, match="time"):
        fresh.check(1.0, math.nan)  # rejected below the threshold too


def test_an_unbreakable_cube_never_breaks(config: OB.ObjectConfig) -> None:
    model = S.build_scene()
    data = mujoco.MjData(model)
    for name in CLASS_NAMES:
        spec = _spec(config, name)
        monitor = OB.BreakMonitor(model, data, spec)
        assert monitor.check(1e6, 0.0) is spec.fragile
        assert monitor.threshold == spec.break_force
    with pytest.raises(TypeError, match="ObjectSpec"):
        OB.BreakMonitor(model, data, None)  # type: ignore[arg-type]


def _squeeze(model, spec: OB.ObjectSpec, per_finger: float, seconds: float = 1.0):
    """Close to 4 mm clearance by position, then squeeze, running the break rule every tick.

    Returns ``ids``, ``data``, the monitor, and the per-tick trace
    ``(time, summed finger force read independently, monitor.broken)``.
    """
    ids, data, osc, gripper = _floating_grasp(model, spec)
    monitor = OB.BreakMonitor(model, data, spec, ids)
    trace: list[tuple[float, float, bool]] = []
    for command, duration in (
        (lambda: gripper.close(spec.size + 0.004), 0.5),
        (lambda: gripper.squeeze(per_finger, speed=None), seconds),  # T04 measured the T03 law
    ):
        command()
        for _ in range(round(duration * O.CONTROL_HZ)):
            O.control_step(model, data, [osc, gripper], 1)
            force = sum(S.finger_cube_forces(model, data, ids))
            trace.append((float(data.time), force, monitor.update()))
    assert all(f == 0.0 for f in (row[1] for row in trace[: round(0.5 * O.CONTROL_HZ)]))
    return ids, data, monitor, trace


def _assert_rule_held_every_tick(monitor: OB.BreakMonitor, trace) -> None:
    """``broken`` turned on at the first tick whose summed force exceeded the threshold."""
    threshold = monitor.threshold if monitor.threshold is not None else math.inf
    first = next((k for k, (_, f, _) in enumerate(trace) if f > threshold), None)
    for k, (_, _, broken) in enumerate(trace):
        assert broken == (first is not None and k >= first), k
    if first is not None:
        assert (monitor.break_time, monitor.break_load) == trace[first][:2]
    assert monitor.peak_force == max(f for _, f, _ in trace)


@pytest.mark.parametrize("seed", [0, 1])
def test_break_rule_fires_past_a_fragile_cubes_threshold(config: OB.ObjectConfig, seed) -> None:
    """ROADMAP T04: a squeeze whose summed force passes the threshold breaks the cube."""
    model = _motor_model()
    spec = _spec(config, "fragile_soft", seed)
    assert 6.0 <= spec.break_force <= 12.0
    ids, data, monitor, trace = _squeeze(model, spec, 0.75 * spec.break_force)  # sum 1.5×
    _assert_rule_held_every_tick(monitor, trace)
    assert monitor.broken
    assert monitor.break_load > spec.break_force
    # The 15 g fingers carry little momentum, so the summed force jumps to
    # 0.81-0.93x the commanded sum on the first contact tick (measured, seeds 0-5).
    assert monitor.break_load < 1.5 * 1.5 * spec.break_force
    assert 0.5 < monitor.break_time < 0.6  # within 0.1 s of the squeeze command
    np.testing.assert_allclose(model.geom_rgba[ids.cube_geom], OB.BROKEN_RGBA, atol=1e-7)  # float32
    # The next episode gets its color back.
    OB.apply_object(model, data, spec)
    np.testing.assert_allclose(model.geom_rgba[ids.cube_geom], OB.OBJECT_RGBA, atol=1e-7)  # float32
    assert not OB.BreakMonitor(model, data, spec, ids).broken


@pytest.mark.parametrize("seed", [0, 1])
def test_break_rule_does_not_fire_below_the_threshold(config: OB.ObjectConfig, seed) -> None:
    model = _motor_model()
    spec = _spec(config, "fragile_soft", seed)
    ids, data, monitor, trace = _squeeze(model, spec, 0.3 * spec.break_force)  # sum 0.6×
    _assert_rule_held_every_tick(monitor, trace)
    assert not monitor.broken
    assert monitor.break_time is None
    left, right = S.finger_cube_forces(model, data, ids)
    assert left + right == pytest.approx(0.6 * spec.break_force, rel=1e-3)
    assert 0.6 * spec.break_force <= monitor.peak_force < spec.break_force
    np.testing.assert_allclose(model.geom_rgba[ids.cube_geom], OB.OBJECT_RGBA, atol=1e-7)  # float32


def test_a_fingertip_press_on_the_table_does_not_break_the_cube(config: OB.ObjectConfig) -> None:
    """The rule counts finger-to-cube force only; the touch sensors also count the table."""
    model = S.build_scene()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    spec = _spec(config, "fragile_soft", x=0.5, y=-0.15, break_force=0.1)
    OB.apply_object(model, data, spec)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    osc.set_target((0.45, 0.15, -0.01))  # fingertips 1 cm into the table, away from the cube
    monitor = OB.BreakMonitor(model, data, spec, ids)
    for _ in range(round(2.0 * O.CONTROL_HZ)):
        O.control_step(model, data, [osc], 1)
        monitor.update()
    assert min(S.read_finger_touch(model, data, ids)) > 1.0
    assert not monitor.broken and monitor.peak_force == 0.0


def test_the_same_squeeze_breaks_only_the_fragile_cube(config: OB.ObjectConfig) -> None:
    results = {}
    for name in ("fragile_soft", "heavy_rigid"):
        model = _motor_model()
        spec = _spec(config, name, size=0.05)
        _, _, monitor, trace = _squeeze(model, spec, 8.0)  # sum 16 N > every break force
        _assert_rule_held_every_tick(monitor, trace)
        results[name] = monitor.broken
        assert monitor.peak_force > 12.0
    assert results == {"fragile_soft": True, "heavy_rigid": False}


@pytest.mark.render
def test_a_broken_cube_renders_red(tmp_path, config: OB.ObjectConfig) -> None:
    from matplotlib import image as mpimg

    model = S.build_scene()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    spec = _spec(config, "fragile_soft", x=0.5, y=0.0, yaw=0.0)
    OB.apply_object(model, data, spec)
    path_ok = scene_check.render_png(model, data, tmp_path / "ok.png", 320, 240)
    OB.BreakMonitor(model, data, spec).check(100.0, 0.0)
    path_broken = scene_check.render_png(model, data, tmp_path / "broken.png", 320, 240)

    def counts(path) -> tuple[int, int]:
        rgb = mpimg.imread(path)[..., :3]
        blue = (rgb[..., 2] > 0.6) & (rgb[..., 0] < 0.35)
        red = (rgb[..., 0] > 0.6) & (rgb[..., 1] < 0.3) & (rgb[..., 2] < 0.3)
        return int(blue.sum()), int(red.sum())

    blue_ok, red_ok = counts(path_ok)
    blue_broken, red_broken = counts(path_broken)
    assert blue_ok > 20 and red_ok < 5
    assert red_broken > 20 and blue_broken < blue_ok // 4


# -------------------------------------------------------------------- perception


def test_observation_without_noise_is_the_true_pose(config: OB.ObjectConfig) -> None:
    model = S.build_scene()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    spec = _spec(config, "slippery", 4)
    OB.apply_object(model, data, spec)
    obs = OB.observe_cube(model, data)
    assert obs == OB.observe_cube(model, data, OB.PerceptionNoise(), np.random.default_rng(0))
    assert (obs.x, obs.y) == pytest.approx((spec.x, spec.y), abs=1e-12)
    assert obs.z == pytest.approx(0.5 * spec.size, abs=1e-12)
    assert obs.yaw == pytest.approx(spec.yaw, abs=1e-12)  # already inside ±π/4


def test_observation_does_not_reveal_the_class(config: OB.ObjectConfig) -> None:
    """A settled cube sinks into the table by class; the camera must not see it.

    T05's elliptic friction cones removed the friction coupling of the sink: in
    seeds 0-2 a stiff cube sinks 0.08-0.14 mm and a soft one 0.46-0.75 mm (5.2-6.0x).
    """
    for seed in range(2):
        observed, sinks = [], []
        for cls in config.classes.values():
            spec = OB.sample_object(cls, np.random.default_rng(seed))
            model = S.build_scene()
            ids = S.scene_ids(model)
            data = mujoco.MjData(model)
            S.reset_to_home(model, data)
            OB.apply_object(model, data, spec)
            for _ in range(250):  # 0.5 s settle, arm gravity-compensated
                mujoco.mj_step1(model, data)
                data.ctrl[ids.arm_actuators] = data.qfrc_bias[ids.arm_dof]
                mujoco.mj_step2(model, data)
            sinks.append(0.5 * spec.size - float(data.qpos[ids.cube_qpos + 2]))
            obs = OB.observe_cube(model, data, ids=ids)
            assert obs.z == 0.5 * spec.size
            observed.append(np.array([obs.x, obs.y, obs.z, obs.yaw, obs.size]))
        assert min(sinks) > 0.0 and max(sinks) > 4 * min(sinks), sinks  # the leak is real
        for obs in observed[1:]:
            np.testing.assert_allclose(obs, observed[0], rtol=0.0, atol=1e-6)


def test_a_lifted_or_fallen_cube_reads_its_true_height(config: OB.ObjectConfig) -> None:
    model = S.build_scene()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    spec = _spec(config, "fragile_soft")
    OB.apply_object(model, data, spec)
    rest = S.TABLE_TOP_Z + 0.5 * spec.size
    z = ids.cube_qpos + 2
    assert OB.REST_SINK_BAND == 0.005
    cases = [
        (rest + 0.02, rest + 0.02),  # lifted
        (rest + 1e-4, rest + 1e-4),  # just lifted
        (rest - 1e-4, rest),  # resting, sunk 0.1 mm
        (rest - 0.0049, rest),
        (rest - 0.005, rest - 0.005),  # the band is open at its lower edge
        (rest - 0.0051, rest - 0.0051),
        (-0.7, -0.7),  # fell off the table
    ]
    for true_z, expected in cases:
        data.qpos[z] = true_z
        assert OB.observe_cube(model, data, ids=ids).z == expected, true_z


def test_zero_noise_leaves_the_rng_untouched() -> None:
    model = S.build_scene()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    rng = np.random.default_rng(3)
    state = rng.bit_generator.state
    OB.observe_cube(model, data, OB.PerceptionNoise(0.0, 0.0), rng)
    assert rng.bit_generator.state == state


def test_observation_noise_is_seeded_and_has_the_configured_sigma() -> None:
    model = S.build_scene()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    noise = OB.PerceptionNoise(pos_sigma=0.003, yaw_sigma=0.02)
    with pytest.raises(ValueError, match="Generator"):
        OB.observe_cube(model, data, noise)

    def draws(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        obs = [OB.observe_cube(model, data, noise, rng) for _ in range(3000)]
        return np.array([(o.x, o.y, o.z, o.yaw) for o in obs])

    a = draws(5)
    np.testing.assert_array_equal(a, draws(5))
    assert not np.array_equal(a, draws(6))
    truth = OB.observe_cube(model, data)
    err = a - np.array([truth.x, truth.y, truth.z, truth.yaw])
    sigma = np.array([0.003, 0.003, 0.003, 0.02])
    assert np.all(np.abs(err.mean(axis=0)) < 4.0 * sigma / math.sqrt(len(err)))
    np.testing.assert_allclose(err.std(axis=0), sigma, rtol=0.06)


@pytest.mark.parametrize("turns", [0, 1, -1, 2, 3, -5])
def test_observed_yaw_uses_the_quarter_turn_symmetry(turns: int) -> None:
    model = S.build_scene()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    S.place_cube(model, data, 0.5, 0.0, yaw=0.3 + turns * math.pi / 2)
    assert OB.observe_cube(model, data).yaw == pytest.approx(0.3, abs=1e-9)


def test_observed_yaw_of_a_cube_resting_on_another_face() -> None:
    model = S.build_scene()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    tip, turn = np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(tip, [1.0, 0.0, 0.0], math.pi / 2)
    mujoco.mju_axisAngle2Quat(turn, [0.0, 0.0, 1.0], -0.2)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, turn, tip)  # tip onto a side face, then turn about world z
    data.qpos[ids.cube_qpos + 3 : ids.cube_qpos + 7] = quat
    assert OB.observe_cube(model, data, ids=ids).yaw == pytest.approx(-0.2, abs=1e-9)
    for scale in (3.0, 1e-300, 1e200):  # unnormalized; no overflow in the norm
        data.qpos[ids.cube_qpos + 3 : ids.cube_qpos + 7] = scale * quat
        assert OB.observe_cube(model, data, ids=ids).yaw == pytest.approx(-0.2, abs=1e-9)
    for bad in (0.0, math.nan, math.inf):
        data.qpos[ids.cube_qpos + 3 : ids.cube_qpos + 7] = (bad, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="quaternion"):
            OB.observe_cube(model, data, ids=ids)


def test_wrap_cube_yaw_is_half_open() -> None:
    q = math.pi / 4
    assert OB.wrap_cube_yaw(q) == pytest.approx(-q)
    assert OB.wrap_cube_yaw(-q) == pytest.approx(-q)
    assert OB.wrap_cube_yaw(-1e-17) == pytest.approx(0.0, abs=1e-15)
    below = float(np.nextafter(-q, -np.inf))  # rounds onto +π/4 without the guard
    assert -q <= OB.wrap_cube_yaw(below) < q
    for yaw in np.linspace(-10.0, 10.0, 2001):
        w = OB.wrap_cube_yaw(float(yaw))
        assert -q <= w < q
        assert math.remainder(w - yaw, math.pi / 2) == pytest.approx(0.0, abs=1e-9)
