"""The frozen benchmark: chunked, resumable runs, metrics, and calibration (T10).

ARCHITECTURE §9; the benchmark is ``configs/benchmark.toml`` (profiles, seeds,
retry budget, sensor noise, the ``best_fixed`` grid) with
``configs/objects.toml`` and ``configs/best_fixed.toml``, hashed together in
``configs/benchmark.sha256``.

Usage::

    uv run python -m vfa.experiments.benchmark --condition naive --profile quick
    uv run python -m vfa.experiments.benchmark --condition best_fixed --profile full --workers 8
    uv run python -m vfa.experiments.benchmark aggregate --profile full
    uv run python -m vfa.experiments.benchmark calibrate --workers 8 --write  # best_fixed.toml
    uv run python -m vfa.experiments.benchmark freeze   # writes configs/benchmark.sha256
    uv run python -m vfa.experiments.benchmark throughput

**Situations.** Situations come in blocks of one cube per class; block ``b`` of
seed ``s`` orders the classes by a permutation drawn from ``(s, b)``, so the mix
is uniform and stratified but the index does not reveal the class. Situation
``i``'s cube is drawn by ``numpy.random.default_rng((s, i))``. Every condition
gets the same situations, and attempt ``a`` resets the env with
``env_seed(s, i, a)``, so each attempt also sees the same sensor noise in every
condition (a paired design). A condition (``vfa.experiments.baselines``) picks θ
for each attempt from the situation's opaque ``key`` (never the index, the
class, or the cube) and retries a failed situation up to ``max_attempts``
times.

**Chunks** (CLAUDE.md, Runtime limits). A chunk is one condition × one seed ×
``CHUNK`` situations (a learning condition, the memory of T15 or the bandits of
T17a: one chunk per seed, because what it learned evolves in situation order),
written to
``runs/benchmark/<profile>/<condition>/seed<s>_<start>-<end>.json`` with the
records of every attempt. A run skips chunks whose file already exists with
the same key (the situation files' hash, a fingerprint of the code that runs
an attempt, the condition, its fixed θ, the range); a wall-clock deadline
(``--deadline``, default 400 s, so a command ends within about 8 min with the
chunks it had started) stops submitting chunks, and the same command continues
the work. ``--workers`` runs chunks in parallel processes (default
``min(8, cpu − 2)``). ``aggregate`` reads the chunks and writes
``runs/benchmark/<profile>/metrics.json``; it refuses a condition with missing
chunks unless ``--partial`` (a learning condition with no chunk file at all is
left out). A learning condition's run first runs ``naive`` on the same seeds and
then prints success@1 per half and per block of ``BLOCK`` situations for both.
The bandits (``bandit``, ``bandit_flat``; ``vfa.experiments.bandit``) draw from a
generator seeded by the benchmark seed, and their chunk keys carry their arms and
bin edges (``configs/bandit.toml`` is not in the frozen hash).

**Metrics** (per seed, then mean ± sample std over seeds): success@1 and
success@k (k = ``max_attempts``) over situations, and success@1 over
situations 21–60 (``success_at_1_late``, T18's window); attempts-to-success (an
unsolved situation counts ``max_attempts + 1``); break, drop, slip, and
safety-stop rates per attempt, and the share of situations with a break;
peak-force excess over successful attempts (the largest summed finger–cube
normal force in SQUEEZE, LIFT, and HOLD minus the summed need ``m(g + a)/μ``
with the lift's peak acceleration, from the privileged cube properties; lower
is gentler); force jerk over successful attempts (RMS rate of change of the
summed, low-passed pad force over SQUEEZE and LIFT [N/s]; lower is smoother; a
failure's collapse would dominate it; the realistic pad noise adds a floor of
about 19 N/s in quadrature). ``SLIP`` labels come from ground truth
(``run_attempt`` labels an expired hold by the cube's height).

**Pick-and-place** (T19, ``--task pick_place``): its own config and hash
(``configs/benchmark_pick_place.toml``, ``benchmark_pick_place.sha256``, with the
shared ``objects.toml``), the ``quick``, ``calib``, and ``heldout`` profiles on
seeds no lift profile uses, the output root ``runs/benchmark_pick_place``, and the conditions
``naive``, ``random_retry`` (which also draws the place fields), and
``memory_full`` (``PlaceEmbedder`` keys); other conditions are refused. Its
metrics add the place-break, topple, and place-miss rates, and its peak force
reads SQUEEZE, LIFT, and TRANSPORT (the jerk reads SQUEEZE and LIFT, as the
lift's). The lift's chunks, keys, and
metrics are as before.

**Calibration** (``calibrate``). Every θ of the ``[grid]`` gets one attempt on
each calib-profile situation; the highest pooled success@1 wins, ties go to the
lower grip, then the lower lift acceleration (the rule was fixed before the
first run). ``--write`` saves it to ``configs/best_fixed.toml`` with the
calibration numbers. The report also gives each class's best grid θ (the
per-class ceiling: what a policy that knew the class could reach).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vfa.experiments.bandit import BANDIT_CONDITIONS
from vfa.experiments.baselines import (
    BEST_FIXED_PATH,
    CONDITIONS,
    NAIVE,
    REPO_ROOT,
    Baseline,
    grid_thetas,
    make_baseline,
)
from vfa.memory.embeddings import PLACE_STATS_PATH, STATS_PATH
from vfa.sim import objects as OB
from vfa.sim.env import PandaVFAEnv
from vfa.sim.sensors import REALISTIC_NOISE, SensorConfig, SensorNoise
from vfa.skills.params import SkillParams
from vfa.skills.pick import AttemptResult, GraspLiftSkill, run_attempt

CONFIG_PATH = REPO_ROOT / "configs" / "benchmark.toml"
OBJECTS_PATH = REPO_ROOT / "configs" / "objects.toml"
HASH_PATH = REPO_ROOT / "configs" / "benchmark.sha256"
FROZEN_FILES: tuple[Path, ...] = (CONFIG_PATH, OBJECTS_PATH, BEST_FIXED_PATH)
DEFAULT_OUT = REPO_ROOT / "runs" / "benchmark"
# The pick-and-place benchmark (T19): its own config and hash (over it and the
# shared objects.toml), its own output root, and the conditions it supports.
PICK_PLACE_CONFIG = REPO_ROOT / "configs" / "benchmark_pick_place.toml"
PICK_PLACE_HASH = REPO_ROOT / "configs" / "benchmark_pick_place.sha256"
PICK_PLACE_FILES: tuple[Path, ...] = (PICK_PLACE_CONFIG, OBJECTS_PATH)
PICK_PLACE_OUT = REPO_ROOT / "runs" / "benchmark_pick_place"
PICK_PLACE_CONDITIONS: tuple[str, ...] = ("naive", "random_retry", "memory_full")
TASKS: tuple[str, ...] = ("lift", "pick_place")
CHUNK = 15  # situations per chunk
DEFAULT_DEADLINE_S = 400.0  # + the chunks already running (≤ about 100 s each)
LATE_FROM = 20  # success_at_1_late: situations 21-60 (0-based index >= 20)
# The code that decides an attempt's outcome: chunk keys carry its fingerprint, so a
# change to it re-runs the chunks instead of reusing results from older code.
ASSETS_DIR = REPO_ROOT / "assets" / "menagerie" / "franka_emika_panda"
CODE_FILES: tuple[str, ...] = (
    "sim/*.py",
    "control/*.py",
    "skills/*.py",
    "experiments/baselines.py",
    "experiments/bandit.py",
    "experiments/benchmark.py",
    "memory/*.py",
    "cognition/*.py",
)
# Learning conditions (T15 memory, T17a bandits): one chunk per seed, because what they
# learned evolves within a seed. A memory condition's RetrievalPolicy settings (T18 adds
# the four component ablations, ARCHITECTURE §9):
MEMORY_POLICIES: dict[str, dict[str, Any]] = {
    "memory_full": {},
    "memory_ctx_only": {"keys": "context"},
    "memory_no_reinforce": {"reinforce": False},
    "memory_no_semantic": {"semantic": False},
    "memory_no_probe": {"probe_trigger": False},
    "memory_no_failure_bounds": {"failure_bounds": False},
}
# memory_frozen (T18's held-out run): memory_full learns on the warm-up seed's situations
# (unscored), then freezes and is scored on the chunk's seed with no further writes.
FROZEN = "memory_frozen"
WARMUP = {"seed": 1000, "situations": 60}  # the first evaluation seed, fixed before any run
MEMORY_CONDITIONS: tuple[str, ...] = (*MEMORY_POLICIES, FROZEN)
LEARNING_CONDITIONS: tuple[str, ...] = MEMORY_CONDITIONS + BANDIT_CONDITIONS
ALL_CONDITIONS: tuple[str, ...] = CONDITIONS + LEARNING_CONDITIONS
BANDIT_STREAM = 0x42414E44  # "BAND": the bandit's generator is (seed, BANDIT_STREAM)
BLOCK = 6  # situations per stratified block (one cube per class)
NOISES: dict[str, SensorNoise] = {"none": SensorNoise(), "realistic": REALISTIC_NOISE}
G = 9.81
_FORCE_PHASES = ("SQUEEZE", "LIFT", "HOLD")
_PLACE_FORCE_PHASES = ("SQUEEZE", "LIFT", "TRANSPORT")  # pick_place: the carry's grip
PLACE_RESULTS: tuple[str, ...] = ("PLACE_BREAK", "TOPPLE", "PLACE_MISS")
_JERK_PHASES = ("SQUEEZE", "LIFT")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- config


@dataclass(frozen=True, slots=True)
class Profile:
    seeds: tuple[int, ...]
    situations: int


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    task: str
    max_attempts: int
    noise: str
    calib_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    profiles: Mapping[str, Profile]
    grid: Mapping[str, tuple[float, ...]]


def load_benchmark_config(path: str | Path = CONFIG_PATH) -> BenchmarkConfig:
    """Read ``configs/benchmark.toml`` (every key required, unknown keys are errors), or
    the pick-and-place benchmark (``task = "pick_place"``, ``load_pick_place_config``)."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if data.get("benchmark", {}).get("task") == "pick_place":
        return _pick_place_config(data, str(path))
    _keys(data, {"benchmark", "profiles", "grid"}, str(path))
    b = data["benchmark"]
    _keys(b, {"task", "max_attempts", "noise", "calib_seeds", "eval_seeds"}, "[benchmark]")
    if b["task"] != "lift":
        raise ValueError(f"[benchmark] task must be 'lift' (the T08 skill), got {b['task']!r}")
    if b["noise"] not in NOISES:
        raise ValueError(f"[benchmark] noise must be one of {sorted(NOISES)}, got {b['noise']!r}")
    if not (isinstance(b["max_attempts"], int) and b["max_attempts"] >= 1):
        raise ValueError("[benchmark] max_attempts must be an int >= 1")
    calib, evals = _seeds(b["calib_seeds"]), _seeds(b["eval_seeds"])
    if set(calib) & set(evals):
        raise ValueError("calibration and evaluation seeds must be disjoint")
    profiles = {}
    for name, p in data["profiles"].items():
        _keys(p, {"seeds", "situations"}, f"[profiles.{name}]")
        if not (isinstance(p["situations"], int) and p["situations"] >= 1):
            raise ValueError(f"[profiles.{name}] situations must be an int >= 1")
        profiles[name] = Profile(_seeds(p["seeds"]), p["situations"])
    for name in ("quick", "full", "calib", "heldout"):
        if name not in profiles:
            raise ValueError(f"profile {name!r} is missing")
    if set(profiles["calib"].seeds) != set(calib):
        raise ValueError("the calib profile must use the calibration seeds")
    if set(profiles["full"].seeds) != set(evals):
        raise ValueError("the full profile must use the evaluation seeds")
    quick, heldout = set(profiles["quick"].seeds), set(profiles["heldout"].seeds)
    if quick & (set(calib) | set(evals)) or heldout & (set(calib) | set(evals) | quick):
        raise ValueError("quick and heldout seeds must differ from every other profile's")
    grid = data["grid"]
    _keys(grid, {"grip_force", "lift_accel"}, "[grid]")
    grid = {k: tuple(float(v) for v in grid[k]) for k in ("grip_force", "lift_accel")}
    return BenchmarkConfig(b["task"], b["max_attempts"], b["noise"], calib, evals, profiles, grid)


def load_pick_place_config(path: str | Path = PICK_PLACE_CONFIG) -> BenchmarkConfig:
    """Read ``configs/benchmark_pick_place.toml`` (T19)."""
    config = load_benchmark_config(path)
    if config.task != "pick_place":
        raise ValueError(f"{path}: task must be 'pick_place', got {config.task!r}")
    return config


def _pick_place_config(data: Mapping[str, Any], where: str) -> BenchmarkConfig:
    """The pick-and-place benchmark: quick, calib, and heldout profiles, no grid; its
    seeds differ from each other's and from every seed of the lift benchmark."""
    _keys(data, {"benchmark", "profiles"}, where)
    b = data["benchmark"]
    _keys(b, {"task", "max_attempts", "noise", "calib_seeds"}, "[benchmark]")
    if b["noise"] not in NOISES:
        raise ValueError(f"[benchmark] noise must be one of {sorted(NOISES)}, got {b['noise']!r}")
    if not (isinstance(b["max_attempts"], int) and b["max_attempts"] >= 1):
        raise ValueError("[benchmark] max_attempts must be an int >= 1")
    calib = _seeds(b["calib_seeds"])
    profiles = {}
    for name, p in data["profiles"].items():
        _keys(p, {"seeds", "situations"}, f"[profiles.{name}]")
        if not (isinstance(p["situations"], int) and p["situations"] >= 1):
            raise ValueError(f"[profiles.{name}] situations must be an int >= 1")
        profiles[name] = Profile(_seeds(p["seeds"]), p["situations"])
    if set(profiles) != {"quick", "calib", "heldout"}:
        raise ValueError("the pick_place benchmark has the quick, calib, and heldout profiles")
    if set(profiles["calib"].seeds) != set(calib):
        raise ValueError("the calib profile must use the calibration seeds")
    seen: set[int] = set()
    for prof in profiles.values():
        if seen & set(prof.seeds):
            raise ValueError("the pick_place profiles must use different seeds")
        seen |= set(prof.seeds)
    lift = load_benchmark_config(CONFIG_PATH)
    used = {s for prof in lift.profiles.values() for s in prof.seeds}
    if used & {s for prof in profiles.values() for s in prof.seeds}:
        raise ValueError("pick_place seeds must differ from every lift benchmark seed")
    return BenchmarkConfig("pick_place", b["max_attempts"], b["noise"], calib, (), profiles, {})


def _keys(table: Mapping, expected: set[str], where: str) -> None:
    if set(table) != expected:
        missing, extra = sorted(expected - set(table)), sorted(set(table) - expected)
        raise ValueError(f"{where}: missing {missing}, unknown {extra}")


def _seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds or not all(isinstance(s, int) and s >= 0 for s in seeds):
        raise ValueError(f"seeds must be a non-empty list of ints >= 0, got {values!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct, got {values!r}")
    return seeds


def benchmark_hash(paths: Sequence[Path] = FROZEN_FILES) -> str:
    """One SHA-256 over the frozen files (name and bytes, in order)."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).name.encode() + b"\0")
        digest.update(Path(path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


# ----------------------------------------------------------------- situations


@dataclass(frozen=True, slots=True)
class Situation:
    """One cube. Only ``key`` may reach a policy: ``index`` and ``spec`` are scoring
    data (the index, with the seed, would let a policy re-derive the class)."""

    seed: int
    index: int
    spec: OB.ObjectSpec

    @property
    def class_name(self) -> str:
        return self.spec.class_name

    @property
    def key(self) -> int:
        """A stable opaque id of the situation. It reveals neither the index nor the
        class to a policy that does not re-run this module's situation generator."""
        return int(np.random.SeedSequence([self.seed, self.index, 0x51]).generate_state(1)[0])


def situation_class(seed: int, index: int, names: Sequence[str]) -> str:
    """The class of situation ``index`` of ``seed``: blocks of one cube per class,
    each block in its own seeded order."""
    block, slot = divmod(index, len(names))
    order = np.random.default_rng((seed, block, 0xC1A55)).permutation(len(names))
    return names[int(order[slot])]


def make_situations(seed: int, n: int, objects: OB.ObjectConfig, start: int = 0) -> list[Situation]:
    """Situations ``start .. n-1`` of ``seed`` (refer to the module doc)."""
    out = []
    for i in range(start, n):
        cls = objects.classes[situation_class(seed, i, objects.names)]
        out.append(Situation(seed, i, OB.sample_object(cls, np.random.default_rng((seed, i)))))
    return out


def env_seed(seed: int, index: int, attempt: int) -> int:
    """The env reset seed of attempt ``attempt`` of situation ``index`` of ``seed``."""
    return int(np.random.SeedSequence([seed, index, attempt]).generate_state(1, np.uint32)[0])


def make_env(config: BenchmarkConfig, **render: Any) -> PandaVFAEnv:
    """The benchmark env: lift (or pick_place), osc at 500 Hz, the skill's carries,
    safety on (``render``: ``render_mode`` and ``render_size`` for a video)."""
    sensors = SensorConfig(noise=NOISES[config.noise])
    if config.task == "pick_place":
        from vfa.skills import place

        return place.make_env(sensors, **render)
    return PandaVFAEnv(
        "lift", "osc", step_hz=500, auto_carry=False, sensor_config=sensors, **render
    )


def make_skill(task: str) -> GraspLiftSkill:
    """The System 1 skill of ``task``."""
    if task == "pick_place":
        from vfa.skills.place import PickPlaceSkill

        return PickPlaceSkill()
    return GraspLiftSkill()


# ------------------------------------------------------------------- attempts


def attempt_record(result: AttemptResult, situation: Situation, attempt: int) -> dict[str, Any]:
    """The JSON record of one attempt (scores read the privileged columns)."""
    ep = result.episode
    spec = situation.spec
    record: dict[str, Any] = {
        "situation": situation.index,
        "class": situation.class_name,
        "attempt": attempt,
        "theta": result.theta.to_dict(),
        "theta_final": result.theta_final.to_dict(),
        "result": result.result,
        "success": result.success,
        "sim_s": result.sim_s,
        "wall_s": result.wall_s,
        "probe": None if result.probe is None else result.probe.to_dict(),
        "safety": result.safety,
        "peak_force": None,
        "peak_force_excess": None,
        "force_jerk": None,
    }
    if ep is None:
        return record
    phases = np.asarray(ep.phases)[ep.arrays["phase"]]
    force_phases = _PLACE_FORCE_PHASES if ep.meta.get("task") == "pick_place" else _FORCE_PHASES
    grip = np.isin(phases, force_phases)
    if np.any(grip):
        peak = float(np.max(np.sum(ep.arrays["cube_finger_force"][grip], axis=1)))
        need = spec.mass * (G + result.theta_final.lift_accel) / spec.friction[0]
        record["peak_force"] = peak
        record["peak_force_excess"] = peak - need
    jerk_rows = np.isin(phases, _JERK_PHASES)
    if np.count_nonzero(jerk_rows) >= 2:
        force = np.sum(ep.arrays["finger_force"][jerk_rows], axis=1)
        rate = np.diff(force) / np.diff(ep.arrays["t"][jerk_rows])
        record["force_jerk"] = float(np.sqrt(np.mean(rate * rate)))
    return record


def run_situation(
    env: PandaVFAEnv,
    skill: GraspLiftSkill,
    policy: Baseline,
    situation: Situation,
    max_attempts: int,
    on_attempt: Callable[[AttemptResult, dict], None] | None = None,
) -> list[dict[str, Any]]:
    """Attempts until one succeeds or ``max_attempts`` ran; one record each.

    ``on_attempt(result, record)`` receives scoring data (the record holds the
    class, the index, and privileged force metrics; the episode, privileged
    columns, and ``meta`` with the class and hidden properties): a learning
    policy (T15) must take only the key, the observable episode signals, and the
    outcome from it.
    """
    records = []
    for attempt in range(1, max_attempts + 1):
        theta = policy.theta(situation.key, attempt)
        result = run_attempt(
            env,
            skill,
            theta,
            seed=env_seed(situation.seed, situation.index, attempt),
            options={"object": situation.spec},
        )
        record = attempt_record(result, situation, attempt)
        records.append(record)
        if on_attempt is not None:
            on_attempt(result, record)
        if result.success:
            break
    return records


# --------------------------------------------------------------------- chunks


@dataclass(frozen=True, slots=True)
class Job:
    """One chunk: a condition (or a grid θ) × one seed × situations [start, end)."""

    condition: str  # a baseline, a memory condition (one chunk per seed), or "grid"
    seed: int
    start: int
    end: int
    path: str
    key: dict[str, Any]
    theta: dict[str, float] | None  # best_fixed and grid: the fixed θ
    max_attempts: int
    noise: str
    save_episodes: str | None = None
    task: str = "lift"


_WORKER: dict[str, Any] = {}


def _worker_env(
    noise: str, task: str = "lift"
) -> tuple[PandaVFAEnv, GraspLiftSkill, OB.ObjectConfig]:
    if _WORKER.get("noise") != noise or _WORKER.get("task") != task:
        sensors = SensorConfig(noise=NOISES[noise])
        if task == "pick_place":
            from vfa.skills import place

            _WORKER["env"] = place.make_env(sensors)
        else:
            _WORKER["env"] = PandaVFAEnv(
                "lift", "osc", step_hz=500, auto_carry=False, sensor_config=sensors
            )
        _WORKER["skill"] = make_skill(task)
        _WORKER["objects"] = OB.load_object_config()
        _WORKER["noise"], _WORKER["task"] = noise, task
    return _WORKER["env"], _WORKER["skill"], _WORKER["objects"]


def run_chunk(job: Job) -> tuple[str, int, float]:
    """Run one chunk and write its records (atomically). Returns (path, attempts, wall)."""
    logging.disable(logging.CRITICAL)  # thousands of attempts: keep worker logs quiet
    wall0 = time.perf_counter()
    env, skill, objects = _worker_env(job.noise, job.task)
    situations = make_situations(job.seed, job.end, objects, start=job.start)
    save = None if job.save_episodes is None else Path(job.save_episodes)
    if job.condition in LEARNING_CONDITIONS:
        records = _run_learner(job, env, skill, situations, save, objects)
        payload = {"key": job.key, "records": records, "wall_s": time.perf_counter() - wall0}
        _write_json(Path(job.path), payload)
        return job.path, len(records), payload["wall_s"]
    if job.theta is not None:
        policy = Baseline(job.condition, SkillParams.from_dict(job.theta))
    else:
        policy = make_baseline(job.condition, place=job.task == "pick_place")
    records: list[dict[str, Any]] = []
    for situation in situations:

        def keep(result: AttemptResult, record: dict, situation=situation) -> None:
            if save is not None and result.episode is not None:
                name = f"seed{situation.seed}_s{situation.index:03d}_a{record['attempt']}"
                result.episode.save(save / job.condition / name)

        records += run_situation(env, skill, policy, situation, job.max_attempts, keep)
    payload = {"key": job.key, "records": records, "wall_s": time.perf_counter() - wall0}
    _write_json(Path(job.path), payload)
    return job.path, len(records), payload["wall_s"]


def _run_learner(
    job: Job,
    env: PandaVFAEnv,
    skill: GraspLiftSkill,
    situations: Sequence[Situation],
    save: Path | None,
    objects: OB.ObjectConfig | None = None,
) -> list[dict[str, Any]]:
    """A learning condition on one seed: a fresh memory (T15) or bandit (T17a),
    situations in order (``memory_frozen``: after the warm-up seed's, unscored).

    The policy sees only the situation's opaque key and what the core's percepts
    hold; the scoring record (``attempt_record``) never reaches it.
    """
    from vfa.cognition.cognitive_core import CognitiveCore
    from vfa.cognition.retry import solve_situation
    from vfa.cognition.system2 import RetrievalPolicy
    from vfa.experiments.bandit import make_policy

    if job.condition in BANDIT_CONDITIONS:
        rng = np.random.default_rng((job.seed, BANDIT_STREAM))
        policy = make_policy(job.condition, rng, job.key["bandit"])  # the key's arms and edges
    elif job.task == "pick_place":
        from vfa.memory.embeddings import PlaceEmbedder

        policy = RetrievalPolicy(embedder=PlaceEmbedder(), **MEMORY_POLICIES[job.condition])
    else:
        policy = RetrievalPolicy(**MEMORY_POLICIES.get(job.condition, {}))
    core = CognitiveCore(policy)
    if job.condition == FROZEN:
        warmup = job.key["warmup"]
        objects = OB.load_object_config() if objects is None else objects
        for situation in make_situations(warmup["seed"], warmup["situations"], objects):
            solve_situation(
                env,
                core,
                skill,
                job.max_attempts,
                key=situation.key,
                seed_of=lambda a, s=situation: env_seed(s.seed, s.index, a),
                options={"object": situation.spec},
            )
        policy.freeze()
    records: list[dict[str, Any]] = []
    for situation in situations:

        def score(result: AttemptResult, attempt: int, situation=situation) -> None:
            records.append(attempt_record(result, situation, attempt))
            if save is not None and result.episode is not None:
                name = f"seed{situation.seed}_s{situation.index:03d}_a{attempt}"
                result.episode.save(save / job.condition / name)

        solve_situation(
            env,
            core,
            skill,
            job.max_attempts,
            key=situation.key,
            seed_of=lambda a, s=situation: env_seed(s.seed, s.index, a),
            options={"object": situation.spec},
            on_attempt=score,
        )
    return records


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, allow_nan=False, default=_json_default))
    os.replace(tmp, path)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON-serializable")


def _chunk_done(path: Path, key: Mapping[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and doc.get("key") == dict(key)


def _config_hash(task: str = "lift") -> str:
    """The hash of the situation-defining files (benchmark.toml, objects.toml; for
    pick_place, its own config and objects.toml)."""
    if task == "pick_place":
        return benchmark_hash(PICK_PLACE_FILES)
    return benchmark_hash((CONFIG_PATH, OBJECTS_PATH))


def code_fingerprint() -> str:
    """SHA-256 over the source files (``CODE_FILES``), the vendored robot assets
    (hidden files such as ``.DS_Store`` left out), and the MuJoCo, numpy, and
    Gymnasium versions: what a re-run of a chunk depends on."""
    import gymnasium
    import mujoco

    root = Path(__file__).resolve().parents[1]
    files = sorted(p for pattern in CODE_FILES for p in root.glob(pattern))
    files += sorted(
        p
        for p in ASSETS_DIR.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(ASSETS_DIR).parts)
    )
    files.append(STATS_PATH)  # the memory keys' standardization (T11)
    if PLACE_STATS_PATH.exists():
        files.append(PLACE_STATS_PATH)  # the pick-and-place keys' (T19)
    versions = (
        f"mujoco {mujoco.__version__} numpy {np.__version__} gymnasium {gymnasium.__version__}"
    ).encode()
    return hashlib.sha256(benchmark_hash(files).encode() + versions).hexdigest()


def plan_condition(
    config: BenchmarkConfig,
    condition: str,
    profile: str,
    seeds: Sequence[int] | None = None,
    out: Path = DEFAULT_OUT,
    save_episodes: str | None = None,
) -> list[Job]:
    """The chunks of one condition on one profile (optionally only some seeds)."""
    out = _out(config, out)
    prof = config.profiles[profile]
    seeds = prof.seeds if seeds is None else tuple(seeds)
    learner = condition in LEARNING_CONDITIONS
    if not learner and condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {ALL_CONDITIONS}, got {condition!r}")
    if config.task == "pick_place" and condition not in PICK_PLACE_CONDITIONS:
        raise ValueError(
            f"the pick_place benchmark runs {PICK_PLACE_CONDITIONS}, not {condition!r}"
        )
    theta = None if learner else make_baseline(condition).fixed
    fixed = None if condition == "naive" or theta is None else theta.to_dict()
    chunk = prof.situations if learner else CHUNK
    code = code_fingerprint()
    extra: dict[str, Any] = {}
    if condition == FROZEN:
        if WARMUP["seed"] in prof.seeds or WARMUP["seed"] in seeds:
            raise ValueError(
                f"{FROZEN} needs a profile without its warm-up seed {WARMUP['seed']} (use heldout)"
            )
        extra["warmup"] = dict(WARMUP)
    if condition in BANDIT_CONDITIONS:
        from vfa.experiments import bandit

        extra["bandit"] = bandit.describe(bandit.load_config(), condition == "bandit")
    jobs = []
    for seed in seeds:
        for start in range(0, prof.situations, chunk):
            end = min(start + chunk, prof.situations)
            key = {
                "config": _config_hash(config.task),
                "code": code,
                "condition": condition,
                "theta": theta.to_dict() if theta is not None else None,
                "seed": seed,
                "start": start,
                "end": end,
                "max_attempts": config.max_attempts,
                **extra,
            }
            if config.task != "lift":
                key["task"] = config.task
            path = out / profile / condition / f"seed{seed}_{start:03d}-{end:03d}.json"
            jobs.append(
                Job(
                    condition,
                    seed,
                    start,
                    end,
                    str(path),
                    key,
                    fixed,
                    config.max_attempts,
                    config.noise,
                    save_episodes,
                    config.task,
                )
            )
    return jobs


def _out(config: BenchmarkConfig, out: Path) -> Path:
    """The output root: pick_place never writes under the lift's default root."""
    return PICK_PLACE_OUT if config.task == "pick_place" and out == DEFAULT_OUT else out


def plan_grid(config: BenchmarkConfig, out: Path = DEFAULT_OUT) -> list[Job]:
    """The calibration chunks: every grid θ, one attempt, on the calib profile."""
    prof = config.profiles["calib"]
    code = code_fingerprint()
    jobs = []
    for theta in grid_thetas(config.grid):
        tag = f"g{theta.grip_force:g}_a{theta.lift_accel:g}"
        for seed in prof.seeds:
            for start in range(0, prof.situations, CHUNK):
                end = min(start + CHUNK, prof.situations)
                key = {
                    "config": _config_hash(),
                    "code": code,
                    "condition": "grid",
                    "theta": theta.to_dict(),
                    "seed": seed,
                    "start": start,
                    "end": end,
                    "max_attempts": 1,
                }
                path = out / "calib" / "grid" / tag / f"seed{seed}_{start:03d}-{end:03d}.json"
                jobs.append(
                    Job("grid", seed, start, end, str(path), key, theta.to_dict(), 1, config.noise)
                )
    return jobs


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 4) - 2))


def run_jobs(jobs: Sequence[Job], workers: int, deadline_s: float) -> bool:
    """Run the chunks not done yet; True once every chunk is done."""
    todo = [j for j in jobs if not _chunk_done(Path(j.path), j.key)]
    print(f"{len(jobs) - len(todo)} of {len(jobs)} chunks done; {len(todo)} to run", flush=True)
    if not todo:
        return True
    t0 = time.perf_counter()
    if workers <= 1:
        for job in todo:
            if time.perf_counter() - t0 > deadline_s:
                break
            _report(run_chunk(job), t0)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = list(todo)
            running = set()
            while pending and len(running) < workers and time.perf_counter() - t0 <= deadline_s:
                running.add(pool.submit(run_chunk, pending.pop(0)))
            while running:
                done = next(as_completed(running))
                running.remove(done)
                _report(done.result(), t0)
                if pending and time.perf_counter() - t0 <= deadline_s:
                    running.add(pool.submit(run_chunk, pending.pop(0)))
    left = sum(not _chunk_done(Path(j.path), j.key) for j in jobs)
    if left:
        print(f"deadline: {left} chunks left; run the same command again to continue")
    return left == 0


def _report(done: tuple[str, int, float], t0: float) -> None:
    path, n, wall = done
    print(
        f"  {Path(path).parent.name}/{Path(path).name}: {n} attempts in {wall:.0f} s "
        f"(elapsed {time.perf_counter() - t0:.0f} s)",
        flush=True,
    )


def load_records(jobs: Sequence[Job]) -> dict[int, list[dict]]:
    """The records per seed of finished chunks (raises if a chunk is missing)."""
    by_seed: dict[int, list[dict]] = {}
    for job in jobs:
        path = Path(job.path)
        if not _chunk_done(path, job.key):
            raise FileNotFoundError(f"chunk not done: {path}")
        by_seed.setdefault(job.seed, []).extend(json.loads(path.read_text())["records"])
    return by_seed


# -------------------------------------------------------------------- metrics


def summarize(records: Sequence[dict], max_attempts: int) -> dict[str, Any]:
    """Metrics of one seed's records (refer to the module doc)."""
    by_situation: dict[int, list[dict]] = {}
    for r in records:
        by_situation.setdefault(r["situation"], []).append(r)
    n = len(by_situation)
    first, solved, tries, late, late_first, broke = 0, 0, [], 0, 0, 0
    per_class: dict[str, list[int]] = {}
    for index, rows in by_situation.items():
        rows = sorted(rows, key=lambda r: r["attempt"])
        win = next((r["attempt"] for r in rows if r["success"]), None)
        first += int(win == 1)
        solved += int(win is not None)
        broke += int(any(r["result"] == "BREAK" for r in rows))
        if index >= LATE_FROM:
            late += 1
            late_first += int(win == 1)
        tries.append(win if win is not None else max_attempts + 1)
        stats = per_class.setdefault(rows[0]["class"], [0, 0, 0])
        stats[0] += 1
        stats[1] += int(win == 1)
        stats[2] += int(win is not None)
    attempts = len(records)

    def rate(result: str) -> float:
        return sum(r["result"] == result for r in records) / attempts

    excess = [r["peak_force_excess"] for r in records if r["success"]]
    jerk = [r["force_jerk"] for r in records if r["success"] and r["force_jerk"] is not None]
    return {
        "situations": n,
        "attempts": attempts,
        "success_at_1": first / n,
        "success_at_k": solved / n,
        "success_at_1_late": late_first / late if late else None,
        "attempts_to_success": float(np.mean(tries)),
        "break_rate": rate("BREAK"),
        "break_situation_rate": broke / n,
        "drop_rate": rate("DROP"),
        "slip_rate": rate("SLIP"),
        "safety_stop_rate": rate("SAFETY_STOP"),
        **{f"{r.lower()}_rate": rate(r) for r in PLACE_RESULTS},
        "peak_force_excess": float(np.mean(excess)) if excess else None,
        "force_jerk": float(np.mean(jerk)) if jerk else None,
        "per_class": {
            cls: {"situations": s[0], "success_at_1": s[1] / s[0], "success_at_k": s[2] / s[0]}
            for cls, s in sorted(per_class.items())
        },
    }


METRICS: tuple[str, ...] = (
    "success_at_1",
    "success_at_k",
    "success_at_1_late",
    "attempts_to_success",
    "break_rate",
    "break_situation_rate",
    "drop_rate",
    "slip_rate",
    "safety_stop_rate",
    *(f"{r.lower()}_rate" for r in PLACE_RESULTS),
    "peak_force_excess",
    "force_jerk",
    "attempts",
)


def first_attempts(records: Sequence[dict]) -> dict[int, bool]:
    """Situation index → did attempt 1 succeed."""
    return {r["situation"]: bool(r["success"]) for r in records if r["attempt"] == 1}


def learning(
    config: BenchmarkConfig,
    condition: str,
    profile: str,
    out: Path = DEFAULT_OUT,
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """success@1 per half and per block of ``BLOCK`` situations, pooled over the
    seeds (default: the profile's), for ``condition`` and ``naive``; a condition
    whose chunks are not all done is listed under ``missing``."""
    n = config.profiles[profile].situations
    half = n // 2
    result: dict[str, Any] = {"situations": n, "conditions": {}, "missing": []}
    for name in dict.fromkeys((condition, "naive")):
        jobs = plan_condition(config, name, profile, seeds, out=out)
        if not all(_chunk_done(Path(j.path), j.key) for j in jobs):
            result["missing"].append(name)
            continue
        firsts = [first_attempts(r) for r in load_records(jobs).values()]
        pooled = [(i, ok) for f in firsts for i, ok in f.items()]

        def rate(lo: int, hi: int, pooled=pooled) -> float:
            sel = [ok for i, ok in pooled if lo <= i < hi]
            return sum(sel) / len(sel)

        result["conditions"][name] = {
            "first_half": rate(0, half),
            "second_half": rate(half, n),
            "blocks": [rate(b, min(b + BLOCK, n)) for b in range(0, n, BLOCK)],
            "n_per_half": len([1 for i, _ in pooled if i < half]),
        }
    return result


def print_learning(
    config: BenchmarkConfig,
    condition: str,
    profile: str,
    out: Path,
    seeds: Sequence[int] | None = None,
) -> None:
    report = learning(config, condition, profile, out, seeds)
    for name in report["missing"]:
        print(f"{name}: chunks missing or stale on this --out; no learning report")
    half = report["situations"] // 2
    for name, c in report["conditions"].items():
        print(
            f"{name}: success@1 situations 0-{half - 1} {c['first_half']:.3f}, "
            f"{half}-{report['situations'] - 1} {c['second_half']:.3f} "
            f"(pooled, {c['n_per_half']} per half); per block of {BLOCK}: "
            + " ".join(f"{b:.2f}" for b in c["blocks"])
        )


def combine(per_seed: Mapping[int, dict]) -> dict[str, Any]:
    """Mean and sample standard deviation (ddof 1) over seeds of each scalar metric."""
    out: dict[str, Any] = {"seeds": sorted(per_seed)}
    for k in METRICS:
        values = [s[k] for s in per_seed.values() if s[k] is not None]
        out[k] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    return out


def aggregate(
    config: BenchmarkConfig, profile: str, out: Path = DEFAULT_OUT, partial: bool = False
) -> dict:
    """Metrics of every condition → ``<profile>/metrics.json``.

    A condition with missing or stale chunks is listed under ``missing``; unless
    ``partial``, any missing chunk raises ``FileNotFoundError`` before writing.
    """
    out = _out(config, out)
    pick_place = config.task == "pick_place"
    if pick_place:
        frozen = benchmark_hash(PICK_PLACE_FILES)
    else:
        frozen = benchmark_hash() if BEST_FIXED_PATH.exists() else None
    result: dict[str, Any] = {
        "profile": profile,
        "task": config.task,
        "benchmark_hash": frozen,
        "code": code_fingerprint(),
        "max_attempts": config.max_attempts,
        "noise": config.noise,
        "conditions": {},
        "missing": {},
    }
    for condition in PICK_PLACE_CONDITIONS if pick_place else ALL_CONDITIONS:
        if condition in LEARNING_CONDITIONS and not any((out / profile / condition).glob("*.json")):
            continue  # a learning condition that was never run on this --out
        jobs = plan_condition(config, condition, profile, out=out)
        missing = [j.path for j in jobs if not _chunk_done(Path(j.path), j.key)]
        if missing:
            result["missing"][condition] = len(missing)
            continue
        by_seed = load_records(jobs)
        per_seed = {s: summarize(r, config.max_attempts) for s, r in sorted(by_seed.items())}
        result["conditions"][condition] = {"per_seed": per_seed, **combine(per_seed)}
    if result["missing"] and not partial:
        raise FileNotFoundError(
            f"chunks missing or stale on profile {profile!r}: {result['missing']} "
            "(run the conditions, or pass --partial)"
        )
    _write_json(out / profile / "metrics.json", result)
    return result


def select_best_fixed(config: BenchmarkConfig, out: Path = DEFAULT_OUT) -> dict[str, Any]:
    """The calibration report: success@1 per grid θ, the winner, per-class ceilings."""
    rows = []
    jobs = plan_grid(config, out)
    by_theta: dict[str, list[dict]] = {}
    for job in jobs:
        tag = json.dumps(job.theta, sort_keys=True)
        path = Path(job.path)
        if not _chunk_done(path, job.key):
            raise FileNotFoundError(f"calibration chunk not done: {path}")
        by_theta.setdefault(tag, []).extend(json.loads(path.read_text())["records"])
    classes: set[str] = set()
    for tag, records in by_theta.items():
        theta = json.loads(tag)
        wins: dict[str, list[int]] = {}
        for r in records:
            w = wins.setdefault(r["class"], [0, 0])
            w[0] += int(r["success"])
            w[1] += 1
            classes.add(r["class"])
        total = sum(w[0] for w in wins.values()) / sum(w[1] for w in wins.values())
        rows.append(
            {
                "theta": theta,
                "success_at_1": total,
                "per_class": {c: w[0] / w[1] for c, w in sorted(wins.items())},
                "breaks": sum(r["result"] == "BREAK" for r in records) / len(records),
                "n": len(records),
            }
        )
    # The rule: highest success@1; ties to the lower grip, then the lower lift accel.
    rows.sort(
        key=lambda r: (-r["success_at_1"], r["theta"]["grip_force"], r["theta"]["lift_accel"])
    )
    best = rows[0]
    ceiling = {}
    for cls in sorted(classes):
        top = max(rows, key=lambda r: (r["per_class"][cls], -r["theta"]["grip_force"]))
        ceiling[cls] = {"success_at_1": top["per_class"][cls], "theta": top["theta"]}
    return {"best": best, "ceiling": ceiling, "grid": rows}


def write_best_fixed(report: Mapping[str, Any], path: Path = BEST_FIXED_PATH) -> None:
    best = report["best"]
    theta = SkillParams.from_dict(best["theta"])
    lines = [
        "# best_fixed baseline θ (T10): the single grid θ with the highest success@1 on the",
        "# calibration seeds (vfa.experiments.benchmark calibrate). Frozen with the",
        "# benchmark (configs/benchmark.sha256): do not edit.",
        "",
        "[theta]",
        *(f"{k} = {v!r}" for k, v in theta.to_dict().items()),
        "",
        "[calibration]",
        f"success_at_1 = {best['success_at_1']!r}",
        f"attempts = {best['n']}",
        *(f"{c} = {v!r}" for c, v in best["per_class"].items()),
        "",
    ]
    path.write_text("\n".join(lines))


# ------------------------------------------------------------------------ CLI


def throughput(config: BenchmarkConfig, attempts: int) -> dict[str, float]:
    """Wall time per attempt on one core (naive θ over the classes)."""
    logging.disable(logging.CRITICAL)
    objects = OB.load_object_config()
    env, skill = make_env(config), GraspLiftSkill()
    walls, sims = [], []
    for situation in make_situations(0, attempts, objects):
        r = run_attempt(
            env,
            skill,
            NAIVE,
            seed=env_seed(0, situation.index, 1),
            options={"object": situation.spec},
        )
        walls.append(r.wall_s)
        sims.append(r.sim_s)
    median, p95 = np.percentile(walls, [50, 95])
    return {
        "attempts": attempts,
        "wall_median_s": float(median),
        "wall_p95_s": float(p95),
        "sim_mean_s": float(np.mean(sims)),
        "attempts_per_min": 60.0 / float(np.mean(walls)),
        "load": os.getloadavg()[0],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition", choices=ALL_CONDITIONS, required=True)
    parser.add_argument("--profile", default="quick")
    parser.add_argument("--seed", type=int, action="append", help="only these seeds")
    _common(parser)
    parser.add_argument("--save-episodes", default=None, help="save every episode here")
    return parser


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--deadline", type=float, default=DEFAULT_DEADLINE_S, help="seconds")
    parser.add_argument("--out", type=Path, default=None, help=OUT_HELP)
    _task_option(parser)


OUT_HELP = "output root (default runs/benchmark; with --task pick_place runs/benchmark_pick_place)"


def _task_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task",
        choices=TASKS,
        default="lift",
        help="lift (configs/benchmark.toml) or pick_place (configs/benchmark_pick_place.toml; "
        "default --out runs/benchmark_pick_place)",
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv and not argv[0].startswith("-") else "run"
    if argv and argv[0] == command:
        argv = argv[1:]
    task_parser = argparse.ArgumentParser(add_help=False)
    _task_option(task_parser)
    task = task_parser.parse_known_args(argv)[0].task
    pick_place = task == "pick_place"
    config = load_pick_place_config() if pick_place else load_benchmark_config()

    def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
        """Parse, then resolve the default output root (pick_place has its own)."""
        args = parser.parse_args(argv)
        args.out = _out(config, DEFAULT_OUT if args.out is None else args.out)
        return args

    if pick_place and command not in ("run", "aggregate", "freeze"):
        raise SystemExit(f"the pick_place benchmark has no {command!r} command")
    if command == "run":
        args = parse(build_parser())
        if args.profile not in config.profiles:
            raise SystemExit(f"--profile must be one of {sorted(config.profiles)}")
        allowed = config.profiles[args.profile].seeds
        if args.seed and not set(args.seed) <= set(allowed):
            raise SystemExit(f"--seed must be among the {args.profile} seeds {list(allowed)}")
        try:
            jobs = plan_condition(
                config,
                args.condition,
                args.profile,
                args.seed,
                args.out,
                None if args.save_episodes is None else str(Path(args.save_episodes).resolve()),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.save_episodes is not None:
            skipped = sum(_chunk_done(Path(j.path), j.key) for j in jobs)
            if skipped:
                print(f"note: {skipped} chunks are done already; their episodes are not saved")
        t0 = time.perf_counter()
        if args.condition in LEARNING_CONDITIONS:  # "naive" first: the learning report's bar
            naive = plan_condition(config, "naive", args.profile, args.seed, args.out)
            if not run_jobs(naive, args.workers, args.deadline):
                return 3
        left = max(0.0, args.deadline - (time.perf_counter() - t0))
        done = run_jobs(jobs, args.workers, left)
        if done:
            by_seed = load_records(jobs)
            if args.condition in LEARNING_CONDITIONS:
                print_learning(config, args.condition, args.profile, args.out, args.seed)
            for seed, records in sorted(by_seed.items()):
                m = summarize(records, config.max_attempts)
                print(
                    f"{args.condition} seed {seed}: success@1 {m['success_at_1']:.3f} "
                    f"success@{config.max_attempts} {m['success_at_k']:.3f} attempts-to-success "
                    f"{m['attempts_to_success']:.2f} break {m['break_rate']:.3f} "
                    f"drop {m['drop_rate']:.3f} ({m['situations']} situations)"
                )
        return 0 if done else 3
    if command == "aggregate":
        parser = argparse.ArgumentParser(description="aggregate chunk files into metrics.json")
        parser.add_argument("--profile", default="quick" if pick_place else "full")
        parser.add_argument("--out", type=Path, default=None, help=OUT_HELP)
        parser.add_argument("--partial", action="store_true", help="allow missing conditions")
        _task_option(parser)
        args = parse(parser)
        if args.profile not in config.profiles:
            raise SystemExit(f"--profile must be one of {sorted(config.profiles)}")
        try:
            result = aggregate(config, args.profile, args.out, args.partial)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 3
        for name, c in result["conditions"].items():
            print(
                f"{name:15s} success@1 {_pm(c['success_at_1'])}"
                f"  success@k {_pm(c['success_at_k'])}"
                f"  late {_pm(c['success_at_1_late'])}"
                f"  attempts {_pm(c['attempts_to_success'])}"
                f"  break {_pm(c['break_rate'])}"
            )
        if result["missing"]:
            print(f"missing chunks: {result['missing']}")
        print(f"wrote {args.out / args.profile / 'metrics.json'}")
        return 0
    if command == "freeze":
        if pick_place:
            PICK_PLACE_HASH.write_text(benchmark_hash(PICK_PLACE_FILES) + "\n")
            print(f"wrote {PICK_PLACE_HASH}: {PICK_PLACE_HASH.read_text().strip()}")
            return 0
        HASH_PATH.write_text(benchmark_hash() + "\n")
        print(f"wrote {HASH_PATH}: {HASH_PATH.read_text().strip()}")
        return 0
    if command == "calibrate":
        parser = argparse.ArgumentParser(description="best_fixed grid search on the calib seeds")
        _common(parser)
        parser.add_argument("--write", action="store_true", help="write configs/best_fixed.toml")
        args = parse(parser)
        jobs = plan_grid(config, args.out)
        if not run_jobs(jobs, args.workers, args.deadline):
            return 3
        report = select_best_fixed(config, args.out)
        _write_json(args.out / "calib" / "calibration.json", report)
        best = report["best"]
        print(
            f"best_fixed: {SkillParams.from_dict(best['theta']).describe()} "
            f"success@1 {best['success_at_1']:.3f} (n={best['n']}) per class {best['per_class']}"
        )
        for cls, c in report["ceiling"].items():
            print(
                f"  ceiling {cls:15s} {c['success_at_1']:.3f}"
                f" at grip {c['theta']['grip_force']:g} N"
            )
        if args.write:
            write_best_fixed(report)
            print(f"wrote {BEST_FIXED_PATH}")
        return 0
    if command == "throughput":
        parser = argparse.ArgumentParser(description="wall time per attempt on one core")
        parser.add_argument("--attempts", type=int, default=60)
        _task_option(parser)
        args = parser.parse_args(argv)
        print(json.dumps(throughput(config, args.attempts), indent=1))
        return 0
    raise SystemExit(f"unknown command {command!r} (run, aggregate, calibrate, freeze, throughput)")


def _pm(stat: Mapping[str, float | None]) -> str:
    mean, std = stat["mean"], stat["std"]
    if mean is None:
        return "n/a"
    return f"{mean:.3f}" + ("" if std is None else f" ± {std:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
