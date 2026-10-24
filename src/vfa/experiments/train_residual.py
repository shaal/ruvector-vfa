"""Train the residual System 1 policy with numpy evolution strategies (T21).

Usage::

    uv run python -m vfa.experiments.train_residual --quick --workers 8    # a few generations
    uv run python -m vfa.experiments.train_residual --workers 8            # re-run until done
    uv run python -m vfa.experiments.train_residual --workers 8 --install  # then copy the weights

**Data: memory's own grasps, on the calibration seeds only.** The trainer first
runs ``memory_full`` on the benchmark's ``calib`` profile (seeds 0 and 1,
chunked and resumable like any benchmark run). Every attempt of that run becomes
a replay episode: its situation, attempt (the env reset seed), and the θ that
ran at the end. The residual acts after the probe, so replaying that θ from the
start gives memory's grips without System 2 in the loop and without any class
label. That θ already holds the reflex and escalation raises, so the replays
slip less than the original attempts (docs/RESULTS.md §11.6). Seed 0's attempts
are the training pool, seed 1's the validation set.

**Reward** of one episode: ``success − LAMBDA·jerk/J0 − MU·max(excess, 0)/E0``
(force jerk and peak-force excess as ``benchmark.attempt_record`` computes them;
the excess reads the privileged mass and friction, in the trainer only: the
policy's inputs stay observable).

**ES** (OpenAI-style): each generation draws ``PAIRS`` antithetic perturbations
``±SIGMA·ε`` of the center, scores every candidate on the same ``BATCH``
training episodes (common random numbers), turns the scores into centered
ranks, and takes an Adam step of size ``LR`` along the estimated gradient. The
center starts at zero (the untrained residual). After each step the center is
scored on the whole validation set; that curve, smoothed by a running mean
over ``SMOOTH`` generations, is the training curve. The hyperparameters, the
smoothing rule, and the data were fixed before the first training run.

**Resumable.** After every generation ``<out>/state.npz`` (center, Adam
moments, generation) and ``<out>/curve.json`` are written; a command stops
between generations at ``--deadline`` and the same command continues. When all
generations are done it writes ``<out>/residual.npz`` and ``<out>/curve.png``;
``--install`` copies the weights to ``configs/residual.npz``, which the
benchmark's ``memory_full+residual`` condition reads.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from vfa.experiments import benchmark as B
from vfa.skills.residual import RESIDUAL_PATH, ResidualPolicy

DEFAULT_OUT = B.REPO_ROOT / "runs" / "residual"
REPLAY_OUT = DEFAULT_OUT / "replay"  # memory_full on calib, shared by every training run
PAIRS = 8  # antithetic pairs per generation (16 candidates)
SIGMA = 0.5  # perturbation scale, in weight units
LR = 0.1  # Adam step
BATCH = 16  # training episodes per generation
GENERATIONS = 30
QUICK = {"generations": 4, "batch": 8}
SMOOTH = 5  # the training curve: running mean over this many generations
LAMBDA, J0 = 0.25, 30.0  # jerk penalty and scale [N/s]
MU, E0 = 0.05, 10.0  # peak-force excess penalty and scale [N]
DEFAULT_DEADLINE_S = 420.0
CHUNK = 8  # rollouts per pool task
STREAM = 0x45533231  # "ES21": the trainer's generator


def reward(record: dict[str, Any]) -> float:
    """The reward of one replayed attempt (refer to the module doc)."""
    jerk = record["force_jerk"] or 0.0
    excess = max(record["peak_force_excess"] or 0.0, 0.0)
    return float(record["success"]) - LAMBDA * jerk / J0 - MU * excess / E0


# ----------------------------------------------------------------- replay set


def replay_set(out: Path, workers: int, deadline: float) -> dict[int, list[dict]] | None:
    """memory_full's attempts on the calib profile per seed, run under ``out`` (None
    until all chunks ran)."""
    config = B.load_benchmark_config()
    jobs = B.plan_condition(config, "memory_full", "calib", out=out)
    if not B.run_jobs(jobs, workers, deadline):
        return None
    episodes: dict[int, list[dict]] = {}
    for seed, records in sorted(B.load_records(jobs).items()):
        episodes[seed] = [
            {
                "seed": seed,
                "situation": r["situation"],
                "attempt": r["attempt"],
                "theta": r["theta_final"],
            }
            for r in records
        ]
    return episodes


# ------------------------------------------------------------------- rollouts


def rollout(task: tuple[list[float], list[dict]]) -> list[float]:
    """Rewards of one residual (a weight vector) on replay episodes (a pool task)."""
    logging.disable(logging.CRITICAL)
    from vfa.skills.params import SkillParams
    from vfa.skills.pick import GraspLiftSkill, run_attempt

    vector, episodes = task
    env, _, objects = B._worker_env(B.load_benchmark_config().noise)
    skill = GraspLiftSkill(residual=ResidualPolicy.from_vector(vector))
    out = []
    for ep in episodes:
        situation = B.make_situations(ep["seed"], ep["situation"] + 1, objects, ep["situation"])[0]
        result = run_attempt(
            env,
            skill,
            SkillParams.from_dict(ep["theta"]),
            seed=B.env_seed(ep["seed"], ep["situation"], ep["attempt"]),
            options={"object": situation.spec},
        )
        record = B.attempt_record(result, situation, ep["attempt"])
        out.append(
            [
                reward(record),
                float(record["success"]),
                record["force_jerk"] if record["force_jerk"] is not None else np.nan,
                record["peak_force_excess"] if record["peak_force_excess"] is not None else np.nan,
            ]
        )
    return out


def evaluate(
    pool: ProcessPoolExecutor, vectors: list[np.ndarray], episodes: list[dict]
) -> np.ndarray:
    """(len(vectors), len(episodes), 4) rows: reward, success, jerk, excess."""
    tasks, index = [], []
    for i, v in enumerate(vectors):
        for k in range(0, len(episodes), CHUNK):
            tasks.append((v.tolist(), episodes[k : k + CHUNK]))
            index.append(i)
    rows: list[list[list[float]]] = [[] for _ in vectors]
    for i, result in zip(index, pool.map(rollout, tasks), strict=True):
        rows[i].extend(result)
    return np.array(rows, dtype=float)


def summary(rows: np.ndarray) -> dict[str, float | None]:
    """Mean reward and success; jerk and excess over the successful episodes (None
    without one)."""
    ok = rows[:, 1] > 0.5
    jerk = rows[ok, 2][np.isfinite(rows[ok, 2])]
    excess = rows[ok, 3][np.isfinite(rows[ok, 3])]
    return {
        "reward": float(np.mean(rows[:, 0])),
        "success": float(np.mean(rows[:, 1])),
        "jerk": float(np.mean(jerk)) if jerk.size else None,
        "excess": float(np.mean(excess)) if excess.size else None,
    }


def fingerprint(episodes: dict[int, list[dict]], batch: int) -> str:
    """What a resumed run must share with the saved one: the replay, the settings, and
    the code that runs an attempt (``benchmark.code_fingerprint``: the skill, the
    residual, the env)."""
    import hashlib

    settings = {"pairs": PAIRS, "sigma": SIGMA, "lr": LR, "batch": batch, "lambda": LAMBDA}
    settings |= {
        "j0": J0,
        "mu": MU,
        "e0": E0,
        "stream": STREAM,
        "dim": ResidualPolicy().to_vector().size,
        "code": B.code_fingerprint(),
    }
    doc = json.dumps({"settings": settings, "replay": episodes}, sort_keys=True)
    return hashlib.sha256(doc.encode()).hexdigest()


# -------------------------------------------------------------------------- ES


def centered_ranks(scores: np.ndarray) -> np.ndarray:
    """Fitness shaping: ranks scaled to [−0.5, 0.5]."""
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores, kind="stable")] = np.arange(len(scores))
    return ranks / (len(scores) - 1) - 0.5


def es_gradient(eps: np.ndarray, plus: np.ndarray, minus: np.ndarray, sigma: float) -> np.ndarray:
    """The antithetic ES gradient estimate from the shaped scores of ±σε."""
    shaped = centered_ranks(np.concatenate([plus, minus]))
    n = len(plus)
    return (shaped[:n] - shaped[n:]) @ eps / (2.0 * n * sigma)


class Adam:
    def __init__(self, dim: int, lr: float, b1: float = 0.9, b2: float = 0.999) -> None:
        self.lr, self.b1, self.b2 = lr, b1, b2
        self.m, self.v, self.t = np.zeros(dim), np.zeros(dim), 0

    def step(self, grad: np.ndarray) -> np.ndarray:
        """The update for gradient ascent."""
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mhat = self.m / (1 - self.b1**self.t)
        vhat = self.v / (1 - self.b2**self.t)
        return self.lr * mhat / (np.sqrt(vhat) + 1e-8)


def smoothed(values: list[float], window: int = SMOOTH) -> list[float]:
    return [float(np.mean(values[max(0, i - window + 1) : i + 1])) for i in range(len(values))]


def train(out: Path, generations: int, batch: int, workers: int, deadline: float) -> bool:
    """Run generations until ``generations`` or the deadline; True once done."""
    t0 = time.perf_counter()
    episodes = replay_set(REPLAY_OUT, workers, deadline)
    if episodes is None:
        print("replay set: chunks left; run the same command again")
        return False
    train_pool, valid = episodes[min(episodes)], episodes[max(episodes)]
    dim = ResidualPolicy().to_vector().size
    state_path, curve_path = out / "state.npz", out / "curve.json"
    center, adam, gen = np.zeros(dim), Adam(dim, LR), 0
    curve: list[dict[str, Any]] = []
    key = fingerprint(episodes, batch)
    if state_path.exists():
        with np.load(state_path) as s:
            if "fingerprint" not in s.files or str(s["fingerprint"]) != key:
                raise SystemExit(
                    f"{out} holds a run with another replay or settings: use a new --out"
                )
            center, gen = s["center"], int(s["generation"])
            adam.m, adam.v, adam.t = s["m"], s["v"], int(s["t"])
        if curve_path.exists():  # the curve is written first: drop what the state lacks
            curve = [
                c
                for c in json.loads(curve_path.read_text())["generations"]
                if c["generation"] <= gen
            ]
        if not curve or curve[-1]["generation"] != gen:
            curve = []  # rebuilt below from the saved center's score
    if gen and not curve:
        raise SystemExit(f"{out}: the curve does not match the saved state; use a new --out")
    print(f"replay: {len(train_pool)} training, {len(valid)} validation episodes; generation {gen}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if not curve:  # generation 0: the untrained center
            curve.append({"generation": 0, **summary(evaluate(pool, [center], valid)[0])})
            _save(out, center, adam, gen, curve, key)
        last = 0.0  # the last generation's wall time: the next one must fit the deadline
        while gen < generations and time.perf_counter() - t0 + last < deadline:
            started = time.perf_counter()
            rng = np.random.default_rng((STREAM, gen))
            eps = rng.standard_normal((PAIRS, dim))
            pick = rng.choice(len(train_pool), size=min(batch, len(train_pool)), replace=False)
            batch_eps = [train_pool[i] for i in pick]
            candidates = [center + SIGMA * e for e in eps] + [center - SIGMA * e for e in eps]
            rows = evaluate(pool, candidates, batch_eps)
            scores = rows[:, :, 0].mean(axis=1)
            center = center + adam.step(es_gradient(eps, scores[:PAIRS], scores[PAIRS:], SIGMA))
            gen += 1
            valid_rows = evaluate(pool, [center], valid)[0]
            curve.append(
                {
                    "generation": gen,
                    "train_batch_reward": float(scores.mean()),
                    **summary(valid_rows),
                }
            )
            _save(out, center, adam, gen, curve, key)
            last = time.perf_counter() - started
            c = curve[-1]
            print(
                f"generation {gen}: validation reward {c['reward']:.4f} success {c['success']:.3f} "
                f"jerk {_fmt(c['jerk'], 1)} excess {_fmt(c['excess'], 2)} "
                f"({time.perf_counter() - t0:.0f} s)",
                flush=True,
            )
    if gen < generations:
        print(f"deadline: {gen} of {generations} generations; run the same command again")
        return False
    ResidualPolicy.from_vector(center).save(out / "residual.npz")
    _plot(curve, out / "curve.png")
    rewards = [c["reward"] for c in curve]
    smooth = smoothed(rewards)
    monotone = all(b >= a for a, b in zip(smooth, smooth[1:], strict=False))
    print(
        f"done: validation reward {rewards[0]:.4f} → {rewards[-1]:.4f}; smoothed (window {SMOOTH}) "
        f"{'non-decreasing' if monotone else 'NOT non-decreasing'}; wrote {out / 'residual.npz'}"
    )
    return True


def _fmt(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _save(out: Path, center: np.ndarray, adam: Adam, gen: int, curve: list[dict], key: str) -> None:
    """The curve first, then the state: after a crash between the two, a resume drops
    the curve's extra generation and runs it again."""
    out.mkdir(parents=True, exist_ok=True)
    rewards = [c["reward"] for c in curve]
    doc = {
        "hyperparameters": {
            "pairs": PAIRS,
            "sigma": SIGMA,
            "lr": LR,
            "lambda": LAMBDA,
            "j0": J0,
            "mu": MU,
            "e0": E0,
            "smooth": SMOOTH,
        },
        "generations": curve,
        "smoothed_reward": smoothed(rewards),
    }
    B._write_json(out / "curve.json", doc)
    tmp = out / "state.tmp.npz"
    np.savez(tmp, center=center, m=adam.m, v=adam.v, t=adam.t, generation=gen, fingerprint=key)
    tmp.replace(out / "state.npz")


def _plot(curve: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens = [c["generation"] for c in curve]
    rewards = [c["reward"] for c in curve]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(gens, rewards, "o-", color="#888", lw=1, ms=3, label="validation reward")
    axes[0].plot(gens, smoothed(rewards), color="#1f6feb", lw=2, label=f"running mean of {SMOOTH}")
    axes[0].set_xlabel("ES generation")
    axes[0].set_ylabel("reward")
    axes[0].legend(frameon=False)
    jerk = [np.nan if c["jerk"] is None else c["jerk"] for c in curve]
    axes[1].plot(gens, jerk, color="#d1242f", label="force jerk [N/s]")
    axes[1].set_xlabel("ES generation")
    ax2 = axes[1].twinx()
    ax2.plot(gens, [c["success"] for c in curve], color="#1a7f37", label="success")
    axes[1].set_ylabel("force jerk [N/s]")
    ax2.set_ylabel("success")
    lines = axes[1].get_lines() + ax2.get_lines()
    axes[1].legend(lines, [ln.get_label() for ln in lines], frameon=False, loc="center right")
    fig.suptitle("Residual ES on the calibration replay (validation set: calib seed 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help=f"{QUICK['generations']} generations")
    parser.add_argument("--workers", type=int, default=B.default_workers())
    parser.add_argument("--deadline", type=float, default=DEFAULT_DEADLINE_S, help="seconds")
    parser.add_argument("--out", type=Path, default=None, help="default runs/residual/<full|quick>")
    parser.add_argument("--install", action="store_true", help="copy the weights to configs/")
    args = parser.parse_args(argv)
    out = args.out or DEFAULT_OUT / ("quick" if args.quick else "full")
    generations = QUICK["generations"] if args.quick else GENERATIONS
    batch = QUICK["batch"] if args.quick else BATCH
    done = train(out, generations, batch, args.workers, args.deadline)
    if done and args.install:
        shutil.copyfile(out / "residual.npz", RESIDUAL_PATH)
        print(f"installed {RESIDUAL_PATH}")
    return 0 if done else 3


if __name__ == "__main__":
    raise SystemExit(main())
