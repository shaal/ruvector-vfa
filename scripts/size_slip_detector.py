"""T06: size the F/T slip detector's noise margin (docs/FEASIBILITY.md §8).

T05 handed T06 the detector with a warning: with white noise of 0.05 N on F_z
and 0.02 m/s² on a_z its held-out precision fell to 0.773, below the 0.9 bar
(FEASIBILITY §5). T06's detector (``vfa.sim.sensors.SlipDetector``) is T05's
plus, with noise only, a margin and a window: the drop must also exceed
``noise_k`` standard deviations of the filtered estimate (from the sensor's
noise spec), and the peak is the largest estimate of the last 0.5 s. Without
noise the detector is T05's, so only ``noise_k`` is sized here, for
``REALISTIC_NOISE``.

**Physics once, noise offline.** The detector does not act on the physics, so
each attempt runs once, noise-free, in T05's near-threshold harness
(``spike_grip.run_attempt``: margin 0.98, 1.0, 1.04, 1.3; release 0.9; five
classes). It logs the detector's inputs every tick: the tare window, the
squeeze, the lift and hold, and the ground truth (the cube's fall relative to
the ee). The replay adds white noise to F_z and a_z (the tare samples too) for
several noise seeds and runs the detector exactly as ``SensorSuite`` does: the
filter runs from the tare, the detector watches from the lift
(``watch_slip``). The replay is vectorized; ``--check`` compares it with
``SlipDetector`` itself (exact equality).

**Scoring.** Ground truth (T05): the cube falls more than 5 mm relative to the
ee. A firing counts as a true detection only from 250 ms before that moment on
(noise-free, T05's detector never fires earlier than 186 ms before it); an
earlier firing is a false alarm, and the slip a miss (the detector latches).
Each table gives the lag quantiles and the share of detections after the 5 mm
line, and 95 % intervals from a bootstrap over attempts (the noise seeds of one
attempt share its physics, so they are resampled together).

**Protocol.** Three disjoint cube sets: ``t05`` (samples 9100-9102, T05's
held-out cubes; the noise-free replay with T05's semantics must give T05's
counts), ``tune`` (9200-9205: the ``noise_k`` choice), ``holdout`` (9700-9705:
scored with the choice only; the choice reads ``tune`` and the noise model).
Earlier T06 designs used cubes 9300-9305, 9400-9405, 9500-9505, and 9600-9605
for their own held-out scores (FEASIBILITY §8); this protocol does not touch
them. Selection rule, fixed before ``holdout`` ran: the smallest ``noise_k``
that (a) reaches 0.9 precision and 0.8 recall as ``tune`` bootstrap 95 % lower
bounds under ``REALISTIC_NOISE`` and (b) keeps false alarms on a synthetic
10 s hold of a 50 g cube (the lightest in the classes) under
``REALISTIC_NOISE`` at most 5 of 100 seeds. The smallest margin keeps the most
sensitivity; (b) covers carries longer than the harness's 2 s, where a margin
that suits the harness false-alarms more and more on light cubes.

Usage::

    uv run python scripts/size_slip_detector.py --workers 8     # physics (resumable) + tables
    uv run python scripts/size_slip_detector.py --physics       # physics only
    uv run python scripts/size_slip_detector.py --aggregate     # tables only
    uv run python scripts/size_slip_detector.py --check         # vectorized replay == SlipDetector

Writes ``runs/slip_sizing/{chunks/*.npz, summary.md, metrics.json}``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spike_grip as SG  # noqa: E402  (the T05 harness, beside this script)

from vfa.sim import objects as OB  # noqa: E402
from vfa.sim.sensors import (  # noqa: E402
    REALISTIC_NOISE,
    T05_SLIP,
    LowPass,
    SlipConfig,
    SlipDetector,
)

OUT = SG.REPO_ROOT / "runs" / "slip_sizing"
SETS: dict[str, range] = {
    "t05": range(SG.NOISE_FIRST_FULL, SG.NOISE_FIRST_FULL + SG.NOISE_SAMPLES),
    "tune": range(9200, 9206),
    "holdout": range(9700, 9706),
}
VARIANTS = SG.NOISE_VARIANTS
# (σ F_z [N], σ a_z [m/s²]). White noise per axis in the sensor frame has the
# same σ on the world z axis (the rotation is orthonormal).
NOISE: dict[str, tuple[float, float]] = {
    "none": (0.0, 0.0),
    "t05_low": (0.02, 0.01),
    "t05_high": (0.05, 0.02),
    "realistic": (REALISTIC_NOISE.force, REALISTIC_NOISE.accel),
}
SEEDS = 5  # noise seeds per attempt and noisy level
NOISE_KS: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)
BAR = (0.9, 0.8)  # ROADMAP T05: precision >= 0.9, recall >= 0.8
EARLY_S = 0.25  # a firing earlier than this before the 5 mm line is a false alarm
MASS_BANDS: tuple[tuple[float, float], ...] = ((0.0, 0.075), (0.075, 0.1), (0.1, 0.15), (0.15, 9.9))
BOOTSTRAP = 2000
LONG_HOLD = (0.05, 10.0, 100, 5)  # rule (b): cube [kg], hold [s], seeds, allowed alarms
G = 9.81
DT = SG.DT
KEYS = ("tare_f", "tare_a", "sq_f", "sq_a", "t", "f_z", "a_z", "rel")
_CLASS_INDEX: dict[str, int] = {c: i for i, c in enumerate(OB.load_object_config().names)}


# ------------------------------------------------------------------ physics


def run_chunk(set_name: str, cls: str, path: str, fingerprint: str) -> tuple[str, int, float]:
    """All attempts of one (set, class): noise-free physics, per-tick logs → one .npz."""
    wall0 = time.perf_counter()
    config = OB.load_object_config()
    arm = SG.Arm()
    meta, logs = [], {k: [] for k in KEYS}
    for sample in SETS[set_name]:
        for v, (variant, factor) in enumerate(VARIANTS):
            spec = SG.sample_spec(config, cls, sample)
            grip, release = SG.harness_grip(spec, variant, factor)
            log: dict[str, list[float]] = {}
            r = SG.run_attempt(
                arm, spec, grip, SG.HARNESS_SPEED, sample, keep_trace=False, release=release,
                tick_log=log,
            )  # fmt: skip
            meta.append(
                {
                    "set": set_name,
                    "cls": cls,
                    "sample": sample,
                    "variant": v,
                    "mass": spec.mass,
                    "broken": r.broken,
                    "gt_slip": r.gt_slip,
                    "gt_slip_time": r.gt_slip_time,
                    "rel_drop": r.rel_drop,
                    "t05_fired": r.ft_slip,
                    "t05_min_ratio": r.ft_min_ratio,
                    "lift_ticks": len(log.get("t", [])),
                }
            )
            for k in KEYS:
                logs[k].append(np.asarray(log.get(k, []), dtype=float))
    arrays = {}
    for k in KEYS:
        arrays[k] = np.concatenate(logs[k]) if logs[k] else np.zeros(0)
        arrays[k + "_off"] = np.cumsum([0] + [len(x) for x in logs[k]])
    tmp = Path(path).with_suffix(".tmp.npz")
    meta_json = json.dumps({"fingerprint": fingerprint, "attempts": meta})
    np.savez_compressed(tmp, meta=np.array(meta_json), **arrays)
    os.replace(tmp, path)
    return Path(path).name, len(meta), time.perf_counter() - wall0


def chunk_path(out: Path, set_name: str, cls: str) -> Path:
    return out / "chunks" / f"{set_name}_{cls}.npz"


def chunk_current(path: Path, fingerprint: str) -> bool:
    """True if the chunk file exists, reads, and came from this code (else re-run it)."""
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as npz:
            meta = json.loads(str(npz["meta"]))
            whole = all(k in npz.files and k + "_off" in npz.files for k in KEYS)
            # Read every member: a damaged one fails its CRC here, and a member that is
            # not an .npy array comes back as raw bytes.
            arrays = all(isinstance(npz[k], np.ndarray) for k in npz.files)
            return bool(
                whole
                and arrays
                and isinstance(meta, dict)
                and isinstance(meta.get("attempts"), list)
                and meta["fingerprint"] == fingerprint
            )
    except Exception:  # noqa: BLE001 - any unreadable chunk is simply run again
        return False


def run_physics(out: Path, workers: int, deadline_s: float) -> bool:
    fingerprint = SG.code_fingerprint()
    todo = [
        (s, c)
        for s in SETS
        for c in _CLASS_INDEX
        if not chunk_current(chunk_path(out, s, c), fingerprint)
    ]
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    print(f"{len(todo)} chunks to run with {workers} workers")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_chunk, s, c, str(chunk_path(out, s, c)), fingerprint): (s, c)
            for s, c in todo
        }
        for fut in as_completed(futures):
            name, n, wall = fut.result()
            print(f"  {name}: {n} attempts in {wall:.1f} s")
            if time.perf_counter() - t0 > deadline_s:
                for f in futures:
                    f.cancel()
                print("deadline reached; re-run the same command to continue")
                return False
    return True


# ------------------------------------------------------------------- replay


@dataclass(slots=True)
class Attempt:
    meta: dict
    tare_f: np.ndarray
    tare_a: np.ndarray
    sq_f: np.ndarray
    sq_a: np.ndarray
    t: np.ndarray
    f_z: np.ndarray
    a_z: np.ndarray
    rel: np.ndarray


def load_attempts(out: Path, set_name: str) -> list[Attempt]:
    fingerprint = SG.code_fingerprint()
    attempts = []
    for cls in _CLASS_INDEX:
        path = chunk_path(out, set_name, cls)
        if not chunk_current(path, fingerprint):
            raise SystemExit(f"{path} is missing or stale: run the physics first")
        with np.load(path, allow_pickle=False) as npz:
            metas = json.loads(str(npz["meta"]))["attempts"]
            data = {k: npz[k] for k in npz.files if k != "meta"}
        for i, meta in enumerate(metas):
            parts = {k: data[k][data[k + "_off"][i] : data[k + "_off"][i + 1]] for k in KEYS}
            attempts.append(Attempt(meta, **parts))
    return [a for a in attempts if not a.meta["broken"] and a.t.size]


Row = tuple  # (attempt index, squeeze f, a, lift f, a, hand mass)


def noisy_inputs(attempts: list[Attempt], level: str, seeds: int, set_name: str) -> list[Row]:
    """Per (attempt, seed): the detector's inputs with noise, and the noisy tare."""
    sf, sa = NOISE[level]
    rows = []
    set_i = list(SETS).index(set_name)
    lvl_i = list(NOISE).index(level)
    for i, a in enumerate(attempts):
        m = a.meta
        for seed in range(seeds if sf or sa else 1):
            key = (set_i, _CLASS_INDEX[m["cls"]], m["sample"], m["variant"], lvl_i, seed)
            rng = np.random.default_rng(key)
            n_t, n_s, n_l = a.tare_f.size, a.sq_f.size, a.f_z.size
            ef = rng.standard_normal(n_t + n_s + n_l) * sf
            ea = rng.standard_normal(n_t + n_s + n_l) * sa
            tf, ta = a.tare_f + ef[:n_t], a.tare_a + ea[:n_t]
            hand = float(np.mean(tf)) / float(np.mean(ta))
            rows.append(
                (
                    i,
                    a.sq_f + ef[n_t : n_t + n_s],
                    a.sq_a + ea[n_t : n_t + n_s],
                    a.f_z + ef[n_t + n_s :],
                    a.a_z + ea[n_t + n_s :],
                    hand,
                )
            )
    return rows


def replay(
    rows: list[Row],
    noise: tuple[float, float],
    ks: tuple[float, ...],
    windows: tuple[int, ...],
    semantics: str = "suite",
    cutoff_hz: float = T05_SLIP.cutoff_hz,
) -> dict[tuple[float, int], np.ndarray]:
    """Vectorized ``SlipDetector``: per (noise_k, window), the debounced statistic
    (rows × lift ticks; NaN where the detector has no debounced value yet). It
    fires at a threshold on the first tick whose value is below it.

    ``semantics="suite"``: the filter starts at the first squeeze tick and the
    detector watches from the lift (``SensorSuite.watch_slip``). ``"t05"``: the
    filter starts at the first lift tick (the T05 spike).
    """
    lp = LowPass(cutoff_hz, DT)
    alpha, gain = lp.alpha, lp.noise_gain()
    n = len(rows)
    pre = [r[1].size if semantics == "suite" else 0 for r in rows]
    lift = [r[3].size for r in rows]
    width = max(p + q for p, q in zip(pre, lift, strict=True))
    x = np.full((n, width), np.nan)
    for k, (_, sq_f, sq_a, f, a, hand) in enumerate(rows):
        if semantics == "suite":
            x[k, : pre[k]] = sq_f / sq_a - hand
        x[k, pre[k] : pre[k] + lift[k]] = f / a - hand
    assert np.all(np.isfinite([r[5] for r in rows]))
    m = np.empty_like(x)
    m[:, 0] = x[:, 0]
    for j in range(1, width):
        prev = m[:, j - 1]
        m[:, j] = prev + alpha * (x[:, j] - prev)
    lw = max(lift)
    mass = np.full((n, lw), np.nan)
    start = np.empty(n)
    hand = np.array([r[5] for r in rows])
    for k in range(n):
        mass[k, : lift[k]] = m[k, pre[k] : pre[k] + lift[k]]
        start[k] = m[k, pre[k] - 1] if pre[k] else 0.0  # watch's peak, or a fresh 0
    seeded = np.column_stack([start, mass])
    all_time = np.fmax.accumulate(seeded, axis=1)[:, 1:]
    span = max(1, round(SlipConfig().peak_window_s / DT))
    windowed = _sliding_max(seeded, span)[:, 1:]
    valid = np.isfinite(mass)
    cols = np.arange(lw)
    sf, sa = noise
    xs = (hand[:, None] + mass) * sa
    sigma_w = np.sqrt(sf * sf + xs * xs)
    out = {}
    for k_noise in ks:
        unit = k_noise * gain / G
        noisy = unit > 0 and (sf > 0 or sa > 0)
        peak = windowed if noisy else all_time  # SlipDetector: the window only with noise
        armed = valid & (peak * G >= T05_SLIP.arm_load_n)
        first_armed = np.where(armed.any(axis=1), armed.argmax(axis=1), lw)
        safe_peak = np.where(armed, peak, 1.0)
        if noisy:
            stat = np.where(armed, (mass + unit * sigma_w) / safe_peak, np.nan)
        else:
            stat = np.where(armed, mass / safe_peak, np.nan)
        for w in windows:
            deb = stat.copy()
            for s in range(1, w):
                shifted = np.full_like(stat, np.nan)
                shifted[:, s:] = stat[:, :-s]
                deb = np.fmax(deb, shifted)
            ok = cols[None, :] >= (first_armed[:, None] + w - 1)
            out[(k_noise, w)] = np.where(ok & armed, deb, np.nan)
    return out


def _sliding_max(a: np.ndarray, span: int) -> np.ndarray:
    """Per row, the max of the last ``span`` columns (NaN ignored), by doubling."""
    out, p = a.copy(), 1
    while 2 * p <= span:
        shifted = np.full_like(out, np.nan)
        shifted[:, p:] = out[:, :-p]
        out = np.fmax(out, shifted)  # out[c] = max(a[c - 2p + 1 .. c])
        p *= 2
    rest = span - p
    if rest:
        shifted = np.full_like(out, np.nan)
        shifted[:, rest:] = out[:, :-rest]
        out = np.fmax(out, shifted)
    return out


def scalar_replay(row: Row, config: SlipConfig, noise, semantics: str = "suite") -> SlipDetector:
    """The same row through ``SlipDetector`` itself (the reference for ``--check``)."""
    _, sq_f, sq_a, f, a, hand = row
    det = SlipDetector(
        DT, G, config, track_windows=(1, 3, 5), watching=semantics != "suite", noise=noise
    )
    det.hand_mass = hand
    if semantics == "suite":
        for fz, az in zip(sq_f, sq_a, strict=True):
            det.update(0.0, float(fz), float(az))
        det.watch()
    for k, (fz, az) in enumerate(zip(f, a, strict=True)):
        det.update(k * DT, float(fz), float(az))
    return det


# ------------------------------------------------------------------ scoring


def score(attempts: list[Attempt], rows: list[Row], deb: np.ndarray, threshold: float) -> dict:
    """Timed scoring (module docstring), lag quantiles, false alarms by mass band, and
    bootstrap intervals over attempts."""
    below = np.nan_to_num(deb, nan=np.inf) < threshold
    fired = below.any(axis=1)
    first = np.argmax(below, axis=1)
    n_att = len(attempts)
    tp_a, fp_a, fn_a = np.zeros(n_att), np.zeros(n_att), np.zeros(n_att)
    lags = []
    by_variant: dict[str, list[float]] = {}
    band_fp = np.zeros(len(MASS_BANDS))
    band_n = np.zeros(len(MASS_BANDS))
    for k, row in enumerate(rows):
        i = row[0]
        a = attempts[i]
        gt, gt_t = a.meta["gt_slip"], a.meta["gt_slip_time"]
        fire_t = a.t[first[k]] if fired[k] else None
        early = fire_t is not None and gt and fire_t < gt_t - EARLY_S
        if gt and fire_t is not None and not early:
            tp_a[i] += 1
            lags.append(fire_t - gt_t)
            variant = "{} {:g}".format(*VARIANTS[a.meta["variant"]])
            by_variant.setdefault(variant, []).append(1000 * (fire_t - gt_t))
        else:
            if fire_t is not None:
                fp_a[i] += 1
            if gt:
                fn_a[i] += 1
        if not gt:
            band = next(j for j, (lo, hi) in enumerate(MASS_BANDS) if lo <= a.meta["mass"] < hi)
            band_n[band] += 1
            band_fp[band] += fire_t is not None
    tp, fp, fn = tp_a.sum(), fp_a.sum(), fn_a.sum()
    rng = np.random.default_rng(12345)
    pick = rng.integers(0, n_att, size=(BOOTSTRAP, n_att))
    weights = np.apply_along_axis(np.bincount, 1, pick, minlength=n_att)
    btp, bfp, bfn = weights @ tp_a, weights @ fp_a, weights @ fn_a
    with np.errstate(invalid="ignore", divide="ignore"):
        bp, br = btp / (btp + bfp), btp / (btp + bfn)
    lags_ms = 1000 * np.array(lags) if lags else np.array([math.nan])
    bands = {}
    for j, (lo, hi) in enumerate(MASS_BANDS):
        label = f"{1000 * lo:g}-{1000 * hi:g} g" if hi < 9 else f">={1000 * lo:g} g"
        bands[label] = [int(band_fp[j]), int(band_n[j])]
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(tp / (tp + fp)) if tp + fp else math.nan,
        "recall": float(tp / (tp + fn)) if tp + fn else math.nan,
        "precision_ci": [float(np.nanpercentile(bp, 2.5)), float(np.nanpercentile(bp, 97.5))],
        "recall_ci": [float(np.nanpercentile(br, 2.5)), float(np.nanpercentile(br, 97.5))],
        "lag_p50_ms": float(np.median(lags_ms)),
        "lag_p90_ms": float(np.percentile(lags_ms, 90)),
        "lag_max_ms": float(np.max(lags_ms)),
        "late_share": float(np.mean(lags_ms > 0)) if lags else math.nan,
        "fp_by_mass": bands,
        "lag_by_variant": {
            v: {
                "n": len(x),
                "late_share": float(np.mean(np.array(x) > 0)),
                "p90_ms": float(np.percentile(x, 90)),
                "max_ms": float(np.max(x)),
            }
            for v, x in sorted(by_variant.items())
        },
    }


def long_hold_alarms(noise_k: float, kg: float, hold_s: float, seeds: int) -> int:
    """Rule (b): how many of ``seeds`` synthetic holds of a ``kg`` cube false-alarm under
    REALISTIC_NOISE (0.3 s empty-handed before the watch, as in an attempt)."""
    sf, sa = NOISE["realistic"]
    hand = 0.76
    config = replace(SlipConfig(), noise_k=noise_k)
    alarms = 0
    for seed in range(seeds):
        det = SlipDetector(DT, G, config, watching=False, noise=(sf, sa))
        rng = np.random.default_rng((77, seed))
        det.tare(hand * G, G)
        for k in range(round(0.3 / DT)):
            f_z = hand * G + sf * rng.standard_normal()
            det.update(k * DT, f_z, G + sa * rng.standard_normal())
        det.watch()
        for k in range(round(hold_s / DT)):
            f_z = (hand + kg) * G + sf * rng.standard_normal()
            det.update(0.3 + k * DT, f_z, G + sa * rng.standard_normal())
        alarms += det.fired_at is not None
    return alarms


def _fmt(sc: dict) -> str:
    p_lo, p_hi = sc["precision_ci"]
    r_lo, r_hi = sc["recall_ci"]
    return (
        f"{sc['tp']} | {sc['fp']} | {sc['fn']} | {sc['precision']:.3f} ({p_lo:.2f}-{p_hi:.2f}) "
        f"| {sc['recall']:.3f} ({r_lo:.2f}-{r_hi:.2f}) | {sc['lag_p50_ms']:.0f} / "
        f"{sc['lag_p90_ms']:.0f} | {100 * sc['late_share']:.0f} %"
    )


HEAD = (
    "| TP | FP | FN | precision (95 % CI) | recall (95 % CI) | lag p50 / p90 [ms] "
    "| detections after the 5 mm line |"
)


def _bands(sc: dict) -> str:
    return ", ".join(f"{b}: {v[0]}/{v[1]}" for b, v in sc["fp_by_mass"].items())


def aggregate(out: Path) -> int:
    lines: list[str] = []
    metrics: dict = {}
    w05 = round(T05_SLIP.debounce_s / DT)
    th = T05_SLIP.drop_fraction

    def add(s: str = "") -> None:
        lines.append(s)
        print(s)

    # 1. The replay reproduces T05's held-out counts (T05 semantics, no noise).
    t05 = load_attempts(out, "t05")
    rows = noisy_inputs(t05, "none", 1, "t05")
    deb = replay(rows, (0.0, 0.0), (0.0,), (w05,), semantics="t05")[(0.0, w05)]
    sc = score(t05, rows, deb, th)
    metrics["t05_check"] = sc
    add(f"## Replay check: T05 held-out cubes, T05 detector, no noise ({len(t05)} attempts)")
    add("")
    add(f"TP {sc['tp']}, FP {sc['fp']}, FN {sc['fn']} (FEASIBILITY §5: 34, 3, 0)")
    add("")

    # 2. noise_k on the tune set.
    tune = load_attempts(out, "tune")
    rows = noisy_inputs(tune, "realistic", SEEDS, "tune")
    debs = replay(rows, NOISE["realistic"], NOISE_KS, (w05,))
    table = {k: score(tune, rows, debs[(k, w05)], th) for k in NOISE_KS}
    kg, hold_s, seeds, allowed = LONG_HOLD
    long_hold = {k: long_hold_alarms(k, kg, hold_s, seeds) for k in NOISE_KS}
    ok = [
        k
        for k in NOISE_KS
        if table[k]["precision_ci"][0] >= BAR[0]
        and table[k]["recall_ci"][0] >= BAR[1]
        and long_hold[k] <= allowed
    ]
    chosen = min(ok) if ok else None
    n_slips = sum(a.meta["gt_slip"] for a in tune)
    add(
        f"## noise_k on the tune set under REALISTIC_NOISE ({len(tune)} attempts, "
        f"{n_slips} slips, {SEEDS} seeds)"
    )
    add("")
    add(
        "| noise_k " + HEAD + " false alarms by mass (fired / held) | "
        f"{hold_s:g} s hold of {1000 * kg:g} g: alarms / {seeds} |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|")
    for k in NOISE_KS:
        add(f"| {k:g} | {_fmt(table[k])} | {_bands(table[k])} | {long_hold[k]} |")
    add("")
    add(
        f"Chosen noise_k (smallest whose 95 % lower bounds reach precision 0.9 and "
        f"recall 0.8, with at most {allowed} long-hold alarms): {chosen}"
    )
    add("")
    metrics["tune"] = {str(k): v for k, v in table.items()}
    metrics["long_hold_alarms"] = {str(k): v for k, v in long_hold.items()}
    metrics["chosen_noise_k"] = chosen
    if chosen is None:
        (out / "summary.md").write_text("\n".join(lines) + "\n")
        return 1

    # 3. The holdout set, once, with the chosen noise_k (and T05's detector).
    hold = load_attempts(out, "holdout")
    n_slips = sum(a.meta["gt_slip"] for a in hold)
    add(f"## Holdout set ({len(hold)} attempts, {n_slips} slips, {SEEDS} seeds per noisy level)")
    add("")
    add("| detector | noise " + HEAD + " false alarms by mass (fired / held) |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    # The rule's choice, the next margin up (fewer false alarms on light cubes, later
    # detections), and T05's detector; the choice above read tune only.
    shown = [(f"noise_k {chosen:g}", chosen), ("T05", 0.0)]
    bigger = [k for k in NOISE_KS if k > chosen]
    if bigger:
        shown.insert(1, (f"noise_k {bigger[0]:g}", bigger[0]))
    for name, k in shown:
        for level in NOISE:
            rows = noisy_inputs(hold, level, SEEDS, "holdout")
            deb = replay(rows, NOISE[level], (k,), (w05,))[(k, w05)]
            sc = score(hold, rows, deb, th)
            metrics.setdefault("holdout", {}).setdefault(name, {})[level] = sc
            add(f"| {name} | {level} | {_fmt(sc)} | {_bands(sc)} |")
    add("")
    add("Holdout lag by attempt type, detections only: share after the 5 mm line, p90, max [ms]")
    add("")
    add("| detector | noise | " + " | ".join(f"{v} {f:g}" for v, f in VARIANTS) + " |")
    add("|---|---|" + "---|" * len(VARIANTS))
    for name in metrics["holdout"]:
        for level in ("none", "realistic"):
            by = metrics["holdout"][name][level]["lag_by_variant"]
            cells = []
            for v, f in VARIANTS:
                x = by.get(f"{v} {f:g}")
                cells.append(
                    "-"
                    if x is None
                    else f"{100 * x['late_share']:.0f} %, {x['p90_ms']:+.0f}, {x['max_ms']:+.0f}"
                )
            add(f"| {name} | {level} | " + " | ".join(cells) + " |")
    add("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    return 0


def check(out: Path) -> int:
    """The vectorized replay equals ``SlipDetector`` (lowest statistic per window, exactly)."""
    attempts = load_attempts(out, "tune")[:40]
    bad = total = 0
    for level in ("none", "realistic"):
        rows = noisy_inputs(attempts, level, 2, "tune")
        for semantics in ("suite", "t05"):
            for k_noise in (0.0, 3.0, 6.0):
                debs = replay(rows, NOISE[level], (k_noise,), (1, 3, 5), semantics)
                config = replace(T05_SLIP, noise_k=k_noise)
                for k, row in enumerate(rows):
                    det = scalar_replay(row, config, NOISE[level], semantics)
                    for w in (1, 3, 5):
                        d = debs[(k_noise, w)][k]
                        vec = float(np.nanmin(d)) if np.isfinite(d).any() else None
                        total += 1
                        bad += vec != det.min_ratio_by_window[w]
    print(f"check: {bad} mismatches in {total} (row, window) minima")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4) - 2))
    parser.add_argument("--deadline", type=float, default=480.0, help="physics wall budget [s]")
    parser.add_argument("--aggregate", action="store_true", help="tables only (no physics)")
    parser.add_argument("--check", action="store_true", help="replay == SlipDetector check")
    parser.add_argument("--physics", action="store_true", help="physics only (no tables)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.check:
        return check(args.out)
    if args.physics:
        return 0 if run_physics(args.out, args.workers, args.deadline) else 2
    if not args.aggregate and not run_physics(args.out, args.workers, args.deadline):
        return 2
    return aggregate(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
