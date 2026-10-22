# Results — ruvector-vfa

Measured numbers only, each from the session that wrote it (CLAUDE.md). T10 sets the yardstick: the frozen benchmark and the three non-adaptive baselines. T15, T17a, and T18 add their conditions on the same benchmark.

## 1. The frozen benchmark (T10)

`configs/benchmark.toml` + `configs/objects.toml` + `configs/best_fixed.toml`, SHA-256 `f3c0a3c89f8c51db9a69ab8d32b9592845b843174950659700b63a9c529c4b02` (`configs/benchmark.sha256`, written by `benchmark freeze`, checked by `tests/test_benchmark_frozen.py`).

| Setting | Value |
|---|---|
| Task | lift (`PandaVFAEnv`, osc at 500 Hz, the T08 skill, the T09 supervisor on) |
| Hidden classes | six (ARCHITECTURE §4); situations come in blocks of one cube per class, each block in a seeded order, so the index does not reveal the class |
| Situations | 60 per seed; the cube from `default_rng((seed, i))`, the same for every condition; policies see only an opaque situation key |
| Seeds | evaluation 1000, 1001, 1002; calibration 0, 1; quick (tuning) 500, 501; held-out 2000 (T18) |
| Retry budget | `max_attempts = 5` for every condition |
| Sensor noise | `REALISTIC_NOISE`; attempt `a` of situation `i` resets with the same seed in every condition (paired) |
| Profiles | `quick` 2 seeds × 60; `full` 3 seeds × 60; `calib` 2 seeds × 60; `heldout` 1 seed × 60 |

The `quick` profile uses two tuning seeds with 60 situations each (the user's decisions in T10, against the ROADMAP's "1 seed, 15 situations"): later tasks tune on it without touching the evaluation seeds, and a learning curve pools its halves over both seeds (60 situations per half; one seed of 15 would give halves of 7 cubes). The halves are not equally hard for a fixed θ: pooled first-attempt successes, first half / second half, are `naive` 38 / 42 (0.633 / 0.700; `random_retry` the same), `best_fixed` 43 / 38. A learning check on `quick` (T15, T16) must therefore beat `naive`'s own gain on the same situations, not only show a rise.

**Throughput** (`python -m vfa.experiments.benchmark throughput`, 60 naive attempts on one core, machine load 9.2): wall time per attempt median 0.919 s, p95 1.010 s, **71 attempts/min per core**, 5.70 s of sim time on average. The machine's load moves the wall time a lot. One condition on one seed is at most 300 attempts (about 4–5 min on one core), run in chunks of 15 situations; with `--workers 8` a whole condition on `full` (12 chunks) took 55–76 s and one on `quick` (8 chunks) 33–41 s, at load 9–13.

## 2. Baselines (T10, `full` profile)

`python -m vfa.experiments.benchmark --condition <c> --profile full --workers 8`, then `aggregate --profile full`. Mean ± sample std over the 3 evaluation seeds; rates per attempt unless stated; force metrics over successful attempts (`vfa/experiments/benchmark.py`).

| Condition | success@1 | success@5 | success@1, situations 21–60 | attempts-to-success | break rate | situations with a break | drop rate | peak-force excess [N] | force jerk [N/s] |
|---|---|---|---|---|---|---|---|---|---|
| `naive` (8 N, 0.6 m/s²) | 0.639 ± 0.010 | 0.644 ± 0.010 | 0.658 ± 0.014 | 2.78 ± 0.04 | 0.560 | 0.272 | 0.160 | 7.97 | 49.2 |
| `best_fixed` (5 N, 0.3 m/s²) | 0.650 ± 0.033 | 0.661 ± 0.025 | 0.658 ± 0.029 | 2.73 ± 0.15 | 0.210 | 0.100 | 0.463 | 5.09 | 38.3 |
| `random_retry` | 0.639 ± 0.010 | **0.828 ± 0.025** | 0.658 ± 0.014 | **2.21 ± 0.15** | 0.468 | 0.272 | 0.111 | 7.78 | 48.2 |

No safety stop occurred in any condition. Per class, success@1 / success@5 (mean over seeds):

| Class | `naive` | `best_fixed` | `random_retry` |
|---|---|---|---|
| light_rigid | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| heavy_rigid | 1.00 / 1.00 | 0.83 / 0.87 | 1.00 / 1.00 |
| slippery | 1.00 / 1.00 | 0.60 / 0.63 | 1.00 / 1.00 |
| fragile_soft | 0.00 / 0.00 | 0.40 / 0.40 | 0.00 / 0.43 |
| heavy_slippery | 0.47 / 0.50 | 0.07 / 0.07 | 0.47 / 0.87 |
| brittle | 0.37 / 0.37 | 1.00 / 1.00 | 0.37 / 0.67 |

- **`best_fixed` success@1 = 0.650 ≤ 0.75**: no grid θ solves the benchmark (0.675 on the calibration seeds, 0.650 on the evaluation seeds); it needs adaptation (acceptance met). For T18's first target (+15 points) the like-for-like bar compares the same window: `best_fixed` scores 0.658 over situations 21–60, so memory needs 0.808 there (the ROADMAP's wording also allows `best_fixed`'s overall 0.650, i.e. 0.800). The 40 late situations do not hold every class equally often (two classes have 6 cubes, four have 7, per seed).
- **`random_retry` success@5 = 0.828 and 2.21 attempts-to-success**: the bar for memory's attempts-to-success.
- A fixed θ almost never gains from retries (success@5 − success@1 ≤ 0.011): sensor noise rarely changes an outcome. Random retries reach the classes a fixed θ cannot, with breaks on the way.
- Both fixed θ break `fragile_soft` cubes; beyond that they fail different classes: 8 N (`naive`) breaks most `brittle` cubes, 5 N (`best_fixed`) drops most `heavy_slippery` and many `slippery` cubes. Both land near two thirds.
- Force jerk is over successful attempts; the realistic pad noise adds a floor of about 19 N/s to it in quadrature, so differences between conditions look smaller than they are.

## 3. Calibration and the class change (T10)

`python -m vfa.experiments.benchmark calibrate --workers 8 --write`: one attempt per calib situation (120) for each of 30 grid θ (grip 1–25 N × lift 0.3 or 1.0 m/s²; the other fields naive). The selection rule (highest success@1, ties to the lower grip, then the lower acceleration) was fixed before the first run.

**With the five T04 classes** (and the first situation order, class `i mod 5`) the best θ was 12 N at 0.3 m/s² with success@1 **0.800**: every class but `fragile_soft` at 1.00, as T05 predicted (FEASIBILITY §7). That is above the 0.75 bar, so T10 widened the class conflict in `configs/objects.toml`.

**The change:** a sixth class, `brittle` (stiff, 0.20–0.35 kg, μ 0.45–0.60, breaks at 12–20 N summed). It needs at most about 4 N per finger and breaks at 6 N or more, so the grip that holds the heavy classes breaks it and the `fragile_soft` grip lets it slide; T05's other lever, the closing speed, no longer breaks anything after T08's slow probe. `scripts/probe_classes.py` (benchmark noise; lift 0.6 m/s², the other θ fields naive): the T08 skill at 5 N lifted 20 of 20 `brittle` cubes, and at 12 N broke 10 of 10.

**With six classes** (the frozen benchmark) the best θ is **5 N at 0.3 m/s², success@1 0.675** on the calibration seeds (per class: brittle 1.00, light_rigid 1.00, heavy_rigid 0.95, slippery 0.65, fragile_soft 0.40, heavy_slippery 0.05). Every grid θ from 15 N up scores 0.667 (it breaks both breakable classes). Each class alone is solvable: its best grid θ scores 1.00 (fragile_soft and light_rigid at 1 N, brittle at 2.5 N, heavy_rigid at 8 N, heavy_slippery at 12 N, all at 0.3 m/s²; slippery at 10 N and 1.0 m/s²), so the gap to 1.00 comes from the conflict, not from impossible cubes. With 20 cubes per class the calibration is noisy, and the winner beat the 0.667 plateau of the high grips by one situation; the frozen rule and grid stand.

**Probe separability** (`scripts/probe_classes.py`: 20 cubes per class, the T08 probe, benchmark noise).

| Class | heft peak [N] | indentation [mm] | slides in the probe |
|---|---|---|---|
| light_rigid | 0.55–1.40 | 0.052–0.249 | 0 / 20 |
| heavy_rigid | 2.53–3.59 | 0.020–0.063 | 20 / 20 |
| slippery | 0.54–1.01 | 0.030–0.117 | 20 / 20 |
| fragile_soft | 0.55–1.82 | 0.333–1.399 | 0 / 20 |
| heavy_slippery | 0.87–1.34 | 0.021–0.067 | 20 / 20 |
| brittle | 1.53–2.05 | 0.030–0.102 | 20 / 20 |

`brittle`'s heft peak lies between `heavy_slippery`'s and `heavy_rigid`'s, the two classes whose grips conflict with it; `fragile_soft` stands apart by its indentation. `slippery` and `heavy_slippery` overlap, but a high grip serves both.

## 4. Calibration fixture (T10)

`uv run python scripts/make_fixtures.py --workers 8` → `tests/fixtures/calib_episodes.npz`: the `calib` profile with `naive` (attempts 1–2) and `random_retry` (attempts 2–3; its attempt 1 is naive's), observable signals at 50 Hz plus θ, probe features, context, event counts, outcome, an opaque situation group, and a `prev` link from each retry to the attempt before it; hidden labels (class, mass, friction, break force, situation index) in separate `hidden_*` arrays for test scoring only; episodes in a fixed shuffled order. 243 episodes (light_rigid 20, heavy_rigid 20, slippery 26, fragile_soft 77, heavy_slippery 49, brittle 51; 95 successes, 108 breaks, 31 drops, 9 slips), 3.20 MB.

## 5. Embeddings (T11)

`uv run pytest -q tests/test_embeddings.py -s` on `tests/fixtures/calib_episodes.npz`, one first attempt per cube (120 cubes; class labels from `hidden_class`, for scoring only): the mean cosine of same-class pairs minus that of cross-class pairs.

| Key | Dims | Gap |
|---|---|---|
| `key_probe` (context + probe) | 50 | **0.839** |
| `key_pre` (context only) | 12 | −0.006 |

The probe separates the classes (acceptance: gap ≥ 0.1); the observed context does not, as designed. A calibration attempt replayed live gives the fixture's keys (cosine > 0.9999), and the `key_probe` built from the recorder at `PROBE_DONE` equals the one built from the finished episode.
