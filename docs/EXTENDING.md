# Extending the project

Four how-to sections: an object class, a skill parameter, a memory backend, and a
task. Each gives numbered steps with the files and names to touch, then the
commands that check the change. Design: `docs/ARCHITECTURE.md`. Measured numbers:
`docs/RESULTS.md`.

## Ground rules

- Set up with `uv sync` (Python 3.12, from `.python-version`). Run every command
  with `uv run`.
- After any change, run `uv run ruff check . && uv run ruff format --check .`,
  then `uv run pytest -m "not slow" -q`. CI also skips the `render` and `perf`
  markers. `uv run pytest -m slow -q` runs the physics sweeps.
- Keep every shell command under 9 minutes. Chunk longer runs.
- Frozen files: `configs/benchmark.toml`, `configs/objects.toml`, and
  `configs/best_fixed.toml` (hash in `configs/benchmark.sha256`), and
  `configs/benchmark_pick_place.toml` (hash in `configs/benchmark_pick_place.sha256`,
  which also covers `objects.toml`). Do not edit them unless the task says so
  (`CLAUDE.md`, "Frozen benchmark").
- `benchmark freeze` writes a hash and checks nothing. Run it only in a planned
  re-freeze.
- Run output goes under `runs/`, which git ignores. Fitted parameters go under
  `configs/`.
- A new module in `memory/`, `cognition/`, or `skills/` needs a `RuVector mapping:`
  paragraph in its docstring. Log with the prefixes `[S1]`, `[S2]`, `[SAFETY]`,
  `[MEM]`. Pass a seeded `np.random.Generator`. Never call global `np.random`.

## 1. Add an object class

Each lift situation draws its cube from `configs/objects.toml`.
`situation_class` (`src/vfa/experiments/benchmark.py`) permutes the class names,
so a new class changes the class of every situation index. Treat the change as a
new benchmark version, never as a silent edit. There are two routes.

- **Re-freeze.** This works today. Every number in `docs/RESULTS.md` then
  describes the old benchmark. Record the decision in the PR body and in
  `docs/RESULTS.md`, and rerun every number you cite.
- **New benchmark version.** This is code work, and it is not built. T19 gave
  pick-and-place its own config and hash, but it still reads the shared
  `objects.toml`. A second objects file needs a new frozen-file tuple, a hash
  path, a `freeze` branch, and a path argument at each `load_object_config()`
  call that has none. Grep `load_object_config(`. Examples: `_worker_env`,
  `_run_learner`, and `throughput` in `benchmark.py`, and `PandaVFAEnv` in `env.py`.

Re-freeze steps:

1. Add `[classes.<name>]` to `configs/objects.toml` with `size`, `mass`,
   `friction`, `softness` (a preset from `[softness]`), and the optional
   `break_force`. `load_object_config` (`src/vfa/sim/objects.py`) requires every
   key and rejects unknown ones. Mass must lie in `OBJECT_MASS_RANGE` (0.01 to
   100 kg). Use the shared size range (`test_visible_size_carries_no_class_information`).
2. Make the class matter (`docs/ARCHITECTURE.md` §4, "Learning problem design").
   A class with `break_force` breaks when the summed finger force exceeds it
   (`BreakMonitor.update`). Its grip window must conflict with another class's,
   so no single grip serves both. Add an assertion like
   `test_class_windows_exist_and_conflict`. Add the class's row to the
   ARCHITECTURE table in the same PR.
3. In `tests/test_objects.py`, update `CLASS_NAMES`, the table in
   `test_committed_config_matches_the_architecture_table`, and the name of
   `test_committed_config_has_the_six_classes_in_order`.
4. Set `BLOCK` in `benchmark.py` (6, one cube per class) to the new count.
   `learning()` and `run_all.py` use it for the per-block report.
5. Recalibrate with `uv run python -m vfa.experiments.benchmark calibrate --workers 8 --write`.
   It rewrites `configs/best_fixed.toml`. `docs/RESULTS.md` §3 is the precedent:
   adding `brittle` dropped the best fixed grip's success@1 from 0.800 to 0.675.
   Check the separation with `uv run python scripts/probe_classes.py --workers 10`.
   Its `brittle` check is class-specific, so write one for the new class.
6. Regenerate the calibration fixture with
   `uv run python scripts/make_fixtures.py --workers 8`. It writes
   `tests/fixtures/calib_episodes.npz`, which stores the `hidden_class` labels.
   Keep fixtures under 5 MB (`docs/RESULTS.md` §4 records 3.20 MB for six classes).
7. Refit the standardization with `uv run python -m vfa.memory.embeddings fit`.
   It writes `configs/embedding_stats.npz`, which `code_fingerprint()` hashes, so
   cached chunks re-run.
8. Re-freeze with `uv run python -m vfa.experiments.benchmark freeze`, then
   `... freeze --task pick_place`. The second is needed because the
   pick-and-place hash covers `objects.toml`. Update `SITUATIONS_DIGEST` in
   `tests/test_benchmark_frozen.py` from the rows that
   `test_the_situations_are_pinned` builds.
9. Rerun the chunks behind every cited number. Chunk keys include the config
   hash, so old chunks are not reused.

Checks:

- `uv run pytest -q tests/test_objects.py`
- `uv run pytest -q tests/test_benchmark_frozen.py tests/test_pick_place.py::test_the_pick_place_benchmark_matches_its_hash`

## 2. Add a skill parameter

θ is `SkillParams` in `src/vfa/skills/params.py`. Pick the kind first.

- **Place field** (the T19 pattern, recommended). The lift's vectors, draws, and
  embeddings stay unchanged, so lift results stay valid.
- **Lift field.** This changes the lift benchmark. `random_retry` draws
  `uniform(LOW, HIGH)`. The embedding `outcome` group grows (`DIMS["outcome"]`).
  `SkillParams.from_dict` requires every `FIELDS` key, so `configs/best_fixed.toml`
  stops loading until it is recalibrated. The fixture's `theta` column changes.
  Treat it as the new benchmark version of section 1.

Place field steps:

1. In `params.py`, append the name to `PLACE_FIELDS`, add entries to `BOUNDS` and
   `UNITS`, and add a dataclass field with a default after `release_height`.
   Keep the dataclass order equal to `ALL_FIELDS`, because `describe()` slices
   `fields(self)` by position. `PLACE_LOW` and `PLACE_HIGH` follow from
   `PLACE_FIELDS`. State the bounds and their reason in the module docstring.
2. In `src/vfa/skills/place.py`, read the field in `PickPlaceSkill._after_lift`.
   The `slip_ran` class attribute of `PickPlaceSkill` already includes every
   place field, so a slip escalation keeps them.
3. In `src/vfa/experiments/baselines.py`, `Baseline.theta` draws each
   `PLACE_FIELDS` value from its own stream (tag `0x504C4143`). Pick-and-place
   `random_retry` results change, so rerun `docs/RESULTS.md` §10.
4. `src/vfa/memory/embeddings.py` does not change. `_theta_vector` and
   `outcome_features` read `FIELDS` only.
5. System 2 (`POST_PROBE` in `src/vfa/cognition/system2.py`) sets `grip_force`
   and, when it centers on successes, the other listed fields. Place fields are
   not listed, so they pass through unchanged. To let System 2 choose the new
   one, add it to `POST_PROBE` and check `center()` and `_skill()`.
6. Tests. In `tests/test_pick_place.py`, add the name to
   `test_theta_place_fields_leave_the_lift_vector_alone` (it pins `ALL_FIELDS`)
   and to `test_random_retry_draws_the_place_fields_from_their_own_stream` (it
   checks each place range by name). Add its worst value to
   `test_the_slowest_corner_finishes_in_time`, which is not marked slow.
   `test_params_bounds_and_defaults` in `tests/test_skill.py` loops over
   `ALL_FIELDS` and needs no edit.

Checks:

- `uv run pytest -q tests/test_skill.py tests/test_pick_place.py -m "not slow"`
- `uv run pytest -q tests/test_pick_place.py -k slowest`

Lift field steps:

1. In `params.py`, insert the name into `FIELDS` (vector order), `BOUNDS`,
   `UNITS`, and the dataclass before the place fields.
2. In `src/vfa/skills/pick.py`, read the field in `GraspLiftSkill._run`. If it
   acts before `PROBE_DONE`, add it to the `ran=` tuple of the
   `_escalate(PROBE_DONE, ...)` call. If it acts before a slip escalation, add it
   to `GraspLiftSkill.slip_ran`.
3. Add it to `POST_PROBE` in `system2.py` if System 2 should choose it.
4. Redo section 1, steps 5 to 9: the fixture, `embeddings fit`, `calibrate --write`,
   and `freeze`.
5. Update the pinned tests. `tests/test_embeddings.py::test_dimensions_are_fixed_and_documented`
   pins `DIMS`, and `outcome` becomes 11. `test_theta_place_fields_leave_the_lift_vector_alone`
   pins `FIELDS` and the `(5,)` shape. Add the new field's bound to the slow
   `test_slowest_theta_corner_fits_the_time_limit` in `tests/test_skill.py`.

Checks:

- `uv run pytest -q tests/test_skill.py tests/test_embeddings.py -m "not slow"`
- `uv run pytest -q tests/test_skill.py -m slow -k slowest_theta_corner`

## 3. Add a memory backend

Memory uses only the `MemoryBackend` protocol (`src/vfa/memory/backend.py`) and a
`metric` attribute. `memory_system.py` calls `check`, `add`, `search`, `update`,
`vector`, `payload`, `ids`, `save`, `load`, `__contains__` (the `in` test), and
`len`. `RetrievalPolicy` (`src/vfa/cognition/system2.py`) raises unless
`metric == "cosine"`. `load_index` (`memory_system.py`) refuses a saved index
whose metric differs from the factory's.

1. Create `src/vfa/memory/<name>.py`. Implement every protocol member, including
   `metric`. `search(vector, k, filter=None)` returns `Hit` objects.
2. Match `NumpyBackend` (`backend.py`). `check` returns the unit-norm vector for
   cosine, and `add` calls it first. `add` raises `KeyError` on a duplicate id.
   Ties keep insertion order (a stable sort). Filters go through `plain_filter`
   and `matches`. Payloads are deep-copied and pass through `json_ready`. `save`
   uses `new_save_dir` and `atomic_write`.
3. Wire it in. `MemorySystem(working_size, backend_factory)` (`memory_system.py`)
   passes the factory to `EpisodicMemory`, which makes one index per name in
   `KEY_TYPES`, and to `SemanticMemory`. Pass the system as
   `RetrievalPolicy(memory=...)`. `_run_learner` (benchmark) and `_run` (demo)
   build a `RetrievalPolicy` with no memory, so a run with the new backend needs
   a new option there, as T22's `--backend ruvector` does (`RuVectorBackend` in
   `src/vfa/memory/ruvector_backend.py` is the worked example: a Rust ranking over
   `NumpyBackend`'s side table).
4. Record the design in `docs/ARCHITECTURE.md` §5.4 ("Backend seam"), and add the
   `RuVector mapping:` paragraph to the module docstring.
5. `CODE_FILES` in `benchmark.py` includes `memory/*.py`, so a new module re-runs
   cached chunks.

Checks:

- `uv run pytest -q tests/test_memory.py`. Parametrize
  `test_top_k_equals_brute_force` and `test_backend_save_load_gives_identical_results`
  over your class. Both build a `NumpyBackend` directly now.
- For a Rust or other fast backend, add a parity test: the same unit vectors give
  the same top-k ids, in the same order, on 1000 random queries (the T22
  acceptance rule).

## 4. Add a task

A task is an env task, a skill, and a benchmark config. The benchmark selects
pick-and-place with a boolean (`pick_place = task == "pick_place"`) in several
functions. A third task changes each of them, not only `TASKS`.

1. Env (`src/vfa/sim/env.py`). Add the name to `TASKS`, a `vfa/Panda<Name>-v0`
   entry to `ENV_IDS`, and a `TIME_LIMIT_S` entry. In `PandaVFAEnv.reset`, set
   the goal. In `_obs_spaces`, declare any new observation. In `_check_terminal`,
   write the success and failure rules. In `_reward`, write the reward.
   `register_envs` registers every `ENV_IDS` entry.
2. Skill (`src/vfa/skills/<name>.py`). Subclass `GraspLiftSkill` (`pick.py`) and
   set `task`. Override `validate` (call `super().validate(env)` and check the
   `EventConfig`, as `PickPlaceSkill.validate` does), `_after_lift` (the phases
   after the lift), and `label` (scoring only). Add a `make_env` like the one in
   `place.py`. `validate_env` checks `env.task`.
3. Benchmark (`src/vfa/experiments/benchmark.py`). Add the `<NAME>_CONFIG`,
   `<NAME>_HASH`, `<NAME>_FILES`, `<NAME>_OUT`, and `<NAME>_CONDITIONS` constants,
   as `PICK_PLACE_CONFIG`, `PICK_PLACE_HASH`, `PICK_PLACE_FILES`, `PICK_PLACE_OUT`,
   and `PICK_PLACE_CONDITIONS` do. Add the name to `TASKS`. Extend
   `load_benchmark_config` (it hands `[benchmark] task = "pick_place"` to
   `_pick_place_config`), `load_pick_place_config`, `make_env`, `make_skill`,
   `_worker_env`, `_config_hash`, `plan_condition`, `_out`, `aggregate`, and
   `main`. `freeze` needs a branch that writes the new hash. Give the task its own
   `configs/benchmark_<name>.toml` and `.sha256`. Do not reuse the lift's hash.
4. Demo (`src/vfa/experiments/demo.py`). `--task` takes `B.TASKS`. Add a default
   seed to `DEFAULT_SEEDS`, and a branch to `load_config`.
5. Memory keys, only if the task needs its own. Copy the `PlaceStats` and
   `PlaceEmbedder` pattern in `src/vfa/memory/embeddings.py`. Add a fit subcommand
   to `main` (only `fit` and `fit-place` exist). Add the stats file to
   `code_fingerprint()`, which already lists `PLACE_STATS_PATH`.
6. Tests. Copy `tests/test_pick_place.py`: the hash test, the
   jobs-apart-from-the-lift test, and the `--out` test. `test_ids_are_registered`
   in `tests/test_env.py` loops over `ENV_IDS`, so it covers the new id.

Checks:

- `uv run pytest -q tests/test_env.py tests/test_pick_place.py -m "not slow"`
- `uv run vfa demo --quick --task pick_place` runs the existing task. After
  step 4, `uv run vfa demo --quick --task <name>` runs yours.
