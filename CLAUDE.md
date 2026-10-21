# CLAUDE.md — ruvector-vfa

Force-aware, memory-augmented MuJoCo robot arm (Franka Panda) inspired by
Eka Robotics' Vision-Force-Action (VFA) idea and the RuVector
`ruvector-robotics` cognitive crate. Built task-by-task by `/ship-next`.

## Sources of truth

| File | Role |
|---|---|
| `ROADMAP.md` | **The task list.** `/ship-next` takes the first unchecked task in document order. |
| `docs/BRIEF.md` | The original brief, verbatim. Score "Requirements Fit" against it plus the task text. |
| `docs/ARCHITECTURE.md` | Design decisions. Follow them. Change them only through a task that says so, and update the doc in the same PR. |
| `docs/RESULTS.md` | Measured numbers (created in T10). Never write a number here that you did not measure in this session. |

## Environment (read first — these traps are real on this machine)

```bash
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$HOME/.cargo/bin:$PATH"   # gh, uv, cargo live here
```

- The system `python3` is 3.15. MuJoCo has no wheels for it. **Always use uv with Python 3.12** (`.python-version` pins it).
- Setup: `uv sync` (creates `.venv`, installs the project in editable mode with dev extras).
- Run anything through uv: `uv run python ...`, `uv run pytest ...`.
- The shell is **zsh**: unquoted `$VAR` does not word-split. Use arrays or literal arguments in loops.
- `rm` and `cp` are aliased to interactive mode: they wait for a y/n answer and hang a headless command. Use `/bin/rm -f` and `/bin/cp -f`.
- `gh` is authenticated as `shaal`. The remote is `origin` → `github.com/shaal/ruvector-vfa`.
- GitHub rejects pushes whose commits carry a private email ("push declined due to email privacy restrictions"). This clone's local git config uses `22901+shaal@users.noreply.github.com`. In a fresh clone, run `git config user.email 22901+shaal@users.noreply.github.com` before you commit.

## Verification probes (use these in the /ship confidence gate)

The generic `/ship-next` text mentions `npm test` and browser smoke tests. **Those do not apply here.** Use:

| Probe | Command | Budget |
|---|---|---|
| Fast tests | `uv run pytest -m "not slow" -q` | < 60 s total |
| Lint/format | `uv run ruff check . && uv run ruff format --check .` | < 5 s |
| Slow tests (physics sweeps, benchmarks) | `uv run pytest -m slow -q` | < 10 min |
| Visual check | Render offscreen with `mujoco.Renderer` → save PNG under `runs/` → Read the PNG | < 30 s |
| Demo smoke (after T16) | `uv run vfa demo --quick` | < 3 min |
| Benchmark chunk (after T10) | `uv run python -m vfa.experiments.benchmark --condition <c> --profile <p> --workers 8` | ≤ 9 min per command |

Rules:

- **Never use `mujoco.viewer` / `launch_passive` / `launch` in agent runs.** On macOS it needs `mjpython` and blocks a headless session. Offscreen `mujoco.Renderer(model, h, w)` works (verified with mujoco 3.16).
- **No background jobs.** `/ship-next` runs headless; a background process dies when the turn ends.
- Mark any test over ~5 s with `@pytest.mark.slow`.
- Evidence in the gate must be command output you ran this session (test counts, metric values, PNG paths).

## Review budget (overrides `/ship-next` step 6 in this project)

The adversarial review stays, and the gate still needs an honest ≥ 95. The budget cuts rounds that only find doc and style issues, not rounds that find code bugs. (Evidence from T02–T05: rounds 4–7 of T03 and T05 still found real control bugs, and the late rounds of T02, T04 and T05 found mostly doc nits.)

- **Severity: judge it yourself with these definitions, not the reviewer's label.** *High*: changes a measured result, breaks an acceptance item, loses data, lets the agent see hidden class information (any leak counts, even a 0.1 mm one), or is a physics or control bug that changes outcomes. *Medium*: wrong behavior on an edge case the benchmark does not reach. *Low*: naming, wording, style, a nice-to-have test, or a wrong number in a doc.
- **Stop rule.** Run another round only if the previous round found a high or medium issue **in code**. Stop when a round finds only low or doc issues. Fix a low finding if it takes ≤ 10 min; otherwise list it in the PR body under "Known limitations".
- **Docs-vs-numbers check: once, at the end** (one reviewer on the final code), not in every round.
- **Ceiling: 6 rounds.** If round 6 still finds a high or medium code issue, take the `/ship-next` fail path so a human looks at the design.
- **Reviewers per round, by task risk.** Launch them in one message so they run in parallel.
  - **Full** (2–3 reviewers; the correctness and spec-fit lens on `opus`): T08, T09, T10, T11, T12, T13, T14, T15, T17a, T18, T21, T23, T24. Their code produces or changes the measured results.
  - **Light** (1 reviewer, correctness and spec fit, on `opus`): T07, T16, T17, T19, T20, T22. Plumbing, CLI, charts, docs, and a backend whose parity test guards the results.
  - A task not listed here gets the full review.
- **Docs record decisions and measured numbers, not review history.** Keep each design paragraph in `docs/ARCHITECTURE.md` under about 150 words. Put mechanism details in the module docstring. Do not write "review round N" notes in docs.

## Runtime limits (long runs must be chunked)

- **One shell command can run for at most 10 minutes** (the agent's foreground limit). Design every command to finish in **≤ 9 min**.
- macOS has **no `timeout` binary**. Enforce budgets inside Python (a wall-clock deadline that stops cleanly and saves partial results), not with a shell wrapper.
- Any run that can exceed 9 min must be **chunked and resumable**:
  - one result file per chunk: `runs/<run-id>/<condition>/seed<k>.json` (split further by situation range if needed);
  - the runner skips chunks whose result file already exists, so re-running the same command continues the work;
  - a separate `aggregate` subcommand builds the tables and plots from the chunk files.
- Parallelize independent chunks with `--workers N` (`concurrent.futures.ProcessPoolExecutor`; default `min(8, os.cpu_count() - 2)`; this Mac has 16 cores). Baseline conditions and different seeds are independent. A memory condition stays **sequential within one seed**, because its memory evolves.
- Throughput is a requirement, not a detail. T03 sets the floor, T10 measures attempts/minute and sizes the benchmark from it.

## Data handoffs between tasks

`runs/` is git-ignored and can disappear between tasks. A task must never depend on files another task left in `runs/`. Use one of these instead:

- **Committed fixtures**: small (≤ 5 MB total) files in `tests/fixtures/`, produced by `scripts/make_fixtures.py`, with the generating command in the script docstring.
- **Committed configs**: fitted parameters live in `configs/` (e.g. `configs/embedding_stats.npz`, `configs/best_fixed.toml`).
- **Regeneration**: a named command that rebuilds the input from scratch inside the task's time budget.

## Frozen benchmark (from T10 onward)

- `configs/benchmark.toml` defines the evaluation: object classes, hidden-property ranges, number of situations, seeds, retry budget.
- `configs/objects.toml` defines the hidden object classes (created in T04, final values set in T05/T10).
- `configs/best_fixed.toml` holds the `best_fixed` baseline θ found in T10. Later tasks load it; they never re-search it.
- `configs/benchmark.sha256` holds one hash over these three files. `tests/test_benchmark_frozen.py` fails if any of them changes.
- **Do not edit these files or the hash after T10** unless the task text explicitly says so. Improve the method, not the test.
- Baselines get the same retry budget as the memory agent. Never compare memory-with-retries against a single baseline attempt.

## Missed numeric targets

Some tasks state numeric targets (success rates, attempts-to-success). If a target is missed after honest iteration:

1. Do not tune the benchmark, cherry-pick seeds, or loosen the metric.
2. Write the measured numbers and a diagnosis (what you tried, the most likely cause) in `docs/RESULTS.md`.
3. Insert ONE follow-up task directly below the current task in `ROADMAP.md`, with a concrete hypothesis and the same target.
4. The current task may then ship if every non-numeric acceptance item is met. Say "target missed, follow-up Txx-b added" in the PR body.
5. If a follow-up task (`-b`) also misses its target, **fail the gate** and stop. A human decides.

## Sizing parameters on data (lesson from T06)

When a task picks parameters from measurements (thresholds, filters, margins), as T06 did for the slip detector (`docs/FEASIBILITY.md` §8):

- Choose on a tune set with a rule written down before any held-out set runs; score the held-out set once. A changed design or rule needs fresh held-out samples: record which sample ranges are spent.
- Report intervals (bootstrap over attempts, not over seeds that share physics), tails (p90, max, the share that comes too late), and rates by the subgroup at risk (e.g. cube mass), not only pooled medians.
- Check the choice beyond the harness: longer runs, other control rates, the event pipeline end to end.

## Code conventions

- Package: `src/vfa/` (import name `vfa`). Subpackages: `sim`, `control`, `skills`, `memory`, `cognition`, `experiments`, `viz`. Refer to `docs/ARCHITECTURE.md` for the layout.
- Python 3.11+ syntax, type hints everywhere, `@dataclass(slots=True)` for plain data, `numpy` for math.
- Randomness: pass an `np.random.Generator` (seeded) explicitly. No global `np.random.*` calls. Same seed → same result.
- Configuration: TOML (`tomllib`, stdlib). No YAML dependency.
- Heavy deps (torch, jax) are not allowed unless a task says so. The residual-learning task uses numpy only.
- Every module in `memory/`, `cognition/`, and `skills/` starts with a docstring that has a `RuVector mapping:` paragraph naming the `ruvector-robotics` type it mirrors.
- Run artifacts go in `runs/` (git-ignored). Small test fixtures go in `tests/fixtures/`.
- Logs: use the `logging` module with the channel prefixes `[S1]`, `[S2]`, `[SAFETY]`, `[MEM]` so the dual-process loop is visible.

## Commits and PRs

- Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`). Explain the *why*.
- **No AI-authorship trailers.** No `Co-Authored-By`, no "Generated with Claude Code".
- Stage specific files. Never `git add -A` / `git add .`.
- One ROADMAP task per PR. Tick its checkbox in the same commit.

## Reference: the real RuVector crate

A local clone exists at `~/code/utilities/ruvector/crates/ruvector-robotics` (upstream `github.com/ruvnet/ruvector`). Read `src/cognitive/{memory_system,cognitive_core,skill_learning}.rs` when you mirror an API. Key fact: its `EpisodicMemory::recall_similar` and `SemanticMemory::find_similar` use a **raw dot product**. We L2-normalize every stored vector so dot product equals cosine similarity, which keeps the Python and Rust backends interchangeable.
