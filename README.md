# ruvector-vfa

**Force-aware, memory-augmented robot arm in MuJoCo.** A Franka Panda learns to pick up cubes it cannot tell apart by sight — heavy, slippery, or fragile — by remembering how past grasps *felt*.

Inspired by Eka Robotics' Vision-Force-Action idea ("force is the language of manipulation") and the [`ruvector-robotics`](https://github.com/ruvnet/ruvector/tree/main/crates/ruvector-robotics) cognitive crate (`MemorySystem`, `CognitiveCore`, `SkillLearning`).

> **Status: planned, not yet implemented.** The code is built task by task with the `/ship-next` loop. Progress lives in [`ROADMAP.md`](ROADMAP.md).

## What it will do

- **Simulate** a 7-DoF Panda on a table with a wrist force/torque sensor (MuJoCo + Gymnasium): reach, lift, pick-and-place.
- **Remember** every attempt as vectors built from force profiles, trajectories, context, and outcome (episodic + semantic memory, cosine search, reinforcement of what worked).
- **Think in two speeds**: System 1 runs the grasp skill and reflexes at control rate (100–500 Hz); System 2 queries memory at attempt start, after a gentle touch-and-heft probe, and after a failure. A safety supervisor can override both.
- **Retry until success**, narrowing the safe grip-force window with evidence from similar past slips and breaks.
- **Prove it** against fair baselines (fixed, best-fixed, random-retry with the same retry budget) on a benchmark frozen before the memory code exists, including a force ablation.

## Read next

| Doc | What it holds |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, learning-problem design, memory, dual-process loop, RuVector mapping |
| [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md) | T05 physics spike: grip force × closing speed × hidden class, squeeze calibration, F/T slip detector |
| [`ROADMAP.md`](ROADMAP.md) | 25 tasks, each one PR, with acceptance criteria and verify commands |
| [`docs/BRIEF.md`](docs/BRIEF.md) | The original project brief, verbatim |
| [`CLAUDE.md`](CLAUDE.md) | Rules for the coding agent (environment, probes, frozen benchmark) |

## Build it with shipyard

[shipyard](https://github.com/shaal/shipyard) loops the `/ship-next` skill: one quality-gated, squash-merged PR per task.

```bash
npx --allow-git=all github:shaal/shipyard init                    # one-time: installs /ship + /ship-next into ~/.claude
npx --allow-git=all github:shaal/shipyard 1 --tasks ROADMAP.md    # ship one task
npx --allow-git=all github:shaal/shipyard --tasks ROADMAP.md      # drain the whole roadmap
```

Inside Claude Code, `/ship-next ROADMAP.md` ships exactly one task.

Setup and demo instructions arrive with T01 and T16. You can already build the scene, hold the arm under gravity compensation, read the wrist F/T sensor, render a PNG, time the impedance controller on a scripted grasp-and-lift, draw cubes from the five hidden object classes in `configs/objects.toml` (`vfa.sim.objects`: `sample_object`, `apply_object`, and the `BreakMonitor` break rule), and run the T05 grip-force spike:

```bash
uv sync
uv run vfa scene-check --png runs/scene.png
uv run python -m vfa.control.throughput      # time the OSC + gripper control loop
uv run python scripts/spike_grip.py --quick  # grip force × closing speed × class (~30-60 s, 8 workers)
```

## License

MIT (see [`LICENSE`](LICENSE)). The vendored Franka Panda model keeps its Apache-2.0 license from MuJoCo Menagerie (see [`assets/menagerie/NOTICE.md`](assets/menagerie/NOTICE.md)).
