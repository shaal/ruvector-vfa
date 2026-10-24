# RuVector mapping

This repository mirrors types of the Rust crate `ruvector-robotics` in Python: the memory tiers, the skill library, and the cognitive core. The bandit baseline mirrors `MetaThompsonEngine` and `BetaParams` from `ruvector-domain-expansion`. The default backend is pure Python and numpy (`NumpyBackend`, [ARCHITECTURE](ARCHITECTURE.md) §12 "Decision log", D9), and no Python file imports the Rust crate. The Rust backend (`RuVectorBackend`, a PyO3 binding) is planned in T22 and is not in `src/`.

Sources: [ROADMAP](../ROADMAP.md) (T12 to T14, T17a, T22); [ARCHITECTURE](ARCHITECTURE.md) §5 "Memory", §6 "Dual-process loop", §10 "Mapping to `ruvector-robotics`", §12 "Decision log"; and the mirror modules [memory_system.py](../src/vfa/memory/memory_system.py), [backend.py](../src/vfa/memory/backend.py), [skill_learning.py](../src/vfa/memory/skill_learning.py), [embeddings.py](../src/vfa/memory/embeddings.py), [cognitive_core.py](../src/vfa/cognition/cognitive_core.py), [system2.py](../src/vfa/cognition/system2.py), and [bandit.py](../src/vfa/experiments/bandit.py). Each mirror module has a `RuVector mapping:` paragraph in its docstring.

Rust paths below are relative to `crates/ruvector-robotics/src/cognitive/` in `github.com/ruvnet/ruvector` (local clone `~/code/utilities/ruvector`), unless marked. The bandit's Rust source is `crates/ruvector-domain-expansion/src/transfer.rs`. Python modules are under `src/vfa/`.

## 1. Type map

"—" means the crate has no counterpart. The notes say where our code adds to or differs from the crate.

### 1.1 Memory

| Python (`vfa.memory.memory_system`) | Rust (`memory_system.rs`) | Notes |
|---|---|---|
| `MemoryItem` | `MemoryItem` | Same fields: `key`, `data`, `importance`, `timestamp`, `access_count`. |
| `WorkingMemory(max_size)` | `WorkingMemory` | Evicts the least important item when full. Python needs `max_size >= 1`. |
| `EpisodeRecord` | `Episode` | Different shape. See 2.2. |
| `EpisodicMemory` | `EpisodicMemory` | One instance holds one index per key type (`KEY_TYPES`). Python adds `reinforce`, `save`, `load`. |
| `SemanticMemory` | `SemanticMemory` | Python `store(name, embedding, payload)` keeps a payload. The crate keeps only the embedding. |
| `Recall` | — | Fields `record`, `similarity`, `score`. The crate's `recall_similar` returns `Vec<&Episode>` with no score. |
| `MemorySystem` | — | Facade over the three tiers. The crate's `mod.rs` exports only the tier types. |
| `ranking_score()` | — | `similarity × (0.5 + utility)`. The crate ranks by dot product alone. |

| Python (`vfa.memory.backend`, `vfa.memory.embeddings`) | Rust | Notes |
|---|---|---|
| `MemoryBackend` (Protocol) | — | The adapter seam. See 3.1. |
| `NumpyBackend` | the store behind `EpisodicMemory` and `SemanticMemory` | Brute force. The crate scans a `Vec<Episode>` and a `HashMap<String, Vec<f64>>` by dot product. |
| `Hit` | — | Fields `id`, `score`, `payload`. |
| `EpisodeView`, `Embedder`, `PlaceEmbedder` | — | Build the retrieval keys (ARCHITECTURE §5.1 "Embeddings and retrieval keys"). In the T22 plan, a key is the one entry of `Episode.percepts` (2.2). |

### 1.2 Skills and demonstrations

| Python (`vfa.memory.skill_learning`) | Rust (`skill_learning.rs`) | Notes |
|---|---|---|
| `Demonstration` | `Demonstration` | Adds `theta`. Python `timestamps` is an `np.ndarray`; the crate's is `Vec<i64>`. |
| `Skill` | `Skill` | Same `name`, `trajectory`, `confidence`, `execution_count`. Adds `theta`, the mean over demonstrations. |
| `SkillLibrary` | `SkillLibrary` | `learn_from_demonstration`, `execute_skill`, `improve_skill`, `get`, `__len__` match. `consolidate` is ours. |
| `SkillParams` (`vfa.skills.params`) | — | θ for `GraspLiftSkill`. A frozen dataclass of floats, for example `grip_force`, `close_speed`, `lift_accel`, `grasp_depth`. See 2.3. |
| `GraspLiftSkill` (`vfa.skills.pick`), `PickPlaceSkill` (`vfa.skills.place`) | — | The REACTIVE executor (its docstring). The crate's `ActionCommand` and `ActionType` have no Python class. |
| `SlipReflex` (`vfa.skills.reflexes`) | — | A stimulus → response rule in the REACTIVE mode. |

### 1.3 Cognitive core

| Python (`vfa.cognition.cognitive_core`) | Rust (`cognitive_core.rs`) | Notes |
|---|---|---|
| `CognitiveCore` | `CognitiveCore` | `perceive`, `think(keep=None)`, `act`, `learn`, `mode`, `state` exist in both (see 2.5). `run_attempt` is ours. Settings are constructor arguments; the crate takes a `CognitiveConfig`. |
| `CognitiveMode` (`REACTIVE`, `DELIBERATIVE`, `EMERGENCY`) | `CognitiveMode` (`Reactive`, `Deliberative`, `Emergency`) | Ours follows the triggers. The crate's mode is fixed in its config. |
| `CognitiveState` (`IDLE`, `PERCEIVING`, `THINKING`, `ACTING`, `LEARNING`) | `CognitiveState` (`Idle`, `Perceiving`, `Thinking`, `Acting`, `Learning`) | Same five states. |
| `Percept` | `Percept` | Fields `source`, `data`, `confidence`, `timestamp`. Ours: `data` is a mapping and `timestamp` is a float in sim seconds. Crate: `Vec<f64>` and `i64`. |
| `Decision` | `Decision` | Ours: `theta`, `trigger`, `reasoning`, `utility`, `retrieved`, `evidence`. Crate: `action` (an `ActionCommand`), `reasoning`, `utility`. |
| `Outcome` | `Outcome` | Fields `success`, `reward`, `description`. `Outcome.from_result` gives reward +1 or −1. |
| — (`act` returns `SkillParams`) | `ActionCommand`, `ActionType` | No Python class. The names appear only in docstrings (`pick.py`, `place.py`, `cognitive_core.py`). |
| — (constructor arguments) | `CognitiveConfig` | Ours: `attention_threshold`, `learning_rate`, `max_percepts`. |
| `RetrievalPolicy` (`vfa.cognition.system2`) | — | System 2 (T15). Retrieval over the memory tiers replaces the crate's percept choice in `think`. |
| `DefaultPolicy`, `System2Policy` (Protocol) (`vfa.cognition.cognitive_core`) | — | The stub policy, and the interface `decide(percept, theta)`. |
| `ObservedAttempt`, `Handoff` (`vfa.cognition.cognitive_core`) | — | The failure percept's payload, and one line of `handoffs.jsonl`. |
| `Escalation`, `AttemptResult` (`vfa.skills.pick`) | — | `Escalation` hands control to System 2. `AttemptResult` yields the `Outcome`. |
| `solve_situation` (`vfa.cognition.retry`) | — | Runs the attempts at one cube and stores each one. |
| `SafetySupervisor` (`vfa.control.safety`) | `CognitiveMode::Emergency` (a mode value) | A separate class. A safety stop sets `EMERGENCY` until the next attempt starts. |

### 1.4 Bandit baseline

| Python (`vfa.experiments.bandit`) | Rust (`transfer.rs`, `ruvector-domain-expansion`) | Notes |
|---|---|---|
| `BetaParams` | `BetaParams` | `uniform`, `mean`, `variance`, `sample`, `update` in both. The crate's fields are `f32`. `sample` differs. See 2.6. |
| `ThompsonGripBandit` | `MetaThompsonEngine` | `select_arm` and `record_outcome`. Ours keeps `alpha` and `beta` arrays per (bucket, arm), and takes integer indices. The crate takes a `domain_id`, a `ContextBucket`, and an `ArmId`. |
| probe bins (`heft_edges`, `slope_edges`) | `ContextBucket` (`difficulty_tier`, `category`) | The bucket is a pair of probe bins, not a difficulty tier. |
| arm index | `ArmId(String)` | Arms are `SkillParams` values. |
| `BanditPolicy` | — | A System 2 policy (`decide`) over `ThompsonGripBandit`. |
| `Baseline` (`vfa.experiments.baselines`) | — | A `CognitiveCore` with no memory. Its `think` returns the same decision whatever was perceived. |

## 2. Semantic differences

### 2.1 Dot product, cosine, and normalization

- The crate's `recall_similar` and `find_similar` rank by a raw dot product. Its `dot_product` zips the two slices and truncates to the shorter one, with no error.
- Every stored vector is L2-normalized (ARCHITECTURE §12 "Decision log", D7). On unit vectors the dot product equals the cosine. `NumpyBackend(metric="cosine")` sums left to right, as the crate's `dot_product` does, so equal vectors get equal scores.
- Embeddings normalize last. Each group is standardized, clipped, divided by `sqrt(dims)`, and weighted. The whole key is then L2-normalized (`embeddings.py`).
- A `euclidean` metric exists in Python only. Its score is the negative distance, and vectors are stored as given.
- For the cosine metric, `check(vector)` returns the unit-norm vector. It raises on a wrong size, a non-finite value, or a zero vector. The crate's silent truncation makes this check necessary before any call into Rust.
- Similarities are not comparable across key types. A retrieval policy needs a threshold per key (`embeddings.py`).

### 2.2 Episode fields

The Rust `Episode` (`memory_system.rs`) holds `percepts: Vec<Vec<f64>>`, `actions: Vec<String>`, `reward: f64`, and `timestamp: i64`. It has no id, success flag, situation, θ, or utility.

Our `EpisodeRecord` (`memory_system.py`) holds `id`, `situation`, `attempt`, `theta`, `result`, `utility`, `timestamp`, `info`, and `theta_final`. `success` and `failure_mode` derive from `result`.

The planned mapping (ROADMAP T22; the `memory_system.py` docstring):

- The key vector goes into `percepts`, one entry per `Episode`. Each key type gets its own Rust `EpisodicMemory`, because the crate's `recall_similar` ranks every episode in one store.
- θ as JSON, and the record id, go into `actions`.
- `reward` holds the sign of the outcome: +1 for a success, −1 for a failure.
- `situation`, `attempt`, the failure label, `utility`, `info`, and `theta_final` go in a Python side table keyed by id.

`reward` holds one signed number, so the failure label stays in the side table. The crate's `recall_similar` returns episodes, not ids, so the id is read back from `actions`.

### 2.3 Side tables for θ

- The crate has no type for θ. The crate's `Skill` stores only a trajectory, a confidence, and an execution count, besides its name (`params.py` docstring).
- Our θ is kept in these places:
  - `EpisodeRecord.theta` (as chosen) and `EpisodeRecord.theta_final` (after escalations and reflexes).
  - `Demonstration.theta` and `Skill.theta` (the mean over the demonstrations).
  - The `SemanticMemory` payload (the mean θ and the member ids). The crate's `SemanticMemory::store` takes only a name and an embedding.
- Skills average θ as chosen (`record.theta`, not `theta_final`), because the reflexes add their raise at run time (`skill_learning.py` docstring).
- In the Rust backend, θ JSON sits beside each `Skill` and each `Episode` (ARCHITECTURE §10 "Mapping to `ruvector-robotics`").

### 2.4 Utility reinforcement and consolidation

The crate has no utility field, no `reinforce`, and no consolidation. These parts are ours.

- Utility starts at `INITIAL_UTILITY = 0.5` and stays in [0, 1]. `EpisodicMemory.reinforce(id, success, step=UTILITY_STEP)` moves it by `UTILITY_STEP = 0.1` toward 1 or toward 0 (`memory_system.py`).
- `RetrievalPolicy` reinforces the records a decision used (`Decision.retrieved`) once the attempt's outcome is known (`system2.py`).
- `SkillLibrary.consolidate(memory)` clusters the successful episodes on `key_probe` by cosine similarity. A leader pass joins a vector to a cluster at `CLUSTER_THRESHOLD = 0.8`. Then up to `CLUSTER_ITERATIONS = 10` spherical k-means passes run. Clusters with fewer than `MIN_SUPPORT = 3` members are dissolved, unless none reaches that size. Each cluster becomes `skill_00`, `skill_01`, and so on. Consolidation never reads class labels.
- Each consolidation starts over. It replaces every concept and every earlier consolidated skill, so `improve_skill` feedback lasts only until the next one.
- The confidence formula and the clamping match the crate. One difference: Python rejects non-finite feedback. The crate stores NaN for NaN and clamps ±inf (`skill_learning.py` docstring).

### 2.5 CognitiveCore deviations

These come from the `cognitive_core.py` docstring.

- The crate's `think()` picks the most confident percept (`max_by` on `confidence`). Ours asks the System 2 policy about the newest percept.
- The crate's mode is fixed in its config. Ours follows the triggers: `START`, `FAILURE`, `PROBE`, and `ESCALATION` run `DELIBERATIVE`. A safety stop sets `EMERGENCY` until the next attempt.
- Our `Decision` carries θ, its trigger, and the retrieved ids. The crate's `Decision` carries an `ActionCommand`. Our `Decision.utility` comes from the policy. The stub policy gives 0.
- Percepts hold observable data only. The `AttemptResult` holds the class, the hidden properties, and ground-truth columns, and it never reaches a policy.
- Shared rules: `perceive` drops a percept below the attention threshold and keeps at most `max_percepts`. `learn` adds `reward × learning_rate` to the cumulative reward. It moves the threshold by 0.01 (down on success, up on failure, within [0.1, 0.9]), clears the percepts, and returns to `IDLE`.

### 2.6 Bandit deviations

These come from the `bandit.py` docstring.

- **Sampler.** The crate's `BetaParams::sample` maps one uniform draw through a one-term normal approximation when α and β are both above 1. Otherwise it uses a power-law formula. The result is not a Beta draw. Ours uses numpy's `Generator.beta`. For Beta(1, 1), the crate's formula gives variance 0.128 against the exact 0.083, measured on a Python transcription of the crate's code (ARCHITECTURE §12 "Decision log", D19).
- **Ties.** The crate's `select_arm` uses a strict `>`, so the first tied arm wins. Ours picks a random tied arm.
- **Not ported.** Transfer priors (`init_domain_with_transfer`, `TransferPrior`), cost averages (`update_cost_ema`; the crate's `record_outcome` also takes a `cost`), the dual-path checks (`TransferVerification`), and `BetaParams::from_observations` and `merge`. We use one domain with uniform priors.

### 2.7 Other differences

- **Persistence.** The crate has none. Its `EpisodicMemory` and `SemanticMemory` derive `Debug, Clone, Default`, not `Serialize`. Ours is `MemorySystem.save` and `load`, which write versioned directories.
- **Filters.** The crate has none. Ours: `search` takes a mapping or a callable and filters before the top-k cut (`NumpyBackend.search`).
- **Ties.** The crate's `EpisodicMemory` builds its scores in insertion order and sorts with `sort_by`, which is stable. Its ties therefore keep insertion order, as `NumpyBackend` does with `argsort(..., kind="stable")`. The crate's `SemanticMemory` iterates a `HashMap`, so its ties come out in no fixed order.
- **Python only.** `reinforce`, payload `update`, and `consolidate`. The crate's `recall_similar` returns no scores, so the `similarity` and `score` fields of `Recall` and the `score` of `Hit` have no crate counterpart on the episodic side.

## 3. The drop-in (built in T22)

T22 built the plan below: `rust/ruvector_vfa_py` (a maturin + PyO3 crate over `ruvector-robotics` at `rev = "3e8429739fc0a3878b553a87bb91e7f66d8c0cbd"`), `vfa/memory/ruvector_backend.py` (`RuVectorBackend`), `--backend ruvector` on the benchmark and the demo, and `vfa/experiments/backend_speed.py`. The answers to the open questions are in 3.5; the measured parity and speed are in RESULTS §12.

### 3.1 The seam (exists now)

- `MemoryBackend` (`backend.py`, a `Protocol`) declares `add`, `check`, `search`, `update`, `vector`, `payload`, `ids`, `save`, `load`, `__contains__`, and `__len__`.
- `BackendFactory = Callable[[], MemoryBackend]` (`memory_system.py`) takes no arguments.
- `EpisodicMemory(backend_factory)` builds one index per entry of `KEY_TYPES` (`key_pre`, `key_probe`, `key_retry`, `full_embedding`). `SemanticMemory(backend_factory)` builds one index. `MemorySystem(working_size, backend_factory)` passes the factory to both.
- `load_index` checks the saved `metric` against the factory's. `MemorySystem.save` records the backend class name, and `load` refuses a save from another class.
- To switch backends: `RetrievalPolicy(memory=MemorySystem(backend_factory=...))`. The plan is that no cognition code changes.

### 3.2 What a Rust backend must provide

- The attribute `metric` (`"cosine"`), which `load_index` checks against the saved index; `MemorySystem.save` records the backend class name.
- `check(vector)`, run before any call into Rust.
- `add(id, vector, payload)`. The vector goes to the crate. The payload stays in Python, because the crate has no payload.
- `search(vector, k, filter)`. It returns `Hit`s in descending score order, with ties in insertion order. The score is the dot product. The crate's `recall_similar` returns no score, so the binding must return one or the backend must recompute it (open, see 3.5).
- `update`, `payload`, `ids`, `__contains__`, and `__len__` use a Python side table.
- `save` and `load`. The crate has no persistence, so Python writes the files.

### 3.3 Parity and tests (ROADMAP T22)

- For the same unit-norm vectors, the Rust backend and `NumpyBackend` return the same top-k ids in the same order, on 1 000 random queries.
- `vfa demo --quick --backend ruvector` runs, and its summary matches the numpy run with the same seed.
- The Rust tests skip when the extension is not built. The optional `ruvector` extra uses `maturin`.
- Speed: `vfa/experiments/backend_speed.py` times both backends at 1 000, 10 000, and 100 000 records, one command per size (RESULTS §12). Speed is not a gate.

### 3.4 What stays and what changes

- **Stays in Python:** embeddings and keys, `ranking_score` and utility, filters, payloads, `reinforce`, `consolidate`, the θ side tables, `RetrievalPolicy`, the retry loop, the bandit, and the benchmark.
- **Changes:** the index inside `EpisodicMemory` and `SemanticMemory`, which the factory supplies. ROADMAP T22 also lists `SkillLibrary` and `CognitiveCore` for the binding. The decision logic of `CognitiveCore` stays in Python. The crate's `think()` takes no arguments and picks the most confident percept, so it does not express our policy call.

### 3.5 How T22 answered the open questions

1. **Scores.** The binding's `recall_similar_scored` calls the crate's `recall_similar` and returns each episode with its score, recomputed by the crate's own formula (a left-to-right dot product of the query and the flattened percepts). The order is the crate's; the scores equal `NumpyBackend`'s bit for bit.
2. **Filter cost.** A filtered search asks the crate for its whole ranking and filters in Python; an unfiltered one asks for k.
3. **One factory, two index kinds.** `RuVectorBackend` keeps every index, the semantic tier's included, in a Rust `EpisodicMemory`: the key is the one percept, `actions` holds the record id and θ as JSON, `reward` the signed outcome. The binding also exposes `SemanticMemory`, but memory does not use it.
4. **Semantic ties.** Avoided by 3: the `EpisodicMemory` sort is stable, so ties keep insertion order as in `NumpyBackend`.
5. **`CognitiveCore` binding.** Exposed for its transitions (perceive → think → act → learn, mode, state) and tested; the decision logic stays in Python.
6. **Pin.** Confirmed with the GitHub API on 2026-10-23: `3e8429739fc0a3878b553a87bb91e7f66d8c0cbd` is still upstream's latest commit touching `crates/ruvector-robotics`.
