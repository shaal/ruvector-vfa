//! PyO3 binding over `ruvector-robotics`' cognitive types (ruvector-vfa T22).
//!
//! Exposes the crate's `EpisodicMemory` (`store`, `recall_similar`),
//! `SemanticMemory` (`store`, `find_similar`), `SkillLibrary`
//! (`learn_from_demonstration`, `improve_skill`), and `CognitiveCore` (its
//! perceive → think → act → learn transitions, mode, and state) to Python.
//! The crate's `recall_similar` returns episodes without their scores; the
//! binding's `recall_similar_scored` recomputes each returned episode's score
//! with the crate's own formula (a left-to-right dot product of the query and
//! the flattened percepts, truncated to the shorter length), so the order is
//! the crate's and the score is the one it ranked by.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ruvector_robotics::cognitive::cognitive_core::{
    ActionCommand, ActionType, CognitiveConfig, CognitiveMode, CognitiveState, Decision, Outcome,
    Percept,
};
use ruvector_robotics::cognitive::{
    CognitiveCore as RsCore, Demonstration, Episode, EpisodicMemory as RsEpisodic,
    SemanticMemory as RsSemantic, SkillLibrary as RsSkills,
};

/// The crate's dot product (private there): left to right, truncated to the shorter slice.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[pyclass(module = "ruvector_vfa_py")]
struct EpisodicMemory {
    inner: RsEpisodic,
}

#[pymethods]
impl EpisodicMemory {
    #[new]
    fn new() -> Self {
        Self { inner: RsEpisodic::new() }
    }

    /// Store an `Episode { percepts, actions, reward, timestamp }`.
    fn store(&mut self, percepts: Vec<Vec<f64>>, actions: Vec<String>, reward: f64, timestamp: i64) {
        self.inner.store(Episode { percepts, actions, reward, timestamp });
    }

    /// The crate's `recall_similar`: the `k` best episodes as (actions, reward, timestamp).
    fn recall_similar(&self, query: Vec<f64>, k: usize) -> Vec<(Vec<String>, f64, i64)> {
        self.inner
            .recall_similar(&query, k)
            .into_iter()
            .map(|ep| (ep.actions.clone(), ep.reward, ep.timestamp))
            .collect()
    }

    /// `recall_similar` with each episode's score: (actions, score), best first.
    fn recall_similar_scored(&self, query: Vec<f64>, k: usize) -> Vec<(Vec<String>, f64)> {
        self.inner
            .recall_similar(&query, k)
            .into_iter()
            .map(|ep| {
                let flat: Vec<f64> = ep.percepts.iter().flatten().copied().collect();
                (ep.actions.clone(), dot(&query, &flat))
            })
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pyclass(module = "ruvector_vfa_py")]
struct SemanticMemory {
    inner: RsSemantic,
}

#[pymethods]
impl SemanticMemory {
    #[new]
    fn new() -> Self {
        Self { inner: RsSemantic::new() }
    }

    fn store(&mut self, name: &str, embedding: Vec<f64>) {
        self.inner.store(name, embedding);
    }

    fn retrieve(&self, name: &str) -> Option<Vec<f64>> {
        self.inner.retrieve(name).cloned()
    }

    /// The crate's `find_similar`: the `k` best (name, score); its ties come out in
    /// `HashMap` order (not fixed).
    fn find_similar(&self, query: Vec<f64>, k: usize) -> Vec<(String, f64)> {
        self.inner
            .find_similar(&query, k)
            .into_iter()
            .map(|(name, score)| (name.to_string(), score))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pyclass(module = "ruvector_vfa_py")]
struct SkillLibrary {
    inner: RsSkills,
}

#[pymethods]
impl SkillLibrary {
    #[new]
    fn new() -> Self {
        Self { inner: RsSkills::new() }
    }

    /// Learn `name` from demonstrations given as (trajectory [[x, y, z], ...],
    /// timestamps, metadata); returns (name, trajectory, confidence, execution_count).
    fn learn_from_demonstration(
        &mut self,
        name: &str,
        demos: Vec<(Vec<[f64; 3]>, Vec<i64>, String)>,
    ) -> (String, Vec<[f64; 3]>, f64, u64) {
        let demos: Vec<Demonstration> = demos
            .into_iter()
            .map(|(trajectory, timestamps, metadata)| Demonstration { trajectory, timestamps, metadata })
            .collect();
        let s = self.inner.learn_from_demonstration(name, &demos);
        (s.name, s.trajectory, s.confidence, s.execution_count)
    }

    fn improve_skill(&mut self, name: &str, feedback: f64) {
        self.inner.improve_skill(name, feedback);
    }

    fn execute_skill(&mut self, name: &str) -> Option<Vec<[f64; 3]>> {
        self.inner.execute_skill(name)
    }

    /// (name, trajectory, confidence, execution_count) of `name`, or None.
    fn get(&self, name: &str) -> Option<(String, Vec<[f64; 3]>, f64, u64)> {
        self.inner
            .get(name)
            .map(|s| (s.name.clone(), s.trajectory.clone(), s.confidence, s.execution_count))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

fn mode_name(mode: CognitiveMode) -> &'static str {
    match mode {
        CognitiveMode::Reactive => "Reactive",
        CognitiveMode::Deliberative => "Deliberative",
        CognitiveMode::Emergency => "Emergency",
    }
}

fn state_name(state: CognitiveState) -> &'static str {
    match state {
        CognitiveState::Idle => "Idle",
        CognitiveState::Perceiving => "Perceiving",
        CognitiveState::Thinking => "Thinking",
        CognitiveState::Acting => "Acting",
        CognitiveState::Learning => "Learning",
    }
}

#[pyclass(module = "ruvector_vfa_py")]
struct CognitiveCore {
    inner: RsCore,
}

#[pymethods]
impl CognitiveCore {
    /// `mode`: "Reactive", "Deliberative", or "Emergency" (the crate fixes it in its config).
    #[new]
    #[pyo3(signature = (mode = "Reactive", attention_threshold = 0.5, learning_rate = 0.01, max_percepts = 100))]
    fn new(mode: &str, attention_threshold: f64, learning_rate: f64, max_percepts: usize) -> PyResult<Self> {
        let mode = match mode {
            "Reactive" => CognitiveMode::Reactive,
            "Deliberative" => CognitiveMode::Deliberative,
            "Emergency" => CognitiveMode::Emergency,
            other => return Err(PyValueError::new_err(format!("unknown mode {other:?}"))),
        };
        let config = CognitiveConfig { mode, attention_threshold, learning_rate, max_percepts };
        Ok(Self { inner: RsCore::new(config) })
    }

    /// Returns the state after the call.
    fn perceive(&mut self, source: String, data: Vec<f64>, confidence: f64, timestamp: i64) -> &'static str {
        state_name(self.inner.perceive(Percept { source, data, confidence, timestamp }))
    }

    /// The decision as (reasoning, utility, action priority, action confidence), or None.
    fn think(&mut self) -> Option<(String, f64, u8, f64)> {
        self.inner
            .think()
            .map(|d| (d.reasoning, d.utility, d.action.priority, d.action.confidence))
    }

    /// Act on a wait decision (the binding's only action); returns the action's priority.
    fn act_wait(&mut self, millis: u64, priority: u8, confidence: f64, utility: f64) -> u8 {
        let decision = Decision {
            action: ActionCommand { action: ActionType::Wait(millis), priority, confidence },
            reasoning: String::new(),
            utility,
        };
        self.inner.act(decision).priority
    }

    fn learn(&mut self, success: bool, reward: f64, description: String) {
        self.inner.learn(Outcome { success, reward, description });
    }

    #[getter]
    fn state(&self) -> &'static str {
        state_name(self.inner.state())
    }

    #[getter]
    fn mode(&self) -> &'static str {
        mode_name(self.inner.mode())
    }

    #[getter]
    fn cumulative_reward(&self) -> f64 {
        self.inner.cumulative_reward()
    }

    #[getter]
    fn percept_count(&self) -> usize {
        self.inner.percept_count()
    }
}

#[pymodule]
fn ruvector_vfa_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EpisodicMemory>()?;
    m.add_class::<SemanticMemory>()?;
    m.add_class::<SkillLibrary>()?;
    m.add_class::<CognitiveCore>()?;
    m.add("RUVECTOR_REV", "3e8429739fc0a3878b553a87bb91e7f66d8c0cbd")?;
    m.add("DEBUG_BUILD", cfg!(debug_assertions))?;
    Ok(())
}
