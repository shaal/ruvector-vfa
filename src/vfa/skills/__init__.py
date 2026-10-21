"""System 1 skills: parameterized grasp-and-lift, reflexes, residual policy.

Built across T08 and T21. Refer to ``docs/ARCHITECTURE.md`` §4.

RuVector mapping: skills are the REACTIVE (System 1) layer that
``ruvector_robotics::cognitive::cognitive_core::CognitiveMode::Reactive`` selects;
learned prototypes consolidate into ``skill_learning::Skill`` (see :mod:`vfa.memory`).
Each implementation module names the exact Rust type it mirrors.
"""
