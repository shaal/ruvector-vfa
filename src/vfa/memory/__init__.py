"""RuVector-inspired memory: embeddings, backend, memory system, skill library.

Built across T11–T13 and T22. Refer to ``docs/ARCHITECTURE.md`` §5 and §10.

RuVector mapping: this package mirrors ``ruvector_robotics::cognitive::memory_system``
(``WorkingMemory``, ``EpisodicMemory``, ``SemanticMemory``) and
``ruvector_robotics::cognitive::skill_learning`` (``SkillLibrary``, ``Skill``,
``Demonstration``). Each implementation module names the exact Rust type it mirrors.
"""
