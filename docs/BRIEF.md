# Original brief (verbatim)

This is the project brief as the owner wrote it on 2026-10-20. It is the source of truth for "Requirements Fit" in every `/ship` confidence gate. Do not edit it. Record design decisions in `docs/ARCHITECTURE.md` instead.

Owner's framing: "create new public github repo under github.com/shaal, create a plan so I will use ship-next to fully implement this and experiment with it".

---

You are an expert robotics and systems engineer. Build a complete, working MuJoCo-based robotic arm simulation that demonstrates how RuVector can improve robot movement in a style inspired by Eka Robotics’ Vision-Force-Action (VFA) approach.Core Goal

Create a force-aware, memory-augmented control system where the robot arm:
Learns from past experiences (successful and failed force/trajectory profiles)
Uses retrieval to improve subsequent attempts
Has a dual-process (System 1 / System 2) control loop
Can retry until success using memory of what worked before
Recommended Stack
Physics / Simulation: MuJoCo (via mujoco Python package) + Gymnasium-style environment
Memory & Cognition: RuVector (specifically the ruvector-robotics crate concepts — MemorySystem, CognitiveCore dual-process, SkillLearning, and vector embeddings of force/trajectory data). Since RuVector is primarily Rust, implement a clean Python interface or mock the core memory/retrieval behavior first, with clear hooks so the real RuVector crates can be dropped in later.
Control: Combination of operational-space / impedance control + residual policy or skill retrieval
AI Layer: Simple RL or residual learning + retrieval-augmented decision making
What to Build
MuJoCo Environment
A 7-DoF Franka Panda (or similar) arm on a table with a graspable object (cube or small object).
Include force/torque sensing at the end-effector (or joint torques that can be converted).
Tasks: Reach, Lift, and a simple Pick-and-Place. Focus especially on contact-rich phases.
Clean Gymnasium-compatible interface (reset, step, render).
Logging of: end-effector pose, velocity, force/torque, contact events, success/failure, and full trajectories.

RuVector-Inspired Memory System
Store episodes as vectors that include:
Force/torque profiles (compressed or key moments)
Trajectory embeddings (end-effector path)
Outcome (success/failure + reward)
Context (object position, initial state)

Implement similarity search (cosine or Euclidean) over these embeddings.
Support episodic + semantic memory distinction.
Add a simple feedback/learning mechanism so successful patterns are reinforced (can be a basic scoring or weighted retrieval at first).

Dual-Process Control Loop
System 1 (Reactive): Fast, low-latency controller that executes a skill or residual policy.
System 2 (Deliberative): Queries memory, selects or adapts a skill, and handles recovery when System 1 fails.
Clear handoff between the two.
Emergency stop / safety override.

Skill Retrieval + Adaptation
On each attempt, query the memory for the most similar past experiences (especially successful force profiles).
Bias the controller toward those retrieved strategies.
After the episode, store the new experience (with force data) back into memory.

Retry-Until-Success Behavior
When a pick or lift fails, automatically try again using memory of previous failures and successes.
Demonstrate that later attempts improve because of the retrieved force/trajectory knowledge.

Deliverables
Complete, runnable Python codebase with clear structure.
README.md with setup instructions (pip install mujoco gymnasium ...), how to run the demo, and how the RuVector-inspired memory improves performance.
A simple demo script that shows:
Random or naive control (baseline)
Control with memory retrieval enabled
Visible improvement across retries

Logging and visualization of force profiles and retrieval hits.
Clear comments and architecture notes explaining how the real ruvector-robotics CognitiveCore / MemorySystem / SkillLearning would map onto this design.
Requirements.txt or pyproject.toml.
Constraints & Preferences
Prefer MuJoCo for the physics (better contact modeling than PyBullet).
Keep the initial version runnable on a normal machine (CPU is fine; GPU optional).
Make the memory/retrieval layer modular so it can later be replaced or augmented with the actual RuVector Rust crates.
Focus on clarity and demonstrable improvement from memory, not maximum performance.
Include force data in the memory embeddings — this is the most important parallel to Eka’s "force is the language of manipulation."
Success Criteria
The arm can complete a lift or pick task.
With memory enabled, subsequent retries or similar situations show better success rate or smoother force profiles than the baseline.
The dual-process loop is visible in the code and logs.
The project is well-documented and easy to extend.
Start by outlining the architecture, then generate the full code. Prioritize a working end-to-end demo over perfect sophistication.
