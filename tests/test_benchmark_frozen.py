"""T10: the benchmark is frozen (CLAUDE.md, Frozen benchmark).

``configs/benchmark.sha256`` holds one SHA-256 over ``configs/benchmark.toml``,
``configs/objects.toml``, and ``configs/best_fixed.toml``. A change to any of
them fails this test: improve the method, not the test.
"""

from __future__ import annotations

from vfa.experiments.benchmark import HASH_PATH, benchmark_hash


def test_the_benchmark_files_match_their_frozen_hash():
    frozen = HASH_PATH.read_text().split()[0]
    assert benchmark_hash() == frozen, (
        "configs/benchmark.toml, objects.toml, or best_fixed.toml changed after the T10 "
        "freeze (configs/benchmark.sha256)"
    )


# The code that turns the frozen files into situations is not in the hash; pin its
# output instead: the first 12 situations of seed 1000 (class, size, mass, opaque
# key) and their first-attempt env seeds.
SITUATIONS_DIGEST = "c701741902954ed7f94694e82bcae760a8a20122197984fdcb6e37466768ac7d"


def test_the_situations_are_pinned():
    import hashlib
    import json

    from vfa.experiments import benchmark as B
    from vfa.sim import objects as OB

    objects = OB.load_object_config()
    rows = [
        [
            s.class_name,
            round(s.spec.size, 9),
            round(s.spec.mass, 9),
            s.key,
            B.env_seed(1000, s.index, 1),
        ]
        for s in B.make_situations(1000, 12, objects)
    ]
    assert hashlib.sha256(json.dumps(rows).encode()).hexdigest() == SITUATIONS_DIGEST
