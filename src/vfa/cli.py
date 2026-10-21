"""Command-line entry point for ``vfa``.

Stub for T01: prints the version and the planned subcommand list. Later tasks
wire in the real subcommands (``scene-check`` in T02, ``benchmark`` in T10,
``demo`` in T16).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from vfa import __version__

# (name, one-line help, roadmap task that implements it). The stub lists them
# so ``vfa --help`` shows the shape of the finished tool.
_SUBCOMMANDS: tuple[tuple[str, str, str], ...] = (
    ("scene-check", "build the Panda scene, hold a pose, render a PNG", "T02"),
    ("demo", "baseline vs memory, visible improvement across retries", "T16"),
    ("benchmark", "run a benchmark condition (chunked, resumable)", "T10"),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with the subcommand stubs listed."""
    epilog_lines = ["planned subcommands (implemented task-by-task):"]
    epilog_lines += [f"  {name:<12} {help_} [{task}]" for name, help_, task in _SUBCOMMANDS]
    parser = argparse.ArgumentParser(
        prog="vfa",
        description="Force-aware, memory-augmented MuJoCo Franka Panda.",
        epilog="\n".join(epilog_lines),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ruvector-vfa {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. With no subcommand, print the version and subcommand list."""
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    del args  # only --version is defined so far
    if extra:
        parser.error(f"subcommand not implemented yet: {' '.join(extra)}")
    print(f"ruvector-vfa {__version__}")
    print("subcommands (not implemented yet):")
    for name, help_, task in _SUBCOMMANDS:
        print(f"  {name:<12} {help_} [{task}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
