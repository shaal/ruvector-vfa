"""Command-line entry point for ``vfa``.

With no subcommand it prints the version and the subcommand list. Subcommands
are wired in task-by-task: ``scene-check`` (T02), ``benchmark`` (T10), ``demo``
(T16). A subcommand gets every argument after its name, so
``vfa scene-check --help`` shows the subcommand's own help.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from vfa import __version__

# (name, one-line help, roadmap task that implements it). All are listed
# so ``vfa --help`` shows the shape of the finished tool.
_SUBCOMMANDS: tuple[tuple[str, str, str], ...] = (
    ("scene-check", "build the Panda scene, hold a pose, render a PNG", "T02"),
    ("demo", "baseline vs memory, visible improvement across retries", "T16"),
    ("benchmark", "run a benchmark condition (chunked, resumable)", "T10"),
)


def _scene_check(argv: Sequence[str]) -> int:
    from vfa.sim import scene_check

    return scene_check.main(argv, prog="vfa scene-check")


def _benchmark(argv: Sequence[str]) -> int:
    from vfa.experiments import benchmark

    return benchmark.main(argv)


def _demo(argv: Sequence[str]) -> int:
    from vfa.experiments import demo

    return demo.main(argv, prog="vfa demo")


# Implemented subcommands: name -> handler(remaining argv) -> exit code.
_HANDLERS: dict[str, Callable[[Sequence[str]], int]] = {
    "scene-check": _scene_check,
    "demo": _demo,
    "benchmark": _benchmark,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with the subcommand stubs listed."""
    epilog_lines = ["subcommands (implemented task-by-task):"]
    epilog_lines += [f"  {_describe(name, help_, task)}" for name, help_, task in _SUBCOMMANDS]
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


def _describe(name: str, help_: str, task: str) -> str:
    status = "" if name in _HANDLERS else " (not implemented yet)"
    return f"{name:<12} {help_} [{task}]{status}"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Dispatch a subcommand, or print the version and subcommand list."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _HANDLERS:
        return _HANDLERS[argv[0]](argv[1:])
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    del args  # only --version is defined at the top level
    if extra:
        parser.error(f"subcommand not implemented yet: {' '.join(extra)}")
    print(f"ruvector-vfa {__version__}")
    print("subcommands:")
    for name, help_, task in _SUBCOMMANDS:
        print(f"  {_describe(name, help_, task)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
