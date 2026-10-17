"""Harmonized access to the MFE 5209 global multi-asset dynamic configuration study."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE = [
    ("config", "config.py"),
    ("data", "data_pipeline.py"),
    ("models", "models.py"),
    ("strategies", "strategies.py"),
    ("reporting", "reporting.py"),
    ("stress", "stress_tests.py"),
    ("benchmarks", "benchmarks.py"),
    ("robustness", "robustness.py"),
    ("attribution", "attribution.py"),
    ("visuals", "run_visual_diagnostics.py"),
]


def execute_stage(stage_file: Path, context: dict) -> None:
    """Implements a phase in the same shared context that maintains the original Notebook variable dependency."""
    source = stage_file.read_text(encoding="utf-8")
    context["__file__"] = str(stage_file)
    exec(compile(source, str(stage_file), "exec"), context)


def main(through: str = "visuals") -> dict:
    """Starts the configuration in sequence to the specified stage and returns the full study context."""
    stage_names = [name for name, _ in PIPELINE]
    if through not in stage_names:
        raise ValueError(f"Unknown stage:{through};optional values:{stage_names}")

    original_working_directory = Path.cwd()
    context = {
        "__name__": "__mfe5209_pipeline__",
        "PROJECT_ROOT": PROJECT_ROOT,
    }
    try:
        os.chdir(PROJECT_ROOT)
        final_index = stage_names.index(through)
        for position, (stage_name, filename) in enumerate(PIPELINE):
            if position > final_index:
                break
            print(f"\n [run phase]{stage_name}: {filename}")
            execute_stage(PROJECT_ROOT / filename, context)
    finally:
        os.chdir(original_working_directory)
    print(f"\n process is complete, termination phase:{through}")
    return context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Operation of the MFE 5209 global multi-asset dynamic configuration study")
    parser.add_argument(
        "--through",
        choices=[name for name, _ in PIPELINE],
        default="visuals",
        help="Runs from the beginning to the specified stage; default runs all stages.",
    )
    arguments = parser.parse_args()
    main(arguments.through)
