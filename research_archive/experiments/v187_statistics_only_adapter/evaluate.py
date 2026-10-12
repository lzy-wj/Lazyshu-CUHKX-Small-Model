#!/usr/bin/env python3
from pathlib import Path

import experiments.v184_world_magnetic_main.evaluate as evaluation


HERE = Path(__file__).resolve().parent


def candidate_path(variant, fold):
    return HERE / "outputs/checkpoints" / f"v187_statistics_only_f{fold}" / "validation.npz"


if __name__ == "__main__":
    evaluation.HERE = HERE
    evaluation.VARIANTS = ("adapter",)
    evaluation.candidate_path = candidate_path
    evaluation.main()
