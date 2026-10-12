#!/usr/bin/env python3
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent


class ScopedRoot:
    def __truediv__(self, value):
        relative = Path(value)
        parts = relative.parts
        if parts[:2] == ("outputs", "checkpoints"):
            return EXPERIMENT / "checkpoints" / Path(*parts[2:])
        if parts[:2] == ("outputs", "swanlog"):
            return EXPERIMENT / "swanlog" / Path(*parts[2:])
        return ROOT / relative


def main():
    script = ROOT / "scripts/train_r2plus1d.py"
    spec = importlib.util.spec_from_file_location("ablation_dynamic_crop_train", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ScopedRoot()
    module.main()


if __name__ == "__main__":
    main()
