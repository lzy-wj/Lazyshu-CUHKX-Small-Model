#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.r2plus1d_models import R2Plus1DClassifier as BaselineClassifier


def parse_wrapper_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--async-shift-tokens", type=int, required=True)
    parser.add_argument("--async-shift-seed", type=int, default=424242)
    args, remaining = parser.parse_known_args()
    if args.async_shift_tokens < 0:
        raise ValueError("Async shift magnitude must be nonnegative")
    sys.argv = [sys.argv[0], *remaining]
    return args


def load_baseline_module():
    path = REPOSITORY_ROOT / "scripts/train_r2plus1d.py"
    specification = importlib.util.spec_from_file_location("baseline_train_r2plus1d", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module, path


def shift_tokens(values, shifts):
    batch_size, steps = values.shape[:2]
    source_steps = torch.arange(steps, device=values.device).unsqueeze(0) - shifts.unsqueeze(1)
    source_steps = source_steps.clamp_(0, steps - 1)
    gather_index = source_steps.reshape(batch_size, steps, *([1] * (values.ndim - 2))).expand_as(values)
    return torch.gather(values, 1, gather_index)


def classifier_type(maximum_shift, augmentation_seed):
    class AsyncShiftClassifier(BaselineClassifier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if maximum_shift and self.mode not in ("multimodal", "shared_multimodal"):
                raise ValueError("Async shifts require a multimodal model")
            self._async_generators = {}
            if self.mode in ("multimodal", "shared_multimodal"):
                self.skeleton.register_forward_hook(self._shift_skeleton)
                self.imu.register_forward_hook(self._shift_imu)

        def _generator(self, values, stream_offset):
            key = (values.device.type, values.device.index, stream_offset)
            if key not in self._async_generators:
                generator = torch.Generator(device=values.device)
                generator.manual_seed(augmentation_seed + stream_offset)
                self._async_generators[key] = generator
            return self._async_generators[key]

        def _shift(self, values, stream_offset):
            if not self.training or maximum_shift == 0:
                return values
            shifts = torch.randint(
                -maximum_shift,
                maximum_shift + 1,
                (values.shape[0],),
                device=values.device,
                generator=self._generator(values, stream_offset),
            )
            return shift_tokens(values, shifts)

        def _shift_skeleton(self, module, inputs, output):
            return self._shift(output, 1009)

        def _shift_imu(self, module, inputs, output):
            return self._shift(output, 2017)

    return AsyncShiftClassifier


def main():
    wrapper_args = parse_wrapper_args()
    baseline, baseline_path = load_baseline_module()
    original_parse_args = baseline.parse_args

    def parse_args():
        args = original_parse_args()
        args.async_shift_tokens = wrapper_args.async_shift_tokens
        args.async_shift_seed = wrapper_args.async_shift_seed
        args.async_shift_policy = "video anchored; skeleton and IMU independently shifted with edge replication"
        args.async_shift_validation = "disabled by model.eval()"
        args.baseline_train_sha256 = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        args.baseline_model_sha256 = hashlib.sha256(
            (REPOSITORY_ROOT / "src/r2plus1d_models.py").read_bytes()
        ).hexdigest()
        args.baseline_data_sha256 = hashlib.sha256(
            (REPOSITORY_ROOT / "src/r2plus1d_data.py").read_bytes()
        ).hexdigest()
        return args

    baseline.parse_args = parse_args
    baseline.R2Plus1DClassifier = classifier_type(
        wrapper_args.async_shift_tokens,
        wrapper_args.async_shift_seed,
    )
    baseline.ROOT = EXPERIMENT_ROOT
    baseline.main()


if __name__ == "__main__":
    main()
