import copy
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, ClipDataset, get_records, prepare_batch
from src.scratch_models import ScratchMultiModalModel, validate_scratch_config
from src.validation_splits import validate_diagnostic_split


def main():
    torch.set_num_threads(6)
    torch.manual_seed(42)
    records = get_records()
    report = {"status": "running", "splits": {}, "models": {}}
    for fold in [0, 1]:
        split = json.loads((ROOT / f"configs/spatial_grouped_f{fold}.json").read_text())
        validate_diagnostic_split(split, records, fold)
        rejected = []
        for case in ["overlap", "metadata", "user_leakage", "outer_change"]:
            broken = copy.deepcopy(split)
            if case == "overlap":
                broken["train"].append(broken["valid"][0])
            elif case == "metadata":
                broken["metadata_sha256"] = "wrong"
            elif case == "outer_change":
                broken["outer_valid"].pop()
            else:
                index = broken["valid"].pop()
                broken["train"].append(index)
            try:
                validate_diagnostic_split(broken, records, fold)
            except ValueError:
                rejected.append(case)
            else:
                raise AssertionError(f"Invalid split accepted: {case}")
        report["splits"][str(fold)] = {"rejected": rejected}
    for kind in ["cnn_pooled", "cnn_tokens", "patch_tokens"]:
        feedforward = 1344 if kind == "patch_tokens" else 384
        options = {"modalities": ["skeleton", "imu", "thermal"], "fusion": "cross_transformer",
                   "visual_encoder": kind, "visual_feedforward": feedforward}
        validate_scratch_config(options)
        model = ScratchMultiModalModel(**options).cuda().eval()
        batch = {"skeleton": torch.randn(2, 64, 153, device="cuda"), "imu": torch.randn(2, 64, 85, device="cuda"),
                 "thermal": torch.randn(2, 16, 3, 160, 160, device="cuda")}
        for name in options["modalities"]:
            batch[name + "_mask"] = torch.ones(2, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = model.encoders["thermal"](batch["thermal"])
            expected = 64 if kind == "cnn_pooled" else 1600
            assert encoded.shape == (2, expected, 192)
            logits, _ = model(batch)
            assert logits.shape == (2, 40) and torch.isfinite(logits).all()
            for name in options["modalities"]:
                masked = dict(batch)
                masked[name + "_mask"] = torch.zeros(2, device="cuda")
                masked[name] = torch.full_like(batch[name], float("nan"))
                first, _ = model(masked)
                masked[name] = torch.zeros_like(batch[name])
                second, _ = model(masked)
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            missing = {name: torch.zeros_like(value) if name.endswith("_mask") else torch.full_like(value, float("nan"))
                       for name, value in batch.items()}
            output, _ = model(missing)
            torch.testing.assert_close(output, model.class_prior_logits[None].expand(2, -1))
            real_indices = json.loads((ROOT / "configs/spatial_grouped_f0.json").read_text())["train"][:2]
            dataset = ClipDataset(real_indices, options["modalities"], frames=16, image_size=160, canonical_pose=True,
                                  thermal_view="motion_crop", imu_representation="world")
            real_batch = prepare_batch(next(iter(torch.utils.data.DataLoader(dataset, batch_size=2))), torch.device("cuda"))
            real_output, _ = model(real_batch)
            assert torch.isfinite(real_output).all()
        model.train()
        large_batch = {name: value.repeat(12, *([1] * (value.ndim - 1))) for name, value in batch.items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        for iteration in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output, auxiliary = model(large_batch)
                loss = torch.nn.functional.cross_entropy(output, torch.arange(24, device="cuda") % 40)
                loss = loss + sum(branch.square().mean() for branch in auxiliary.values()) * 0.01
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
            visual_weight = model.encoders["thermal"].frames.weight if kind == "patch_tokens" else model.encoders["thermal"].frames[0].layers[0].weight
            assert visual_weight.grad.abs().sum() > 0
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        stream = io.BytesIO()
        torch.save({name: value.cpu() for name, value in model.state_dict().items()}, stream)
        assert stream.tell() < 100_000_000
        report["models"][kind] = {"parameters": sum(parameter.numel() for parameter in model.parameters()),
                                  "visual_tokens": expected, "state_bytes": stream.tell(), "batch_size": 24,
                                  "seconds_per_training_step_including_warmup": elapsed / 3,
                                  "peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
                                  "checks": ["shape", "real_data_forward", "finite_bf16_backward", "masked_nan_invariance", "all_missing_prior", "visual_gradient", "size"]}
        print(kind, report["models"][kind], flush=True)
        del model, optimizer, large_batch, batch, logits, output, loss, auxiliary, encoded, first, second, masked, missing
        torch.cuda.empty_cache()
    report["status"] = "passed"
    paths = ["src/spatial_tokens.py", "src/scratch_models.py", "src/data.py", "src/validation_splits.py",
             "scripts/train.py", "scripts/run_scratch_study.py", "scripts/run_training.sh", "scripts/audit_spatial_study.py",
             "configs/spatial_token_study.json", "configs/spatial_grouped_f0.json", "configs/spatial_grouped_f1.json",
             "data/metadata/samples.json"]
    report["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    destination = ROOT / "outputs/study_spatial"
    destination.mkdir(exist_ok=True, parents=True)
    (destination / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("AUDIT PASSED", flush=True)


if __name__ == "__main__":
    main()
