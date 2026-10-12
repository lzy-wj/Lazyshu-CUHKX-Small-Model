import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train import evaluate
from src.data import ROOT, ClipDataset, get_indices, get_records
from src.scratch_models import ScratchMultiModalModel, validate_scratch_config
from src.scratch_protocol import ablation_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scratch_ablation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/study_encoders/diagnosis")
    args = parser.parse_args()
    torch.set_num_threads(6)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for matching the original BF16 evaluation")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    records = get_records()
    jobs = json.loads(args.config.read_text())["jobs"]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "running", "protocol": "Same best EMA checkpoint, no training augmentation or modality dropout on either split",
               "reproduction_tolerance": {"max_probability_delta": 0.02, "max_changed_fraction": 0.01,
                                          "note": "BF16 replay is not guaranteed bitwise identical; report replay scores and every changed count, not as model improvement"}, "models": {}}
    for job in jobs:
        folder = ROOT / "outputs/checkpoints" / job["name"]
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        validate_scratch_config(config)
        split = json.loads((folder / "split.json").read_text())
        permitted, expected_valid = get_indices(records, config["fold"])
        if config["fold"] < 0 or not set(split["train"]).issubset(permitted) or split["valid"] != expected_valid:
            raise ValueError("Diagnostic checkpoint does not match its user-isolated split")
        model = ScratchMultiModalModel(config["modalities"], fusion=config.get("fusion", "concat"),
                                      visual_multiplier=config.get("visual_multiplier", 1),
                                      skeleton_representation=config.get("skeleton_representation", "coordinates"),
                                      thermal_view=config.get("thermal_view", "full"), imu_representation=config.get("imu_representation", "raw"))
        model.load_state_dict(checkpoint["model"], strict=True)
        model.cuda().eval()
        target = args.output / job["name"]
        target.mkdir(exist_ok=True)
        result = {"checkpoint_epoch": checkpoint["epoch"],
                  "checkpoint_sha256": hashlib.sha256((folder / "best.pt").read_bytes()).hexdigest(),
                  "modalities": config["modalities"]}
        for name in ["train", "valid"]:
            dataset = ClipDataset(split[name], config["modalities"], training=False, frames=config["frames"],
                                  image_size=config["image_size"], canonical_pose=config.get("canonical_pose", False),
                                  thermal_view=config.get("thermal_view", "full"), imu_representation=config.get("imu_representation", "raw"))
            loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=4, pin_memory=True)
            metrics, predictions = evaluate(model, loader, torch.device("cuda"))
            result[name] = metrics | {"clips": len(dataset)}
            detail = ablation_metrics(predictions, records, config["modalities"])
            result[name]["available_accuracy"] = detail["available_only"]["accuracy"]
            result[name]["macro_f1"] = detail["all_validation"]["macro_f1"]
            np.savez(target / f"{name}.npz", **predictions)
            (target / f"{name}_metrics.json").write_text(json.dumps(detail, indent=2) + "\n")
            if name == "valid":
                with np.load(folder / "validation.npz") as previous:
                    np.testing.assert_array_equal(predictions["indices"], previous["indices"])
                    np.testing.assert_array_equal(predictions["labels"], previous["labels"])
                    delta = np.abs(predictions["probabilities"] - previous["probabilities"]).max(1)
                    changed = predictions["probabilities"].argmax(1) != previous["probabilities"].argmax(1)
                    ranked = np.sort(previous["probabilities"], axis=1)
                    margins = ranked[:, -1] - ranked[:, -2]
                    result["validation_reproduction"] = {
                        "max_probability_delta": float(delta.max()), "changed_predictions": int(changed.sum()),
                        "changed_indices": predictions["indices"][changed].tolist(),
                        "original_accuracy": float((previous["probabilities"].argmax(1) == previous["labels"]).mean()),
                        "changed_original_margins": margins[changed].tolist()}
                if delta.max() > 0.02 or changed.mean() > 0.01 or np.any(margins[changed] > 2 * delta[changed] + 1e-7):
                    raise RuntimeError("Replay differs beyond the declared numerical tolerance; inspect before interpreting diagnosis")
        result["accuracy_gap"] = result["train"]["accuracy"] - result["valid"]["accuracy"]
        summary["models"][job["name"]] = result
        (args.output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(job["name"], json.dumps(result), flush=True)
        del model
        torch.cuda.empty_cache()
    summary["status"] = "complete"
    (args.output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
