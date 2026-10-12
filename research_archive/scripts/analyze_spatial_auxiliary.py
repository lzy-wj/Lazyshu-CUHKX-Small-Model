import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, ClipDataset, get_records, prepare_batch
from src.scratch_models import ScratchMultiModalModel
from src.scratch_protocol import classification_metrics


def evaluate(model, loader):
    probabilities = {"fusion": [], "skeleton": [], "imu": [], "thermal": []}
    indices, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = prepare_batch(batch, torch.device("cuda"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output, auxiliary = model(batch)
            probabilities["fusion"].append(output.float().softmax(1).cpu().numpy())
            for name in auxiliary:
                probabilities[name].append(auxiliary[name].float().softmax(1).cpu().numpy())
            indices.extend(batch["index"].cpu().tolist())
            labels.extend(batch["label"].cpu().tolist())
    return np.asarray(indices), np.asarray(labels), {name: np.concatenate(values) for name, values in probabilities.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["cnn_pooled", "cnn_tokens", "patch_tokens"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    records = get_records()
    destination = ROOT / "outputs/study_spatial/auxiliary"
    destination.mkdir(parents=True, exist_ok=True)
    report = {}
    for seed in [42, 43]:
        for fold in [0, 1]:
            prefix = "spatial_v1" if seed == 42 else "spatial_v1_s43"
            folder = ROOT / f"outputs/checkpoints/{prefix}_{args.kind}_f{fold}"
            state = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
            config = state["config"]
            model = ScratchMultiModalModel(config["modalities"], fusion=config["fusion"], visual_encoder=config["visual_encoder"],
                                           visual_feedforward=config["visual_feedforward"], thermal_view=config["thermal_view"],
                                           imu_representation=config["imu_representation"]).cuda().eval()
            model.load_state_dict(state["model"], strict=True)
            split = json.loads((folder / "split.json").read_text())
            for partition in ["valid", "outer_valid"]:
                dataset = ClipDataset(split[partition], config["modalities"], frames=16, image_size=160, canonical_pose=True,
                                      thermal_view="motion_crop", imu_representation="world")
                loader = DataLoader(dataset, batch_size=24, num_workers=4, pin_memory=True)
                indices, labels, probabilities = evaluate(model, loader)
                np.testing.assert_array_equal(indices, split[partition])
                metrics = {name: classification_metrics(labels, values.argmax(1)) for name, values in probabilities.items()}
                report[f"seed{seed}_f{fold}_{partition}"] = metrics
                np.savez(destination / f"{args.kind}_s{seed}_f{fold}_{partition}.npz", indices=indices, labels=labels,
                         **{name + "_probabilities": values for name, values in probabilities.items()})
                print(seed, fold, partition, {name: round(value["accuracy"] * 100, 2) for name, value in metrics.items()}, flush=True)
            del model
            torch.cuda.empty_cache()
    (destination / f"{args.kind}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
