import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, ClipDataset, get_indices, get_records, prepare_batch
from src.scratch_models import ScratchMultiModalModel


CASES = ["baseline", "drop_skeleton", "drop_imu", "drop_thermal", "shuffle_skeleton", "shuffle_imu", "shuffle_thermal", "hold_thermal", "zero_velocity"]


def perturb(batch, case):
    result = dict(batch)
    if case.startswith("drop_"):
        result[case.removeprefix("drop_") + "_mask"] = torch.zeros_like(batch["skeleton_mask"])
    elif case.startswith("shuffle_"):
        name = case.removeprefix("shuffle_")
        values = []
        for position, index in enumerate(batch["index"].cpu().tolist()):
            permutation = np.random.default_rng(index + 20260907).permutation(batch[name].shape[1])
            values.append(batch[name][position, torch.as_tensor(permutation, device=batch[name].device)])
        result[name] = torch.stack(values)
    elif case == "hold_thermal":
        values = batch["thermal"]
        result["thermal"] = values[:, values.shape[1] // 2:values.shape[1] // 2 + 1].expand_as(values).contiguous()
    elif case == "zero_velocity":
        result["skeleton"] = batch["skeleton"].clone()
        result["skeleton"][:, :, 102:] = 0
    return result


def main():
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    destination = ROOT / "outputs/model_analysis"
    destination.mkdir(parents=True, exist_ok=True)
    records = get_records()
    results = {}
    for fold in [0, 1]:
        folder = ROOT / f"outputs/checkpoints/fusion_input_v1_staged_f{fold}"
        state = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        config = state["config"]
        _, indices = get_indices(records, fold)
        assert json.loads((folder / "split.json").read_text())["valid"] == indices
        model = ScratchMultiModalModel(config["modalities"], fusion=config["fusion"], thermal_view=config["thermal_view"], imu_representation=config["imu_representation"])
        model.load_state_dict(state["model"], strict=True)
        model.cuda().eval()
        dataset = ClipDataset(indices, config["modalities"], frames=config["frames"], image_size=config["image_size"], canonical_pose=True,
                              thermal_view=config["thermal_view"], imu_representation=config["imu_representation"])
        loader = DataLoader(dataset, batch_size=24, num_workers=4, pin_memory=True)
        probabilities = {case: [] for case in CASES}
        with torch.inference_mode():
            for batch in loader:
                batch = prepare_batch(batch, torch.device("cuda"))
                for case in CASES:
                    altered = perturb(batch, case)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = model(altered)
                    probabilities[case].append(logits.float().softmax(1).cpu().numpy())
        labels = np.asarray([records[index]["label"] for index in indices])
        probability = {case: np.concatenate(values) for case, values in probabilities.items()}
        with np.load(folder / "validation.npz") as saved:
            delta = float(np.abs(probability["baseline"] - saved["probabilities"]).max())
            changed = int((probability["baseline"].argmax(1) != saved["probabilities"].argmax(1)).sum())
            assert delta < 0.02 and changed / len(labels) <= 0.01
        base = probability["baseline"].argmax(1) == labels
        fold_result = {"baseline_replay_max_probability_delta": delta, "baseline_replay_changed_predictions": changed, "cases": {}}
        for case, values in probability.items():
            correct = values.argmax(1) == labels
            fold_result["cases"][case] = {"accuracy": float(correct.mean()), "correct": int(correct.sum()), "clips": len(labels),
                                           "gain_percentage_points": float((correct.mean() - base.mean()) * 100),
                                           "corrected": int((correct & ~base).sum()), "regressed": int((~correct & base).sum())}
        results[str(fold)] = fold_result
        np.savez(destination / f"temporal_diagnosis_f{fold}.npz", **probability, labels=labels, indices=np.asarray(indices))
        print("Fold", fold, fold_result, flush=True)
        del model
        torch.cuda.empty_cache()
    pooled = {case: sum(result["cases"][case]["correct"] for result in results.values()) / sum(result["cases"][case]["clips"] for result in results.values()) for case in CASES}
    report = {"status": "complete", "folds": results, "pooled_accuracy": pooled,
              "scope": "Inference-only stress tests of the seed42 three-modality model. No retraining, no claim of causal contribution or guaranteed gains from architectural changes.",
              "shuffle_caveat": "All channels within a modality are permuted together, including existing derivative channels; this breaks order/alignment but retains per-frame pose-velocity associations."}
    (destination / "temporal_reliance.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Pooled", pooled, flush=True)


if __name__ == "__main__":
    main()
