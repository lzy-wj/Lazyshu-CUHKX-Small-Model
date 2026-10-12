import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, ClipDataset, get_indices, get_records, prepare_batch
from src.models import MultiModalModel


def main():
    torch.set_num_threads(4)
    records = get_records()
    folder = ROOT / "outputs/checkpoints/v2_statistics_f0"
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = MultiModalModel(config["modalities"], temporal_backbone="statistics").eval()
    model.load_state_dict(checkpoint["model"])
    train, valid = get_indices(records, 0)
    report = {"model": "v2_statistics_f0", "clean_accuracy": {}, "oracle_existing_models": {}}
    for name, indices in [("train", train), ("valid", valid)]:
        dataset = ClipDataset(indices, config["modalities"], canonical_pose=True)
        correct = 0
        with torch.inference_mode():
            for batch in DataLoader(dataset, batch_size=96):
                batch = prepare_batch(batch, torch.device("cpu"))
                logits, _ = model(batch)
                correct += int((logits.argmax(1) == batch["label"]).sum())
        report["clean_accuracy"][name] = correct / len(indices)
    selected = json.loads((ROOT / "configs/v2_selected_ensemble.json").read_text())
    for fold in [0, 1]:
        arrays = [np.load(ROOT / "outputs/checkpoints" / name.replace("_f0", f"_f{fold}") / "validation.npz")
                  for name in selected["names"]]
        assert all(np.array_equal(array["indices"], arrays[0]["indices"]) for array in arrays)
        correct = np.stack([array["logits"].argmax(1) == array["labels"] for array in arrays])
        report["oracle_existing_models"][f"fold{fold}"] = {
            "at_least_one_correct": float(correct.any(0).mean()),
            "all_models_wrong": int((~correct.any(0)).sum()),
            "clips": len(arrays[0]["indices"]),
        }
    report["caveat"] = "Oracle uses validation labels to diagnose complementarity, is not deployable and is not an upper bound for new models or probability blending. CPU float32 clean evaluation can differ slightly from GPU bf16."
    (ROOT / "outputs/predictions/v3_generalization_diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
