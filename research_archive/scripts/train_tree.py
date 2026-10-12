import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".tools/python"))

import numpy as np
import swanlab
from catboost import CatBoostClassifier
from scipy.special import softmax

from scripts.evaluate_experiments import metrics
from src.data import get_indices, get_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1600)
    args = parser.parse_args()
    folder = ROOT / "outputs/checkpoints" / args.name
    if (folder / "best.cbm").exists():
        raise RuntimeError("Refusing to overwrite a completed tree model")
    folder.mkdir(parents=True, exist_ok=True)
    records = get_records()
    train, valid = get_indices(records, args.fold)
    features = np.load(ROOT / "data/processed/tree_features.npy", mmap_mode="r")
    labels = np.array([record["label"] for record in records])
    assert set(labels[train]) == set(range(40)) and all(records[index]["split"] == "train" for index in train + valid)
    counts = np.bincount(labels[train], minlength=40).clip(1)
    weights = (counts.mean() / counts) ** 0.25
    parameters = {"iterations": args.iterations, "depth": 6, "learning_rate": 0.07, "l2_leaf_reg": 8,
                  "loss_function": "MultiClass", "eval_metric": "Accuracy:use_weights=false", "class_weights": (weights / weights.mean()).tolist(),
                  "random_seed": 42, "border_count": 64, "task_type": "GPU", "devices": "0", "gpu_ram_part": 0.2,
                  "thread_count": 6, "train_dir": str(folder / "catboost_info"), "verbose": 100}
    config = vars(args) | {"parameters": parameters, "train_clips": len(train), "valid_clips": len(valid),
                           "validation_users": sorted({records[index]["user"] for index in valid}),
                           "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"), "pretrained": False,
                           "feature_metadata": json.loads((ROOT / "data/metadata/tree_features.json").read_text()),
                           "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (folder / "split.json").write_text(json.dumps({"train": train, "valid": valid}) + "\n")
    swanlab.init(project="cuhkx-small-multimodal", name=args.name, public=False,
                 log_dir=str(ROOT / "outputs/swanlog"), config=config)
    model = CatBoostClassifier(**parameters)
    started = time.monotonic()
    options = {"eval_set": (features[valid], labels[valid]), "early_stopping_rounds": 120, "use_best_model": True} if valid else {}
    model.fit(features[train], labels[train], **options)
    np.testing.assert_array_equal(model.classes_, np.arange(40))
    model.save_model(str(folder / "best.tmp.cbm"))
    (folder / "best.tmp.cbm").replace(folder / "best.cbm")
    history = model.get_evals_result()
    (folder / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    report = {"epoch": int(model.tree_count_), "minutes": (time.monotonic() - started) / 60}
    if valid:
        logits = np.asarray(model.predict(features[valid], prediction_type="RawFormulaVal"), dtype=np.float32)
        report.update(metrics(softmax(logits, axis=1), labels[valid]))
        np.savez(folder / "validation.npz", logits=logits, labels=labels[valid], indices=np.array(valid))
    else:
        report["training_only"] = True
    (folder / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    swanlab.log({f"result/{key}": value for key, value in report.items()})
    swanlab.finish()
    print("DONE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
