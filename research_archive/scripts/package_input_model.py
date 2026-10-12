import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--destination", type=Path, default=ROOT / "outputs/final_scratch_input")
    parser.add_argument("--validation", type=Path, default=ROOT / "outputs/study_inputs/comparison.json")
    args = parser.parse_args()
    prediction_path = ROOT / "outputs/predictions" / f"{args.prediction}_manifest.json"
    prediction = json.loads(prediction_path.read_text())
    if prediction["rows"] != 405 or prediction["test_accuracy"] is not None:
        raise ValueError("Expected an audited unlabeled 405-clip prediction")
    if prediction["preprocessing_bytes"] != 0:
        raise ValueError("This bundle excludes pretrained preprocessing models")
    destination = args.destination
    if destination.exists():
        raise RuntimeError("Refusing to overwrite an existing final bundle")
    destination.mkdir(parents=True)
    models = []
    for position, model in enumerate(prediction["models"]):
        config = model["config"]
        if config["fold"] != -1 or not config["strict_scratch"] or config["pretrained"]:
            raise ValueError("Only full-data scratch models can enter the final bundle")
        source = Path(model["checkpoint"])
        if not source.is_absolute():
            source = ROOT / source
        if hashlib.sha256(source.read_bytes()).hexdigest() != model["sha256"]:
            raise ValueError("Prediction checkpoint changed")
        target = destination / f"model_{position}.pt"
        shutil.copyfile(source, target)
        models.append({"file": target.name, "sha256": model["sha256"], "bytes": target.stat().st_size,
                       "weight": prediction["weights"][position], "temperature": prediction["temperatures"][position],
                       "config": config, "epoch": model["epoch"]})
    source_files = ["src/scratch_models.py", "src/spatial_tokens.py", "src/data.py", "src/models.py", "src/__init__.py", "scripts/predict.py",
                    "scripts/prepare_data.py", "scripts/prepare_imu_world.py", "scripts/prepare_thermal_motion.py", "scripts/predict_input_raw.py"]
    for relative in source_files:
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    shutil.copyfile(ROOT / "outputs/submissions" / f"{args.prediction}.csv", destination / "submission.csv")
    shutil.copyfile(args.validation, destination / "validation_results.json")
    environment = {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "Pillow", "pandas", "tqdm"]}}
    (destination / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    checkpoint_arguments = " ".join(str(destination / model["file"]) for model in models)
    weights = " ".join(str(model["weight"]) for model in models)
    readme = f"""# Scratch multimodal input-research bundle

This bundle contains full-data model weights, the verified 405-row prediction CSV, validation results, and frozen inference/preprocessing source.

No external pretrained weights or learned detector are used. Staged initialization, if selected, uses only this project's supervised scratch encoders; their weights are already incorporated in the fused checkpoint and the source encoder checkpoints are not needed at inference.

Model selection uses two historically inspected user-isolated validation folds. It does not establish official test accuracy. No Kaggle submission has been made by this packaging script.

From the original repository, with the prepared dataset caches and Train conda environment:

```bash
bash scripts/run_prediction.sh <FREE_GPU> --checkpoints {checkpoint_arguments} --weights {weights} --name scratch_input_bundle_replay
```

For standalone inference directly from raw files, without the training metadata or any prepared cache:

```bash
python source/scripts/predict_input_raw.py --bundle . --test-csv /path/to/test.csv --raw-root /path/to/data/raw --output raw_submission.csv
```

Run the standalone command from this bundle directory with a free CUDA GPU selected using CUDA_VISIBLE_DEVICES. It needs Python, PyTorch, NumPy, Pillow, pandas and tqdm; exact tested versions are in environment.json. The raw-root must contain the small_model_track_test directory referenced by test.csv.

The raw dataset, generated input caches, training logs and optimizer state are not bundled. Thermal motion crop and world-coordinate IMU settings are stored in each checkpoint and must not be omitted.

The 100 MB check includes all files in this bundle, not just tensor parameters. Checkpoints remain FP32; BF16 is used for GPU inference/training, not claimed as the on-disk storage dtype.
"""
    (destination / "README.md").write_text(readme)
    files = {str(path.relative_to(destination)): {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in destination.rglob("*") if path.is_file()}
    manifest = {"models": models, "files": files, "prediction": args.prediction, "test_rows": 405,
                "official_test_accuracy": None, "external_pretrained_weights": False,
                "model_bytes": sum(model["bytes"] for model in models), "payload_bytes": sum(values["bytes"] for values in files.values()),
                "note": "All payload hashes exclude manifest.json itself; folder total is checked after manifest serialization"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    if total >= 100_000_000:
        raise RuntimeError(f"Final bundle exceeds 100 MB: {total}")
    print(json.dumps({"destination": str(destination), "model_bytes": manifest["model_bytes"], "bundle_bytes": total, "rows": 405}, indent=2))


if __name__ == "__main__":
    main()
