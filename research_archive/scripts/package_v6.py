#!/usr/bin/env python3
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "outputs/final_legal_multimodal_v6"


MODEL_FILES = {
    "models/r2_staged_multimodal_full_int8.pt": ROOT / "outputs/checkpoints/r2_staged_multimodal_full/best_int8.pt",
    "models/v3_final_x3d_int8.pt": ROOT / "outputs/checkpoints/quantized/v3_final_x3d_int8.pt",
    "models/v2_final_statistics_int8.pt": ROOT / "outputs/checkpoints/quantized/v2_final_statistics_int8.pt",
    "models/yolo11n.pt": ROOT / ".tools/checkpoints/yolo11n.pt",
    "models/person_detector.pth": ROOT / ".tools/torch/hub/checkpoints/ssdlite320_mobilenet_v3_large_coco-a79551df.pth",
}


SUPPORT_FILES = {
    "submission.csv": ROOT / "outputs/submissions/final_legal_multimodal_v6.csv",
    "predictions/final.npz": ROOT / "outputs/predictions/final_legal_multimodal_v6.npz",
    "predictions/final.json": ROOT / "outputs/predictions/final_legal_multimodal_v6.json",
    "predictions/base_int8.npz": ROOT / "outputs/predictions/deployable_base_int8.npz",
    "config.json": ROOT / "configs/v6_selected_ensemble.json",
    "RESULTS.md": ROOT / "docs/HIGH_CAPACITY_V6_20260907.md",
    "validation/blend.json": ROOT / "outputs/study_r2plus1d/deployable_all_int8_blend/report.json",
    "validation/repeat.json": ROOT / "outputs/study_r2plus1d/deployable_all_int8_repeat_cooccurrence.json",
    "validation/cache.json": ROOT / "data/processed/r2plus1d_yolo_16_128/manifest.json",
    "metadata/samples.json": ROOT / "data/metadata/samples.json",
}


SOURCE_FILES = [
    "scripts/predict.py",
    "scripts/predict_r2plus1d.py",
    "scripts/decode_repeat_submission.py",
    "scripts/evaluate_repeat_decoder.py",
    "scripts/prepare_r2plus1d_cache.py",
    "scripts/prepare_person_crops.py",
    "scripts/prepare_imu_world.py",
    "src/data.py",
    "src/models.py",
    "src/r2plus1d_data.py",
    "src/r2plus1d_models.py",
    "src/scratch_models.py",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DESTINATION.exists():
        raise FileExistsError(f"Refusing to overwrite existing package: {DESTINATION}")
    files = dict(MODEL_FILES) | dict(SUPPORT_FILES)
    files.update({f"source/{name}": ROOT / name for name in SOURCE_FILES})
    for relative, source in files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    model_bytes = sum((DESTINATION / relative).stat().st_size for relative in MODEL_FILES)
    if model_bytes >= 100_000_000:
        raise ValueError(f"Model assets exceed 100 MB: {model_bytes}")
    manifest = {
        "name": "final_legal_multimodal_v6",
        "model_asset_bytes": model_bytes,
        "model_limit_bytes": 100_000_000,
        "submission_sha256": digest(DESTINATION / "submission.csv"),
        "files": {
            str(path.relative_to(DESTINATION)): {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in sorted(DESTINATION.rglob("*")) if path.is_file()
        },
    }
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = DESTINATION.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(DESTINATION.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(DESTINATION.parent))
    report = {
        "directory": str(DESTINATION),
        "directory_bytes": sum(path.stat().st_size for path in DESTINATION.rglob("*") if path.is_file()),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": digest(archive),
        "model_asset_bytes": model_bytes,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
