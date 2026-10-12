#!/usr/bin/env python3
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "outputs/final_legal_multimodal_v9"


MODEL_FILES = {
    "models/r2_staged_soup_full_w075_int8_compact.pt": ROOT / "outputs/checkpoints/r2_staged_soup_full_w075/best_int8_compact.pt",
    "models/v8_x3dl_soup_full_w075_int8_compact.pt": ROOT / "outputs/checkpoints/v8_x3dl_soup_full_w075/best_int8_compact.pt",
    "models/v3_final_x3d_int8_compact.pt": ROOT / "outputs/checkpoints/quantized/v3_final_x3d_int8_compact.pt",
    "models/v2_final_statistics_int8_compact.pt": ROOT / "outputs/checkpoints/quantized/v2_final_statistics_int8_compact.pt",
    "models/yolo11n_compact.pt": ROOT / ".tools/checkpoints/yolo11n_compact.pt",
    "models/person_detector_int8_compact.pth": ROOT / ".tools/torch/hub/checkpoints/ssdlite320_mobilenet_v3_large_coco_int8_compact.pth",
}


SUPPORT_FILES = {
    "submission.csv": ROOT / "outputs/submissions/final_legal_multimodal_v9.csv",
    "predictions/final.npz": ROOT / "outputs/predictions/final_legal_multimodal_v9.npz",
    "predictions/final.json": ROOT / "outputs/predictions/final_legal_multimodal_v9.json",
    "predictions/x3dl_crop_blend.npz": ROOT / "outputs/predictions/v9_x3dl_soup_full_cropblend50.npz",
    "predictions/r2_crop075.npz": ROOT / "outputs/predictions/r2_staged_soup_full_crop075.npz",
    "predictions/base_int8.npz": ROOT / "outputs/predictions/deployable_base_int8.npz",
    "config.json": ROOT / "configs/v9_selected_ensemble.json",
    "RESULTS.md": ROOT / "docs/HIGH_CAPACITY_V9_20260908.md",
    "validation/gated_blend.json": ROOT / "outputs/study_r2plus1d/v9_l_cropblend50_gated/report.json",
    "validation/gate_search.json": ROOT / "outputs/study_r2plus1d/v9_l_gate_decoder_fold0_search.json",
    "validation/cache.json": ROOT / "data/processed/r2plus1d_yolo_32_128/manifest.json",
    "metadata/person_boxes.npz": ROOT / "data/metadata/person_boxes.npz",
    "metadata/samples.json": ROOT / "data/metadata/samples.json",
}


SOURCE_FILES = [
    "scripts/blend_prediction_files.py",
    "scripts/compact_state_dict.py",
    "scripts/decode_repeat_submission.py",
    "scripts/evaluate_candidate_blends.py",
    "scripts/evaluate_gated_auxiliary.py",
    "scripts/evaluate_repeat_decoder.py",
    "scripts/evaluate_r2plus1d_checkpoint.py",
    "scripts/interpolate_r2plus1d_checkpoints.py",
    "scripts/predict.py",
    "scripts/predict_r2plus1d.py",
    "scripts/prepare_imu_world.py",
    "scripts/prepare_person_crops.py",
    "scripts/prepare_r2plus1d_cache.py",
    "scripts/quantize_checkpoint.py",
    "scripts/train.py",
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
        "name": "final_legal_multimodal_v9",
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
    print(json.dumps({
        "directory": str(DESTINATION),
        "directory_bytes": sum(path.stat().st_size for path in DESTINATION.rglob("*") if path.is_file()),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": digest(archive),
        "model_asset_bytes": model_bytes,
    }, indent=2))


if __name__ == "__main__":
    main()
