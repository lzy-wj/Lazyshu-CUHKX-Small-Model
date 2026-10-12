#!/usr/bin/env python3
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/final_legal_multimodal_v14"
MODEL_LIMIT = 100_000_000

MODEL_FILES = {
    "v14_main_full_hybrid_compact.pt": ROOT / "experiments/scene_context_v13/outputs/checkpoints/v14_main_full_hybrid_compact.pt",
    "v14_shared6_fixedsoup_uniform_compact.pt": ROOT / "experiments/scene_context_v13/outputs/checkpoints/v14_shared6_mixup_full_fixedsoup_uniform_l3int6_l4b0.pt",
    "v14_pseudo_head_delta.pt": ROOT / "experiments/scene_context_v13/outputs/checkpoints/v14_full_fixedsoup_pseudo_head_delta.pt",
    "v3_final_x3d_int8_compact.pt": ROOT / "outputs/final_legal_multimodal_v9/models/v3_final_x3d_int8_compact.pt",
    "v2_final_statistics_int8_compact.pt": ROOT / "outputs/final_legal_multimodal_v9/models/v2_final_statistics_int8_compact.pt",
    "yolo11n_compact.pt": ROOT / "outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt",
    "person_detector_int8_compact.pth": ROOT / "outputs/final_legal_multimodal_v9/models/person_detector_int8_compact.pth",
}

ARTIFACT_FILES = {
    "submission.csv": ROOT / "outputs/submissions/v14_fixedsoup_pseudo_pair_aware.csv",
    "predictions/final.npz": ROOT / "outputs/predictions/v14_fixedsoup_pseudo_pair_aware.npz",
    "predictions/domain_blend.npz": ROOT / "outputs/predictions/v14_fixedsoup_pseudo_domain_blend.npz",
    "predictions/fixedsoup.npz": ROOT / "outputs/predictions/v14_shared6_mixup_full_fixedsoup_uniform_tta.npz",
    "predictions/pseudo_adapt.npz": ROOT / "experiments/scene_context_v13/outputs/predictions/v14_full_fixedsoup_pseudo_adapt.npz",
    "validation/fixedsoup.json": ROOT / "experiments/scene_context_v13/outputs/shared6_mixup_fixedsoup_uniform_l3int6_l4b0_domain_pairaware.json",
    "validation/pseudo_c0p6_e2.json": ROOT / "experiments/scene_context_v13/outputs/fixedsoup_pseudo_c0p6_e2_frozen_domain_pairaware.json",
    "validation/test_report.json": ROOT / "experiments/scene_context_v13/outputs/v14_fixedsoup_pseudo_pair_aware.json",
    "metadata/samples.json": ROOT / "data/metadata/samples.json",
    "metadata/person_boxes.npz": ROOT / "outputs/final_legal_multimodal_v9/metadata/person_boxes.npz",
}

SOURCE_FILES = [
    "scripts/predict_r2plus1d.py",
    "scripts/evaluate_r2plus1d_checkpoint.py",
    "scripts/evaluate_repeat_decoder.py",
    "scripts/quantize_r2plus1d_hybrid.py",
    "scripts/soup_r2plus1d_checkpoints.py",
    "src/data.py",
    "src/r2plus1d_data.py",
    "src/r2plus1d_models.py",
    "experiments/scene_context_v13/adapt_pseudo_test.py",
    "experiments/scene_context_v13/build_pair_aware_submission.py",
    "experiments/scene_context_v13/build_test_pseudo_labels.py",
    "experiments/scene_context_v13/build_test_pseudo_composite.py",
    "experiments/scene_context_v13/blend_test_domain_candidate.py",
    "experiments/scene_context_v13/evaluate_frozen_domain_candidate.py",
    "experiments/scene_context_v13/run_test_pseudo_pipeline.sh",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_files(files, prefix):
    for relative, source in files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = OUTPUT / prefix / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    copy_files(MODEL_FILES, "models")
    copy_files(ARTIFACT_FILES, "")
    copy_files({path: ROOT / path for path in SOURCE_FILES}, "source")
    model_bytes = sum(path.stat().st_size for path in (OUTPUT / "models").iterdir())
    if model_bytes >= MODEL_LIMIT:
        raise ValueError(f"Model assets exceed limit: {model_bytes} >= {MODEL_LIMIT}")
    validation = json.loads((OUTPUT / "validation/pseudo_c0p6_e2.json").read_text())
    config = {
        "name": "final_legal_multimodal_v14",
        "architecture": "CNN/RNN/Transformer multimodal ensemble with pair-aware decoding",
        "candidate": "shared R(2+1)D-34 visual encoder, temporal CNN/LSTM sensor encoders, Transformer fusion",
        "training": "BF16, MixUp, full-train epoch soup 0.2/0.3/0.5 at epochs 10/14/18",
        "pseudo_adaptation": {
            "scope": "88,840 head and normalization parameters",
            "confidence_threshold": 0.6,
            "epochs": 2,
            "learning_rate": 0.0002,
            "retention_weight": 0.25,
            "application_scope": "unmatched environment-1 acquisition dates",
        },
        "local_validation": validation["combined"],
        "model_asset_bytes": model_bytes,
        "model_limit_bytes": MODEL_LIMIT,
        "slack_bytes": MODEL_LIMIT - model_bytes,
    }
    (OUTPUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    results = (
        "# Final Legal Multimodal V14\n\n"
        f"- Local subject-held-out score: {validation['combined']['correct']}/{validation['combined']['rows']} "
        f"({validation['combined']['accuracy']:.4%}).\n"
        f"- Model assets: {model_bytes:,} bytes; slack: {MODEL_LIMIT - model_bytes:,} bytes.\n"
        "- Selection protocol: fixed-soup domain gates selected before pseudo-adaptation and transferred to fold 1 blind.\n"
        "- Test pseudo-label self-training uses only model predictions and unlabeled test inputs.\n"
    )
    (OUTPUT / "RESULTS.md").write_text(results)
    files = {}
    for path in sorted(value for value in OUTPUT.rglob("*") if value.is_file()):
        relative = str(path.relative_to(OUTPUT))
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "name": config["name"],
        "model_asset_bytes": model_bytes,
        "model_limit_bytes": MODEL_LIMIT,
        "slack_bytes": MODEL_LIMIT - model_bytes,
        "submission_sha256": sha256(OUTPUT / "submission.csv"),
        "files": files,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
