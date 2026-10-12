import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_spatial_study import write_frozen
from src.data import ROOT
from src.scratch_models import ScratchMultiModalModel


def main():
    base = json.loads((ROOT / "configs/spatial_token_study.json").read_text())
    audit = json.loads((ROOT / "outputs/study_spatial/audit.json").read_text())
    assert audit["status"] == "passed"
    modalities = ["skeleton", "imu", "thermal"]
    jobs = []
    for fold in [0, 1]:
        split = f"configs/spatial_grouped_f{fold}.json"
        sources = []
        for modality in modalities:
            name = f"staged_spatial_v1_{modality}_f{fold}"
            source = {"name": name, "modalities": [modality], "fusion": "concat", "fold": fold,
                      "diagnostic_split": split, "auxiliary_weight": 0.15,
                      "thermal_view": "full", "imu_representation": "world" if modality == "imu" else "raw"}
            if modality == "thermal":
                source.update({"fusion": "cross_transformer", "visual_encoder": "cnn_tokens", "visual_feedforward": 384,
                               "thermal_view": "motion_crop"})
            jobs.append(source)
            sources.append(f"outputs/checkpoints/{name}/best.pt")
        jobs.append({"name": f"staged_spatial_v1_fusion_f{fold}", "modalities": modalities,
                     "fusion": "cross_transformer", "visual_encoder": "cnn_tokens", "visual_feedforward": 384,
                     "fold": fold, "diagnostic_split": split, "encoder_sources": sources,
                     "thermal_view": "motion_crop", "imu_representation": "world"})
    model = ScratchMultiModalModel(modalities, fusion="cross_transformer", visual_encoder="cnn_tokens")
    config = {key: base[key] for key in ["seed", "epochs", "batch_size", "frames", "image_size", "lr", "workers",
                                                  "eval_every", "patience", "gpus", "queue_hours", "warmup_epochs",
                                                  "swanlab_mode"]}
    config.update({"name": "staged_spatial_v1", "jobs": jobs, "parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "protocol": "Seed42 causal comparison on the same grouped train/inner/outer partitions as spatial_v1_cnn_tokens. Each source is a single-modality scratch model trained only on the target fit users and selected on the same inner users. Fusion loads encoder tensors only, then fine-tunes all weights. Outer users are evaluated once after training.",
                   "hypothesis": "Independent modality objectives prevent weak-modality collapse; staged fusion should improve both auxiliary Thermal/IMU accuracy and fused outer accuracy over spatial_v1_cnn_tokens with identical target architecture.",
                   "primary_comparison": {"fold0": "spatial_v1_cnn_tokens_f0", "fold1": "spatial_v1_cnn_tokens_f1"},
                   "selection": "Checkpoint selection uses inner accuracy. Compare staged versus end-to-end pooled outer accuracy only after both folds complete; replicate before promotion.",
                   "caveats": "Source and fusion both inspect inner labels for checkpoint selection. This does not leak outer users but increases inner selection exposure. Compute is intentionally larger than end-to-end training."})
    destination = ROOT / "configs/staged_spatial_study.json"
    write_frozen(destination, config)
    staged_audit = copy.deepcopy(audit)
    paths = ["src/encoder_initialization.py", "scripts/train.py", "src/scratch_models.py", "scripts/run_scratch_study.py",
             "scripts/prepare_staged_spatial_study.py",
             "configs/staged_spatial_study.json"]
    staged_audit["source_sha256"].update({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths})
    staged_audit["scope"] = "Spatial audit plus grouped single-modality encoder provenance and staged target loading."
    folder = ROOT / "outputs/study_staged_spatial"
    folder.mkdir(parents=True, exist_ok=True)
    write_frozen(folder / "audit.json", staged_audit)
    print("Frozen", len(jobs), "jobs", config["parameters"], "target parameters")


if __name__ == "__main__":
    main()
