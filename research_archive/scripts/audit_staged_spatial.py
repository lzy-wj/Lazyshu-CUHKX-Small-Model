import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records
from src.encoder_initialization import load_encoder_sources
from src.scratch_models import ScratchMultiModalModel
from src.validation_splits import load_diagnostic_split


def main():
    records = get_records()
    split_path = ROOT / "configs/spatial_grouped_f0.json"
    fit, inner, outer = load_diagnostic_split(split_path, records, 0)
    sources = []
    temporary = ROOT / "outputs/checkpoints/staged_spatial_audit_source"
    if temporary.exists():
        raise RuntimeError("Refusing to replace staged audit fixture")
    try:
        for modality in ["skeleton", "imu", "thermal"]:
            visual = "cnn_tokens" if modality == "thermal" else "legacy"
            model = ScratchMultiModalModel([modality], fusion="cross_transformer" if modality == "thermal" else "concat",
                                           visual_encoder=visual, thermal_view="motion_crop" if modality == "thermal" else "full",
                                           imu_representation="world" if modality == "imu" else "raw")
            available = __import__("src.scratch_protocol", fromlist=["modality_presence"]).modality_presence([modality]).any(1)
            source_train = [index for index in fit if available[index]]
            folder = temporary / modality
            folder.mkdir(parents=True)
            config = {"name": f"fixture_{modality}", "modalities": [modality], "fusion": "cross_transformer" if modality == "thermal" else "concat",
                      "visual_encoder": visual, "visual_feedforward": 384, "visual_multiplier": 1, "frames": 16, "image_size": 160,
                      "thermal_view": "motion_crop" if modality == "thermal" else "full",
                      "imu_representation": "world" if modality == "imu" else "raw", "skeleton_representation": "coordinates",
                      "strict_scratch": True, "fold": 0, "encoder_sources": [], "weights_source": None,
                      "diagnostic_split": str(split_path.relative_to(ROOT)),
                      "source_sha256": {"data/metadata/samples.json": __import__("hashlib").sha256((ROOT / "data/metadata/samples.json").read_bytes()).hexdigest()}}
            torch.save({"model": model.state_dict(), "config": config, "epoch": 1}, folder / "best.pt")
            (folder / "config.json").write_text(json.dumps(config) + "\n")
            (folder / "split.json").write_text(json.dumps({"train": source_train, "valid": inner, "outer_valid": outer}) + "\n")
            (folder / "completion.json").write_text(json.dumps({"status": "complete", "checkpoint_bytes": (folder / "best.pt").stat().st_size}) + "\n")
            sources.append(folder / "best.pt")
        target = ScratchMultiModalModel(["skeleton", "imu", "thermal"], fusion="cross_transformer", visual_encoder="cnn_tokens",
                                        thermal_view="motion_crop", imu_representation="world")
        provenance = load_encoder_sources(target, sources, records, fit, inner, 0, outer)
        assert len(provenance) == 3 and all(value["grouped"] for value in provenance)
        broken = json.loads((sources[0].parent / "split.json").read_text())
        broken["outer_valid"] = broken["outer_valid"][:-1]
        (sources[0].parent / "split.json").write_text(json.dumps(broken) + "\n")
        try:
            load_encoder_sources(target, sources, records, fit, inner, 0, outer)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched grouped outer split accepted")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    print("Grouped staged loading audit passed")


if __name__ == "__main__":
    main()
