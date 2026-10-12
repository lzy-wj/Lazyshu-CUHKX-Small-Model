import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records
from src.scratch_models import ScratchMultiModalModel
from src.validation_splits import make_grouped_split


def write_frozen(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError(f"Refusing to replace frozen artifact: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    records = get_records()
    jobs, parameters = [], {}
    for fold in [0, 1]:
        split_path = ROOT / f"configs/spatial_grouped_f{fold}.json"
        split = make_grouped_split(records, fold)
        write_frozen(split_path, split)
        print("Split", fold, {key: {"clips": len(split[key]), "users": sorted({records[index]["user"] for index in split[key]})}
                              for key in ["train", "valid", "outer_valid"]})
        for kind in ["cnn_pooled", "cnn_tokens", "patch_tokens"]:
            feedforward = 1344 if kind == "patch_tokens" else 384
            model = ScratchMultiModalModel(["skeleton", "imu", "thermal"], fusion="cross_transformer",
                                          visual_encoder=kind, visual_feedforward=feedforward)
            parameters[kind] = sum(parameter.numel() for parameter in model.parameters())
            jobs.append({"name": f"spatial_v1_{kind}_f{fold}", "modalities": ["skeleton", "imu", "thermal"],
                         "fusion": "cross_transformer", "visual_encoder": kind, "visual_feedforward": feedforward,
                         "fold": fold, "diagnostic_split": str(split_path.relative_to(ROOT))})
    assert max(parameters.values()) / min(parameters.values()) < 1.01
    config = {"name": "spatial_tokens_v1", "seed": 42, "epochs": 60, "batch_size": 24, "frames": 16,
              "image_size": 160, "lr": 0.001, "workers": 4, "eval_every": 2, "patience": 60,
              "gpus": [2, 3, 7], "queue_hours": 4, "warmup_epochs": 5,
              "thermal_view": "motion_crop", "imu_representation": "world", "swanlab_mode": "online",
              "parameters": parameters, "jobs": jobs,
              "protocol": "Fresh scratch models. Inner and outer validation both user-disjoint from fitting. Inner selects checkpoints; outer evaluated once after training. Historically exposed outer folds are not pristine tests.",
              "hypotheses": ["Identical CNN/token Transformer weights and fusion: retaining 10x10 versus pooling to 2x2 spatial tokens improves cross-user recognition.",
                             "Direct 16x16 patch projection can compete with CNN tokenization under matched total parameters and optimizer steps; FF width differs to match capacity."],
              "selection": "Each fold selects architecture on its own inner accuracy, tie by macro-F1 then preset order. Nested outer summary uses fold-local choices. Cross-fold pooled inner ranking is exploratory because users overlap between one fold's inner and another fold's outer.",
              "caveats": "CNN pooled control is not the legacy TCN visual baseline. Equal optimizer steps and parameter counts do not imply equal FLOPs. Three-modal from-scratch training is not the earlier staged ensemble."}
    write_frozen(ROOT / "configs/spatial_token_study.json", config)
    print("Parameters", parameters)


if __name__ == "__main__":
    main()
