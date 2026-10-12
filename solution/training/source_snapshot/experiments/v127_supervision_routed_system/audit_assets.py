#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ASSETS = [
    "experiments/v24_full_refit/outputs/final/v24_full_warm60_soup_int8.pt",
    "outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt",
    "experiments/v24_full_refit/outputs/final/v24_full_warm60_pseudo_head_delta.pt",
    "experiments/v87_v24_highres_transfer/outputs/v24_highres_pseudo_head_delta.pt",
    "experiments/v103_full_protected_adapters/outputs/v103_full_protected_dual/adapter.pt",
    "experiments/v104_domain_gated_adapters/outputs/v104_target_old_cross/adapted_state.pt",
    "experiments/v104_domain_gated_adapters/outputs/v104_target_refreshed_cross/adapted_state.pt",
    "experiments/v114_paired_person_fusion/outputs/v114_full_source16/adapted_state.pt",
    "experiments/v114_paired_person_fusion/outputs/v114_full_source20/adapted_state.pt",
    "experiments/v114_paired_person_fusion/outputs/v114_full_source21/adapted_state.pt",
    "experiments/v24_full_refit/outputs/final/v52_domain_logit_residual_conv1d.pt",
    "experiments/v24_full_refit/outputs/final/v57_round_residual_transformer.pt",
    "experiments/v24_full_refit/outputs/final/v71_gated_film_transformer.pt",
]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    assets = []
    for relative in ASSETS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        assets.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)}
        )
    total = sum(item["bytes"] for item in assets)
    report = {
        "competition_limit_bytes": 100_000_000,
        "assets": assets,
        "training_only_excluded": [
            "experiments/v68_film_full_refit outputs used only to distill V71"
        ],
        "total_bytes": total,
        "slack_bytes": 100_000_000 - total,
        "pass": total < 100_000_000,
    }
    (HERE / "asset_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("total_bytes", "slack_bytes", "pass")}, indent=2))


if __name__ == "__main__":
    main()
