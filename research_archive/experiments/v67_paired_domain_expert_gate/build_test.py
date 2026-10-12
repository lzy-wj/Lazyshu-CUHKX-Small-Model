#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import prepool
from experiments.v66_supervised_expert_transformer.model import load_bundle, predict
from experiments.v66_supervised_expert_transformer.run import MODALITIES, test_values
from src.data import get_records
from src.scratch_protocol import modality_presence


def main():
    records = get_records()
    validation = json.loads((HERE / "outputs/validation.json").read_text())
    weights = {int(key): value for key, value in validation["selected_weights"].items()}
    test = test_values(modality_presence(MODALITIES), records)
    bundle_path = (
        ROOT / "experiments/v24_full_refit/outputs/final/v66_supervised_expert_transformer.pt"
    )
    model, bundle = load_bundle(bundle_path)
    candidate, sequence = predict(
        model,
        test["experts"],
        test["indices"],
        test["domains"],
        test["presence"],
        records,
    )
    v57_archive = np.load(
        ROOT / "experiments/v24_full_refit/outputs/predictions/v57_round_transformer_distilled.npz"
    )
    if not np.array_equal(v57_archive["indices"], test["indices"]):
        raise ValueError("V57 and V67 test rows are not aligned")
    probabilities = v57_archive["probabilities"].astype(np.float32).copy()
    changed_rows = {}
    for domain, weight in weights.items():
        selected = test["domains"] == domain
        original = probabilities[selected].argmax(1)
        probabilities[selected] = (
            (1 - weight) * probabilities[selected] + weight * candidate[selected]
        )
        changed_rows[str(domain)] = {
            "rows": int(selected.sum()),
            "weight": weight,
            "changed_before_decode": int((probabilities[selected].argmax(1) != original).sum()),
        }
    probabilities, repeat_groups = prepool(
        probabilities,
        test["indices"],
        records,
        maximum_gap=150,
        minimum_similarity=0.7,
        power=1.0,
    )
    output = ROOT / "experiments/v24_full_refit/outputs/predictions/v67_paired_domain_expert_predecode.npz"
    np.savez(output, probabilities=probabilities.astype(np.float32), indices=test["indices"])
    report = {
        "bundle": str(bundle_path),
        "parameters": bundle["parameters"],
        "sequence": sequence,
        "selected_weights": weights,
        "domains": changed_rows,
        "repeat_groups": repeat_groups,
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

