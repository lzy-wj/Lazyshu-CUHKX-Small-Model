import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT
from src.scratch_protocol import classification_metrics


def main():
    source = ROOT / "outputs/study_spatial/auxiliary"
    report = {"status": "complete", "architectures": {}}
    for kind in ["cnn_pooled", "cnn_tokens", "patch_tokens"]:
        metrics = {}
        for partition in ["valid", "outer_valid"]:
            labels, probabilities = [], {name: [] for name in ["fusion", "skeleton", "imu", "thermal"]}
            for fold in [0, 1]:
                seeds = []
                for seed in [42, 43]:
                    with np.load(source / f"{kind}_s{seed}_f{fold}_{partition}.npz") as values:
                        seeds.append({name: values[name] for name in values.files})
                np.testing.assert_array_equal(seeds[0]["indices"], seeds[1]["indices"])
                np.testing.assert_array_equal(seeds[0]["labels"], seeds[1]["labels"])
                labels.append(seeds[0]["labels"])
                for name in probabilities:
                    probabilities[name].append((seeds[0][name + "_probabilities"] + seeds[1][name + "_probabilities"]) / 2)
            labels = np.concatenate(labels)
            metrics[partition] = {name: classification_metrics(labels, np.concatenate(values).argmax(1))
                                  for name, values in probabilities.items()}
        report["architectures"][kind] = metrics
    report["interpretation_boundary"] = [
        "Auxiliary heads share encoders and training with fusion; they are not independently trained unimodal models.",
        "Their loss weight is auxiliary_weight divided by three, so branch accuracy underestimates a matched unimodal objective.",
        "A weak auxiliary head can still provide useful fused representations; this analysis diagnoses, not proves, fusion failure.",
    ]
    (source / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for kind, partitions in report["architectures"].items():
        print(kind, {partition: {name: round(value["accuracy"] * 100, 2) for name, value in metrics.items()}
                     for partition, metrics in partitions.items()})


if __name__ == "__main__":
    main()
