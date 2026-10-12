import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records
from src.scratch_protocol import ablation_metrics, classification_metrics


ARCHITECTURES = ["cross", "self"]


def folder(seed, architecture, fold):
    if seed == 42:
        return ROOT / f"outputs/checkpoints/staged_legacy_{architecture}_f{fold}"
    return ROOT / f"outputs/checkpoints/rep_s43_legacy_{architecture}_f{fold}"


def load(path):
    with np.load(path) as values:
        return {name: values[name] for name in ["indices", "labels", "probabilities"]}


def average(values):
    for candidate in values[1:]:
        np.testing.assert_array_equal(candidate["indices"], values[0]["indices"])
        np.testing.assert_array_equal(candidate["labels"], values[0]["labels"])
    return {"indices": values[0]["indices"], "labels": values[0]["labels"],
            "probabilities": np.mean([value["probabilities"] for value in values], axis=0)}


def concatenate(values):
    return {name: np.concatenate([value[name] for value in values]) for name in ["indices", "labels", "probabilities"]}


def accuracy(values):
    return float((values["probabilities"].argmax(1) == values["labels"]).mean())


def main():
    records = get_records()
    predictions, report = {}, {"status": "complete", "single_seed": {}, "two_seed": {}, "folds": {}}
    for architecture in ARCHITECTURES:
        for seed in [42, 43]:
            outer = []
            for fold_index in [0, 1]:
                run = folder(seed, architecture, fold_index)
                completion = json.loads((run / "completion.json").read_text())
                config = json.loads((run / "config.json").read_text())
                assert completion["status"] == "complete" and completion["checkpoint_bytes"] < 100_000_000
                assert config["seed"] == seed and len(config["encoder_provenance"]) == 3
                assert all(source["grouped"] and source["fold"] == fold_index for source in config["encoder_provenance"])
                predictions[seed, architecture, fold_index, "valid"] = load(run / "validation.npz")
                predictions[seed, architecture, fold_index, "outer"] = load(run / "outer_valid.npz")
                outer.append(predictions[seed, architecture, fold_index, "outer"])
            combined = concatenate(outer)
            report["single_seed"][f"{architecture}_s{seed}"] = ablation_metrics(combined, records, ["skeleton", "imu", "thermal"])
        two_seed_outer = []
        for fold_index in [0, 1]:
            for partition in ["valid", "outer"]:
                predictions[architecture, fold_index, partition] = average(
                    [predictions[seed, architecture, fold_index, partition] for seed in [42, 43]])
            two_seed_outer.append(predictions[architecture, fold_index, "outer"])
        report["two_seed"][architecture] = ablation_metrics(concatenate(two_seed_outer), records, ["skeleton", "imu", "thermal"])
    selected_outer = []
    for fold_index in [0, 1]:
        candidates = {}
        for self_weight in [0, 0.25, 0.5, 0.75, 1]:
            values = dict(predictions["cross", fold_index, "valid"])
            values["probabilities"] = ((1 - self_weight) * predictions["cross", fold_index, "valid"]["probabilities"]
                                       + self_weight * predictions["self", fold_index, "valid"]["probabilities"])
            metrics = classification_metrics(values["labels"], values["probabilities"].argmax(1))
            candidates[self_weight] = metrics
        weight = max(candidates, key=lambda value: (candidates[value]["accuracy"], candidates[value]["macro_f1"], -abs(value - 0.5)))
        outer = dict(predictions["cross", fold_index, "outer"])
        outer["probabilities"] = ((1 - weight) * predictions["cross", fold_index, "outer"]["probabilities"]
                                  + weight * predictions["self", fold_index, "outer"]["probabilities"])
        selected_outer.append(outer)
        report["folds"][str(fold_index)] = {"selected_self_weight": weight,
            "inner_candidates": {str(value): candidates[value] for value in candidates},
            "selected_outer": ablation_metrics(outer, records, ["skeleton", "imu", "thermal"])}
    report["nested_selected_outer"] = ablation_metrics(concatenate(selected_outer), records, ["skeleton", "imu", "thermal"])
    report["selection"] = "Per fold, choose self-attention probability weight from {0,.25,.5,.75,1} using only the two-seed inner ensemble; apply once to outer predictions."
    report["caveats"] = ["Two seeds remain a small replication sample.", "Source encoders are also independently retrained for seed43.",
                         "Outer users are not used for architecture or blend selection in this report, but historical studies have exposed these folds.",
                         "Grouped fitting uses fewer clips than the previous final candidate; official test accuracy remains unknown."]
    destination = ROOT / "outputs/study_staged_spatial/legacy_replication.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 强CNN分阶段融合两种子复验", "", "| 融合 | seed42外层 | seed43外层 | 两种子概率平均 | Macro-F1 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for architecture in ARCHITECTURES:
        first = report["single_seed"][f"{architecture}_s42"]["all_validation"]
        second = report["single_seed"][f"{architecture}_s43"]["all_validation"]
        combined = report["two_seed"][architecture]["all_validation"]
        lines.append(f"| {architecture} | {first['accuracy']*100:.2f}% | {second['accuracy']*100:.2f}% | {combined['accuracy']*100:.2f}% | {combined['macro_f1']*100:.2f}% |")
    lines += ["", "## 仅按内层选择融合", ""]
    for fold_index, values in report["folds"].items():
        lines.append(f"- 折{fold_index}: self权重 {values['selected_self_weight']:.2f}，对应外层 {values['selected_outer']['all_validation']['accuracy']*100:.2f}%。")
    lines += [f"- 两折合并：{report['nested_selected_outer']['all_validation']['accuracy']*100:.2f}%，macro-F1 {report['nested_selected_outer']['all_validation']['macro_f1']*100:.2f}%。",
              "", "本地用户隔离验证，不是官方测试成绩；两个种子不足以消除训练和用户抽样不确定性。", ""]
    (ROOT / "docs/LEGACY_STAGED_REPLICATION_20260907.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
