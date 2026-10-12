import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records
from src.scratch_protocol import ablation_metrics


CANDIDATES = ["staged_spatial_v1_fusion", "staged_legacy_cross", "staged_legacy_self"]


def load(path):
    with np.load(path) as values:
        return {name: values[name] for name in ["indices", "labels", "probabilities"]}


def main():
    records = get_records()
    predictions, report = {}, {"status": "complete", "candidates": {}, "thermal_sources": {}}
    for fold in [0, 1]:
        token_source = json.loads((ROOT / f"outputs/checkpoints/staged_spatial_v1_thermal_f{fold}/diagnostic_best.json").read_text())
        legacy_source = json.loads((ROOT / f"outputs/checkpoints/grouped_legacy_thermal_f{fold}/diagnostic_best.json").read_text())
        report["thermal_sources"][str(fold)] = {
            "spatial_tokens": {key: token_source[key]["accuracy"] for key in ["inner_valid", "outer_valid"]},
            "legacy_cnn_tcn": {key: legacy_source[key]["accuracy"] for key in ["inner_valid", "outer_valid"]}}
        for candidate in CANDIDATES:
            folder = ROOT / f"outputs/checkpoints/{candidate}_f{fold}"
            completion = json.loads((folder / "completion.json").read_text())
            config = json.loads((folder / "config.json").read_text())
            diagnostic = json.loads((folder / "diagnostic_best.json").read_text())
            assert completion["status"] == "complete" and config["selection_partition"] == "inner_unseen_users"
            assert len(config["encoder_provenance"]) == 3 and all(value["grouped"] for value in config["encoder_provenance"])
            values = load(folder / "outer_valid.npz")
            predictions[candidate, fold] = values
            report["candidates"][f"{candidate}_f{fold}"] = {
                "inner_accuracy": diagnostic["inner_valid"]["accuracy"], "outer_accuracy": diagnostic["outer_valid"]["accuracy"],
                "epoch": completion["best_epoch"], "parameters": completion["parameters"], "checkpoint_bytes": completion["checkpoint_bytes"]}
    pooled = {}
    for candidate in CANDIDATES:
        values = {name: np.concatenate([predictions[candidate, fold][name] for fold in [0, 1]])
                  for name in ["indices", "labels", "probabilities"]}
        pooled[candidate] = ablation_metrics(values, records, ["skeleton", "imu", "thermal"])
    report["pooled"] = pooled
    choices = {}
    for fold in [0, 1]:
        choices[str(fold)] = max(CANDIDATES, key=lambda candidate: (report["candidates"][f"{candidate}_f{fold}"]["inner_accuracy"], -CANDIDATES.index(candidate)))
    nested = {name: np.concatenate([predictions[choices[str(fold)], fold][name] for fold in [0, 1]])
              for name in ["indices", "labels", "probabilities"]}
    report["fold_local_inner_choices"] = choices
    report["nested_selected_outer"] = ablation_metrics(nested, records, ["skeleton", "imu", "thermal"])
    report["caveats"] = ["Seed42 architecture comparison; replicate selected direction before promotion.",
                         "Legacy self-attention and cross-attention share sources but differ in fusion topology and parameter count.",
                         "All outer folds were evaluated once after each run, but they have historical exposure from earlier studies.",
                         "Architecture selection must use fold-local inner scores, not pooled outer results."]
    destination = ROOT / "outputs/study_staged_spatial/legacy_comparison.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 分阶段视觉与融合对照", "", "| 候选 | 折0内层 | 折1内层 | 折0外层 | 折1外层 | 两折外层 | Macro-F1 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for candidate in CANDIDATES:
        fold0, fold1 = [report["candidates"][f"{candidate}_f{fold}"] for fold in [0, 1]]
        metrics = pooled[candidate]["all_validation"]
        lines.append(f"| {candidate} | {fold0['inner_accuracy']*100:.2f}% | {fold1['inner_accuracy']*100:.2f}% | {fold0['outer_accuracy']*100:.2f}% | {fold1['outer_accuracy']*100:.2f}% | {metrics['accuracy']*100:.2f}% | {metrics['macro_f1']*100:.2f}% |")
    lines += ["", f"- 按折内层选择：{choices}", f"- 对应外层汇总：{report['nested_selected_outer']['all_validation']['accuracy']*100:.2f}%", "",
              "这是单种子开发结果；外层不用于改变每折选择。官方测试未提交。", ""]
    (ROOT / "docs/LEGACY_STAGED_COMPARISON_20260907.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
