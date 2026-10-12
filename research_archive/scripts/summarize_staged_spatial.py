import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records
from src.scratch_protocol import ablation_metrics


def load(path):
    with np.load(path) as values:
        return {name: values[name] for name in ["indices", "labels", "probabilities"]}


def join(values):
    return {name: np.concatenate([value[name] for value in values]) for name in ["indices", "labels", "probabilities"]}


def main():
    records = get_records()
    report = {"status": "complete", "sources": {}, "folds": {}}
    staged_predictions, baseline_predictions = [], []
    for fold in [0, 1]:
        for modality in ["skeleton", "imu", "thermal"]:
            folder = ROOT / f"outputs/checkpoints/staged_spatial_v1_{modality}_f{fold}"
            completion = json.loads((folder / "completion.json").read_text())
            diagnostics = json.loads((folder / "diagnostic_best.json").read_text())
            config = json.loads((folder / "config.json").read_text())
            assert completion["status"] == "complete" and not config["encoder_sources"] and config["selection_partition"] == "inner_unseen_users"
            report["sources"][f"{modality}_f{fold}"] = {"epoch": completion["best_epoch"],
                "inner_accuracy": diagnostics["inner_valid"]["accuracy"], "outer_accuracy": diagnostics["outer_valid"]["accuracy"],
                "parameters": completion["parameters"], "checkpoint_bytes": completion["checkpoint_bytes"]}
        staged_folder = ROOT / f"outputs/checkpoints/staged_spatial_v1_fusion_f{fold}"
        baseline_folder = ROOT / f"outputs/checkpoints/spatial_v1_cnn_tokens_f{fold}"
        staged = load(staged_folder / "outer_valid.npz")
        baseline = load(baseline_folder / "outer_valid.npz")
        np.testing.assert_array_equal(staged["indices"], baseline["indices"])
        np.testing.assert_array_equal(staged["labels"], baseline["labels"])
        staged_diagnostics = json.loads((staged_folder / "diagnostic_best.json").read_text())
        baseline_diagnostics = json.loads((baseline_folder / "diagnostic_best.json").read_text())
        config = json.loads((staged_folder / "config.json").read_text())
        assert len(config["encoder_provenance"]) == 3 and all(source["grouped"] for source in config["encoder_provenance"])
        assert all(source["train_clips"] <= config["train_clips"] for source in config["encoder_provenance"])
        report["folds"][str(fold)] = {
            "end_to_end": {"inner_accuracy": baseline_diagnostics["inner_valid"]["accuracy"], "outer_accuracy": baseline_diagnostics["outer_valid"]["accuracy"]},
            "staged": {"inner_accuracy": staged_diagnostics["inner_valid"]["accuracy"], "outer_accuracy": staged_diagnostics["outer_valid"]["accuracy"]},
            "outer_accuracy_delta": staged_diagnostics["outer_valid"]["accuracy"] - baseline_diagnostics["outer_valid"]["accuracy"],
            "staged_best_epoch": json.loads((staged_folder / "completion.json").read_text())["best_epoch"]}
        staged_predictions.append(staged)
        baseline_predictions.append(baseline)
    staged = join(staged_predictions)
    baseline = join(baseline_predictions)
    staged_metrics = ablation_metrics(staged, records, ["skeleton", "imu", "thermal"])
    baseline_metrics = ablation_metrics(baseline, records, ["skeleton", "imu", "thermal"])
    report["pooled"] = {"end_to_end": baseline_metrics, "staged": staged_metrics,
                        "accuracy_delta": staged_metrics["all_validation"]["accuracy"] - baseline_metrics["all_validation"]["accuracy"]}
    report["decision"] = "replicate" if report["pooled"]["accuracy_delta"] > 0 and all(value["outer_accuracy_delta"] > 0 for value in report["folds"].values()) else "do_not_promote"
    report["caveats"] = ["Seed42 only; replicate before promotion.", "Both source and fusion checkpoint epochs use inner labels.",
                         "Outer folds were historically exposed, though this run evaluates them only after training.",
                         "Staging uses more compute than end-to-end and tests training procedure, not architecture alone."]
    destination = ROOT / "outputs/study_staged_spatial/results.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 独立编码器预训练对照", "", "| 折 | 端到端内层 | 分阶段内层 | 端到端外层 | 分阶段外层 | 外层变化 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for fold, values in report["folds"].items():
        lines.append(f"| {fold} | {values['end_to_end']['inner_accuracy']*100:.2f}% | {values['staged']['inner_accuracy']*100:.2f}% | {values['end_to_end']['outer_accuracy']*100:.2f}% | {values['staged']['outer_accuracy']*100:.2f}% | {values['outer_accuracy_delta']*100:+.2f}pp |")
    lines += ["", f"- 两折端到端：{baseline_metrics['all_validation']['accuracy']*100:.2f}%",
              f"- 两折分阶段：{staged_metrics['all_validation']['accuracy']*100:.2f}%",
              f"- 变化：{report['pooled']['accuracy_delta']*100:+.2f}pp",
              f"- 预注册决策：{report['decision']}", "", "单种子；只在两折同向且汇总提升时进入复验。官方测试未提交。", ""]
    (ROOT / "docs/STAGED_SPATIAL_RESULTS_20260907.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
