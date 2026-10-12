import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_spatial_study import KINDS, join_predictions, load_predictions, paired_user_bootstrap
from src.data import ROOT, get_records
from src.scratch_protocol import ablation_metrics, classification_metrics


def main():
    records = get_records()
    sources = {42: ROOT / "outputs/study_spatial/results.json", 43: ROOT / "outputs/study_spatial_s43/results.json"}
    results = {seed: json.loads(path.read_text()) for seed, path in sources.items()}
    assert all(result["status"] == "complete" for result in results.values())
    names = {int(path.name.split("_", 1)[0]): path.name.split("_", 1)[1]
             for path in (ROOT / "data/raw/HAR/data/Skeleton").iterdir() if path.is_dir()}
    predictions, report = {}, {"status": "complete", "architectures": {}, "sources": {str(seed): str(path.relative_to(ROOT)) for seed, path in sources.items()}}
    for kind in KINDS:
        fold_metrics = {}
        for fold in [0, 1]:
            split = json.loads((ROOT / f"configs/spatial_grouped_f{fold}.json").read_text())
            for partition in ["valid", "outer_valid"]:
                arrays = []
                for seed in [42, 43]:
                    prefix = "spatial_v1" if seed == 42 else "spatial_v1_s43"
                    file = "validation.npz" if partition == "valid" else "outer_valid.npz"
                    arrays.append(load_predictions(ROOT / f"outputs/checkpoints/{prefix}_{kind}_f{fold}/{file}", split[partition], records))
                averaged = dict(arrays[0])
                averaged["probabilities"] = (arrays[0]["probabilities"] + arrays[1]["probabilities"]) / 2
                predictions[kind, fold, partition] = averaged
            fold_metrics[str(fold)] = {partition: ablation_metrics(predictions[kind, fold, partition], records, ["skeleton", "imu", "thermal"])
                                       for partition in ["valid", "outer_valid"]}
        pooled = join_predictions([predictions[kind, fold, "outer_valid"] for fold in [0, 1]])
        seed_scores = {str(seed): result["architectures"][kind]["outer"]["all_validation"]["accuracy"] for seed, result in results.items()}
        values = ablation_metrics(pooled, records, ["skeleton", "imu", "thermal"])
        labels, guessed = pooled["labels"], pooled["probabilities"].argmax(1)
        body = (labels >= 28) & (labels <= 36)
        report["architectures"][kind] = {"single_seed_outer_accuracy": seed_scores,
                                         "mean_single_seed_accuracy": float(np.mean(list(seed_scores.values()))),
                                         "two_seed_probability_ensemble": values,
                                         "folds": fold_metrics,
                                         "whole_body": classification_metrics(labels[body], guessed[body]),
                                         "other_31": classification_metrics(labels[~body], guessed[~body])}
        if kind != "cnn_pooled":
            baseline = join_predictions([predictions["cnn_pooled", fold, "outer_valid"] for fold in [0, 1]])
            report["architectures"][kind]["paired_vs_pooled"] = paired_user_bootstrap(baseline, pooled, records)
            baseline_metrics = classification_metrics(baseline["labels"], baseline["probabilities"].argmax(1))
            classes = []
            for label in range(40):
                before, after = baseline_metrics["class_recall"][label], values["all_validation"]["class_recall"][label]
                classes.append({"label": label, "name": names[label], "support": values["all_validation"]["class_support"][label],
                                "baseline_recall": before, "candidate_recall": after, "delta": after - before})
            report["architectures"][kind]["class_changes"] = sorted(classes, key=lambda value: value["delta"], reverse=True)
    choices = {}
    for fold in [0, 1]:
        def rank(kind):
            scores = report["architectures"][kind]["folds"][str(fold)]["valid"]["all_validation"]
            return scores["accuracy"], scores["macro_f1"], -KINDS.index(kind)
        choices[str(fold)] = max(KINDS, key=rank)
    nested = join_predictions([predictions[choices[str(fold)], fold, "outer_valid"] for fold in [0, 1]])
    report["fold_local_inner_choices"] = choices
    report["nested_selected_outer"] = ablation_metrics(nested, records, ["skeleton", "imu", "thermal"])
    report["caveats"] = ["The seed43 replication and equal-weight ensemble were fixed before reading aggregate outer results.",
                         "One fold's inner users can occur in another fold's outer set: no shared-winner unbiased outer claim.",
                         "Historical outer-fold exposure and only two seeds limit statistical conclusions.",
                         "Grouped inner holdouts reduce fitting data compared with the previous staged 56.96% ensemble; compare against this study's pooled control.",
                         "No official test submission or replacement of the existing final bundle."]
    (ROOT / "outputs/study_spatial/replicated_results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 空间 token 两种子复验", "", "三种架构 × 两个用户划分 × 两个随机种子，共 12 次从零训练。", "",
             "| 架构 | seed42 外层 | seed43 外层 | 单模型均值 | 两种子概率平均 | 平均后 macro-F1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for kind, value in report["architectures"].items():
        scores = value["single_seed_outer_accuracy"]
        ensemble = value["two_seed_probability_ensemble"]["all_validation"]
        lines.append(f"| {kind} | {scores['42'] * 100:.2f}% | {scores['43'] * 100:.2f}% | {value['mean_single_seed_accuracy'] * 100:.2f}% | {ensemble['accuracy'] * 100:.2f}% | {ensemble['macro_f1'] * 100:.2f}% |")
    lines += ["", "## 固定对照的变化", ""]
    for kind in KINDS[1:]:
        value = report["architectures"][kind]
        difference = value["paired_vs_pooled"]
        lower, upper = np.asarray(difference["user_cluster_bootstrap_95_interval"]) * 100
        lines.append(f"- {kind} 相对 cnn_pooled 两种子平均：{difference['accuracy_delta'] * 100:+.2f} 个百分点；用户聚类区间 [{lower:+.2f}, {upper:+.2f}]。")
    lines += ["", "## 内层选型后的外层评估", "", f"- 每折内层选择：{choices}",
              f"- 嵌套选型外层 accuracy：{report['nested_selected_outer']['all_validation']['accuracy'] * 100:.2f}%。", "",
              "## 解释边界", "", "- 以上均为本地跨用户验证，不是 Kaggle 官方测试分数。", "- 样本较少，两个种子不足以证明普遍优越；用户区间也不包含训练随机性。",
              "- 训练样本数少于此前 staged 模型，不能直接把 56.96% 当本轮基线。", "- 全部对照共享新的查询式 Transformer 融合；纯 patch 组也不是所有传感器都改成纯 Transformer。",
              "- 详细逐类、逐用户结果见 outputs/study_spatial/replicated_results.json。", ""]
    (ROOT / "docs/SPATIAL_REPLICATION_20260907.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
