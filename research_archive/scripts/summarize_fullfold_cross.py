import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records
from src.scratch_protocol import ablation_metrics, classification_metrics


def load(path):
    with np.load(path) as values:
        return {name: values[name] for name in ["indices", "labels", "probabilities"]}


def average(values):
    for value in values[1:]:
        np.testing.assert_array_equal(value["indices"], values[0]["indices"])
        np.testing.assert_array_equal(value["labels"], values[0]["labels"])
    return {"indices": values[0]["indices"], "labels": values[0]["labels"],
            "probabilities": np.mean([value["probabilities"] for value in values], axis=0)}


def concatenate(values):
    return {name: np.concatenate([value[name] for value in values]) for name in ["indices", "labels", "probabilities"]}


def main():
    records = get_records()
    folds, report = {}, {"status": "complete", "fixed_self_weight": 0.75, "folds": {}}
    for fold in [0, 1]:
        self_models = [load(ROOT / f"outputs/checkpoints/{name}/validation.npz") for name in
                       [f"fusion_input_v1_staged_f{fold}", f"fusion_input_v1_staged_s43_f{fold}"]]
        cross_models = [load(ROOT / f"outputs/checkpoints/fullfold_cross_s{seed}_f{fold}/validation.npz") for seed in [42, 43]]
        self_average, cross_average = average(self_models), average(cross_models)
        blend = dict(self_average)
        blend["probabilities"] = 0.75 * self_average["probabilities"] + 0.25 * cross_average["probabilities"]
        folds[fold] = {"self": self_average, "cross": cross_average, "blend": blend}
        report["folds"][str(fold)] = {name: ablation_metrics(values, records, ["skeleton", "imu", "thermal"])
                                      for name, values in folds[fold].items()}
        report["folds"][str(fold)]["single_cross"] = {
            str(seed): ablation_metrics(values, records, ["skeleton", "imu", "thermal"])
            for seed, values in zip([42, 43], cross_models)}
    for name in ["self", "cross", "blend"]:
        report[name] = ablation_metrics(concatenate([folds[fold][name] for fold in [0, 1]]), records,
                                                ["skeleton", "imu", "thermal"])
    baseline = concatenate([folds[fold]["self"] for fold in [0, 1]])
    candidate = concatenate([folds[fold]["blend"] for fold in [0, 1]])
    labels = baseline["labels"]
    base_correct = baseline["probabilities"].argmax(1) == labels
    candidate_correct = candidate["probabilities"].argmax(1) == labels
    users = np.asarray([records[index]["user"] for index in baseline["indices"]])
    unique = np.unique(users)
    differences = candidate_correct.astype(float) - base_correct.astype(float)
    sums = np.asarray([differences[users == user].sum() for user in unique])
    counts = np.asarray([(users == user).sum() for user in unique])
    generator = np.random.default_rng(20260907)
    draws = generator.integers(0, len(unique), size=(10000, len(unique)))
    distribution = sums[draws].sum(1) / counts[draws].sum(1)
    report["paired_blend_vs_self"] = {"accuracy_delta": float(differences.mean()),
        "user_cluster_bootstrap_95_interval": np.quantile(distribution, [0.025, 0.975]).tolist(),
        "per_user_delta": {str(user): float(differences[users == user].mean()) for user in unique}}
    report["exploratory_weight_curve"] = {}
    for weight in [0, 0.25, 0.5, 0.75, 1]:
        probability = weight * baseline["probabilities"] + (1 - weight) * concatenate([folds[fold]["cross"] for fold in [0, 1]])["probabilities"]
        report["exploratory_weight_curve"][str(weight)] = classification_metrics(labels, probability.argmax(1))
    report["decision"] = "promote_to_full_refit" if (report["blend"]["all_validation"]["accuracy"] > report["self"]["all_validation"]["accuracy"]
        and all(report["folds"][str(fold)]["blend"]["all_validation"]["accuracy"] >= report["folds"][str(fold)]["self"]["all_validation"]["accuracy"] for fold in [0, 1])) else "keep_existing_candidate"
    report["caveats"] = ["The 0.75 self weight was fixed on separate grouped-inner predictions before these full-fold cross models were trained.",
                         "The original self models and full-fold cross models use the same fold-specific scratch source encoders.",
                         "These validation folds have extensive historical exposure and do not estimate an untouched test score.",
                         "Bootstrap covers eight fixed validation users, not retraining variance. Official test labels remain unavailable."]
    (ROOT / "outputs/study_staged_spatial/fullfold_cross.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 完整训练折查询融合复核", "", "| 预测 | 折0 | 折1 | 两折合并 | Macro-F1 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in ["self", "cross", "blend"]:
        lines.append(f"| {name} | {report['folds']['0'][name]['all_validation']['accuracy']*100:.2f}% | {report['folds']['1'][name]['all_validation']['accuracy']*100:.2f}% | {report[name]['all_validation']['accuracy']*100:.2f}% | {report[name]['all_validation']['macro_f1']*100:.2f}% |")
    lower, upper = np.asarray(report["paired_blend_vs_self"]["user_cluster_bootstrap_95_interval"]) * 100
    lines += ["", f"- 固定75% self + 25% cross相对原候选：{report['paired_blend_vs_self']['accuracy_delta']*100:+.2f}pp。",
              f"- 探索性用户聚类区间：[{lower:+.2f}, {upper:+.2f}]pp。", f"- 决策：{report['decision']}。", "",
              "比例来自独立的训练侧分组内层实验；本表不重新选择比例。验证折已历史暴露，官方测试未提交。", ""]
    (ROOT / "docs/FULLFOLD_CROSS_RESULTS_20260907.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
