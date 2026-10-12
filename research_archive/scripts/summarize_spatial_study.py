import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records
from src.scratch_protocol import ablation_metrics, classification_metrics
from src.validation_splits import validate_diagnostic_split


KINDS = ["cnn_pooled", "cnn_tokens", "patch_tokens"]


def load_predictions(path, expected, records):
    with np.load(path) as archive:
        result = {name: archive[name] for name in ["indices", "labels", "probabilities"]}
    np.testing.assert_array_equal(result["indices"], expected)
    np.testing.assert_array_equal(result["labels"], [records[index]["label"] for index in expected])
    assert result["probabilities"].shape == (len(expected), 40)
    assert np.isfinite(result["probabilities"]).all()
    np.testing.assert_allclose(result["probabilities"].sum(1), 1, atol=1e-5)
    return result


def join_predictions(values):
    return {key: np.concatenate([value[key] for value in values]) for key in ["indices", "labels", "probabilities"]}


def paired_user_bootstrap(baseline, candidate, records):
    np.testing.assert_array_equal(baseline["indices"], candidate["indices"])
    users = np.asarray([records[index]["user"] for index in baseline["indices"]])
    unique = np.unique(users)
    delta = (candidate["probabilities"].argmax(1) == candidate["labels"]).astype(float)
    delta -= baseline["probabilities"].argmax(1) == baseline["labels"]
    sums = np.asarray([delta[users == user].sum() for user in unique])
    counts = np.asarray([(users == user).sum() for user in unique])
    generator = np.random.default_rng(20260907)
    draws = generator.integers(0, len(unique), size=(10000, len(unique)))
    distribution = sums[draws].sum(1) / counts[draws].sum(1)
    return {"accuracy_delta": float(delta.mean()), "user_cluster_bootstrap_95_interval": np.quantile(distribution, [0.025, 0.975]).tolist(),
            "users": len(unique), "per_user_delta": {str(user): float(delta[users == user].mean()) for user in unique},
            "caveat": "Exploratory fixed-checkpoint interval across only eight outer users; excludes training-seed uncertainty and historical selection bias."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/spatial_token_study.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/study_spatial")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs/SPATIAL_STUDY_RESULTS_20260907.md")
    args = parser.parse_args()
    records = get_records()
    configuration = json.loads(args.config.read_text())
    audit = json.loads((args.output / "audit.json").read_text())
    assert audit["status"] == "passed"
    for name, digest in audit["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    outcomes, inner_outputs, outer_outputs = {}, {}, {}
    for job in configuration["jobs"]:
        name, kind, fold = job["name"], job["visual_encoder"], job["fold"]
        folder = ROOT / "outputs/checkpoints" / name
        completion = json.loads((folder / "completion.json").read_text())
        assert completion["status"] == "complete"
        assert (ROOT / f"outputs/logs/{name}.exit").read_text().strip() == "0"
        config = json.loads((folder / "config.json").read_text())
        assert config["selection_partition"] == "inner_unseen_users" and not config["pretrained"]
        assert config["init"] is None and not config["encoder_sources"]
        assert config["visual_encoder"] == kind
        assert config["parameters"] == configuration["parameters"][kind]
        assert config["epochs"] == configuration["epochs"] and config["batch_size"] == configuration["batch_size"] and config["seed"] == configuration["seed"]
        split = json.loads((ROOT / job["diagnostic_split"]).read_text())
        fit, inner, outer = validate_diagnostic_split(split, records, fold)
        assert json.loads((folder / "split.json").read_text()) == {"train": fit, "valid": inner, "outer_valid": outer}
        inner_outputs[kind, fold] = load_predictions(folder / "validation.npz", inner, records)
        outer_outputs[kind, fold] = load_predictions(folder / "outer_valid.npz", outer, records)
        clean = load_predictions(folder / "clean_train.npz", fit, records)
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        assert len(history) == 60
        best = max((row for row in history if "valid/accuracy" in row), key=lambda row: row["valid/accuracy"])
        assert best["epoch"] == completion["best_epoch"]
        diagnostic_history = [json.loads(line) for line in (folder / "diagnostic_history.jsonl").read_text().splitlines()]
        assert all("outer_valid" not in row and "clean_train" not in row for row in diagnostic_history)
        checkpoint = folder / "best.pt"
        assert checkpoint.stat().st_size == completion["checkpoint_bytes"] < 100_000_000
        outcomes[name] = {"kind": kind, "fold": fold, "best_epoch": best["epoch"], "parameters": config["parameters"],
                          "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          "inner": ablation_metrics(inner_outputs[kind, fold], records, config["modalities"]),
                          "outer": ablation_metrics(outer_outputs[kind, fold], records, config["modalities"]),
                          "clean_train": ablation_metrics(clean, records, config["modalities"]),
                          "elapsed_seconds": completion["elapsed_seconds"],
                          "mean_training_epoch_seconds": float(np.mean([row["system/epoch_seconds"] for row in history]))}
    aggregate = {}
    for kind in KINDS:
        predictions = join_predictions([outer_outputs[kind, fold] for fold in [0, 1]])
        assert len(np.unique(predictions["indices"])) == len(predictions["indices"])
        label, guessed = predictions["labels"], predictions["probabilities"].argmax(1)
        body = (label >= 28) & (label <= 36)
        aggregate[kind] = {"outer": ablation_metrics(predictions, records, ["skeleton", "imu", "thermal"]),
                           "whole_body": classification_metrics(label[body], guessed[body]),
                           "other_31": classification_metrics(label[~body], guessed[~body])}
        if kind != "cnn_pooled":
            baseline = join_predictions([outer_outputs["cnn_pooled", fold] for fold in [0, 1]])
            aggregate[kind]["paired_vs_pooled"] = paired_user_bootstrap(baseline, predictions, records)
    selected = {}
    for fold in [0, 1]:
        def rank(kind):
            values = inner_outputs[kind, fold]
            metrics = classification_metrics(values["labels"], values["probabilities"].argmax(1))
            return metrics["accuracy"], metrics["macro_f1"], -KINDS.index(kind)
        selected[str(fold)] = max(KINDS, key=rank)
    nested = join_predictions([outer_outputs[selected[str(fold)], fold] for fold in [0, 1]])
    report = {"status": "complete", "selection": configuration["selection"], "experiments": outcomes,
              "architectures": aggregate, "fold_local_inner_choices": selected,
              "nested_selected_outer": ablation_metrics(nested, records, ["skeleton", "imu", "thermal"]),
              "caveats": [configuration["protocol"], configuration["caveats"],
                          "One seed per architecture: require replication before claiming robust superiority.",
                          "Outer folds cannot tune epochs or select a different architecture after this report."]}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 空间 token 受控实验结果", "", "所有模型从零训练；每折仅根据其内层未见用户选 checkpoint。", "",
             "| 架构 | 折 | 最佳 epoch | 干净训练 | 内层验证 | 外层验证 | 外层 macro-F1 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for outcome in outcomes.values():
        score = lambda name: outcome[name]["all_validation"]["accuracy"] * 100
        lines.append(f"| {outcome['kind']} | {outcome['fold']} | {outcome['best_epoch']} | {score('clean_train'):.2f}% | {score('inner'):.2f}% | {score('outer'):.2f}% | {outcome['outer']['all_validation']['macro_f1'] * 100:.2f}% |")
    lines += ["", "## 两折外层汇总（仅作固定模型比较）", ""]
    for kind, value in aggregate.items():
        lines.append(f"- {kind}: accuracy {value['outer']['all_validation']['accuracy'] * 100:.2f}%，macro-F1 {value['outer']['all_validation']['macro_f1'] * 100:.2f}%。")
        if "paired_vs_pooled" in value:
            paired = value["paired_vs_pooled"]
            lower, upper = np.asarray(paired["user_cluster_bootstrap_95_interval"]) * 100
            lines.append(f"  与池化对照差值 {paired['accuracy_delta'] * 100:+.2f} 个百分点；探索性用户聚类 bootstrap 95% 区间 [{lower:+.2f}, {upper:+.2f}]。")
    lines += ["", "## 按折内层选型", "", f"- 选择：{selected}",
              f"- 按上述选择组合外层预测：{report['nested_selected_outer']['all_validation']['accuracy'] * 100:.2f}%。", "",
              "## 限制", "", "- 外层用户此前已用于历史研究，不是全新测试集。", "- 内外层都按用户隔离，训练量小于之前全训练折模型，不能直接拿旧 56.96% 当本轮对照。",
              "- 不同折之间存在用户复用，因此不能汇总两折内层分数选统一赢家，再把该赢家的外层分数声称为无偏评估。",
              "- 仅一个训练种子；bootstrap 只反映固定模型下用户抽样，不包含重新训练波动。",
              "- CNN 池化与保留 token 两组参数、模块相同，差别是提前 2×2 池化；patch 组通过增加 FFN 宽度匹配总参数，因此不是只替换输入层的严格单因素比较。",
              "- 官方测试集未提交；本报告不声称官方成绩提升。", ""]
    args.markdown.write_text("\n".join(lines))
    print("Fold-local choices", selected)
    for kind, value in aggregate.items():
        print(kind, value["outer"]["all_validation"]["accuracy"], value.get("paired_vs_pooled", {}))


if __name__ == "__main__":
    main()
