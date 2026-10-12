import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from finalize_v2 import copy_verified


ROOT = Path(__file__).resolve().parents[1]


def main():
    selected = json.loads((ROOT / "configs/v4_selected_ensemble.json").read_text())
    current = json.loads((ROOT / "outputs/predictions/v4_ensemble_validation_report.json").read_text())
    previous = json.loads((ROOT / "outputs/predictions/v3_ensemble_validation_report.json").read_text())
    prediction = json.loads((ROOT / "outputs/predictions/research_multimodal_v4_manifest.json").read_text())
    original = np.load(ROOT / "outputs/predictions/research_multimodal_v4.npz")
    repeated = np.load(ROOT / "outputs/predictions/research_multimodal_v4_recheck.npz")
    np.testing.assert_array_equal(original["indices"], repeated["indices"])
    np.testing.assert_allclose(original["probabilities"], repeated["probabilities"], atol=1e-6, rtol=0)
    assert original["probabilities"].shape == (405, 40) and np.isfinite(original["probabilities"]).all()
    np.testing.assert_allclose(original["probabilities"].sum(1), 1, atol=1e-6, rtol=0)
    source_csv = ROOT / "outputs/submissions/research_multimodal_v4.csv"
    assert source_csv.read_bytes() == (ROOT / "outputs/submissions/research_multimodal_v4_recheck.csv").read_bytes()
    with source_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 405 and len({row["path"] for row in rows}) == 405
    np.testing.assert_array_equal([int(row["prediction"]) for row in rows], original["probabilities"].argmax(1))
    old_manifest = json.loads((ROOT / "outputs/final_v3/manifest.json").read_text())
    for artifact in old_manifest["artifacts"]:
        assert hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    legacy = np.load(ROOT / "outputs/predictions/research_multimodal_v3.npz")
    legacy_repeated = np.load(ROOT / "outputs/predictions/research_multimodal_v3_v4_regression.npz")
    np.testing.assert_array_equal(legacy["indices"], legacy_repeated["indices"])
    np.testing.assert_allclose(legacy["probabilities"], legacy_repeated["probabilities"], atol=1e-6, rtol=0)
    assert (ROOT / "outputs/submissions/research_multimodal_v3.csv").read_bytes() == (ROOT / "outputs/submissions/research_multimodal_v3_v4_regression.csv").read_bytes()
    bundle = ROOT / "outputs/final_v4"
    artifacts = []
    assert [model["config"]["name"] for model in prediction["models"]] == selected["final_models"]
    for model in prediction["models"]:
        assert model["config"]["fold"] == -1 and model["config"]["train_clips"] == 3036
        artifacts.append(copy_verified(Path(model["checkpoint"]), bundle / "models" / f"{model['config']['name']}.pt"))
    for name, model in prediction["preprocessing_models"].items():
        artifacts.append(copy_verified(Path(model["path"]), bundle / "models" / f"{name}.pth"))
    model_bytes = sum(artifact["bytes"] for artifact in artifacts)
    assert model_bytes == prediction["checkpoint_bytes_total"] and model_bytes <= 100_000_000
    artifacts.append(copy_verified(source_csv, bundle / source_csv.name))
    for source, target in [
        ("configs/v4_selected_ensemble.json", "ensemble.json"),
        ("configs/v4_protocol.json", "protocol.json"),
        ("outputs/predictions/v4_ensemble_validation_report.json", "validation.json"),
        ("outputs/predictions/v4_selection.json", "selection.json"),
        ("outputs/predictions/v4_data_audit.json", "data_audit.json"),
        ("outputs/predictions/v4_input_distribution_audit.json", "input_distribution_audit.json"),
        ("data/metadata/person_detail.json", "preprocessing.json"),
    ]:
        copy_verified(ROOT / source, bundle / target)
    improved = (current["weighted_accuracy"] > previous["weighted_accuracy"]
                and current["fold1"]["accuracy"] >= previous["fold1"]["accuracy"])
    manifest = {"artifacts": artifacts, "total_inference_model_bytes": model_bytes, "rows": len(rows),
                "target_accuracy": 0.95, "target_achieved": False, "test_accuracy": None,
                "csv_exact_match": True, "probability_max_absolute_difference": float(np.abs(original["probabilities"] - repeated["probabilities"]).max()),
                "v3_artifact_hashes_unchanged": True, "improved_over_v3_on_validation": improved,
                "v3_prediction_csv_unchanged": True,
                "v3_probability_max_absolute_difference": float(np.abs(legacy["probabilities"] - legacy_repeated["probabilities"]).max()),
                "recommended_research_bundle": "outputs/final_v4" if improved else "outputs/final_v3",
                "reproduction_scope": "Packaged classifier weights with audited existing preprocessing caches, not an end-to-end raw detector rerun."}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    confusion = sum(np.asarray(current[f"fold{fold}"]["confusion_matrix"]) for fold in [0, 1])
    old_confusion = sum(np.asarray(previous[f"fold{fold}"]["confusion_matrix"]) for fold in [0, 1])
    class_names = {int(path.name.split("_", 1)[0]): path.name.split("_", 1)[1]
                   for path in (ROOT / "data/raw/HAR/data/Skeleton").iterdir() if path.is_dir()}
    classes = [{"label": label, "action": class_names[label], "support": int(confusion[label].sum()),
                "v3_correct": int(old_confusion[label, label]), "v4_correct": int(confusion[label, label]),
                "v4_recall": float(confusion[label, label] / max(1, confusion[label].sum()))} for label in range(40)]
    with (bundle / "validation_classes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(classes[0]))
        writer.writeheader()
        writer.writerows(classes)
    experiments = []
    for path in sorted((ROOT / "outputs/checkpoints").glob("v4_*/metrics.json")):
        values = json.loads(path.read_text())
        exit_path = ROOT / f"outputs/logs/{path.parent.name}.exit"
        status = "完成" if exit_path.exists() and exit_path.read_text().strip() == "0" else "未正常结束"
        experiments.append(f"| {path.parent.name} | {values['accuracy']:.2%} | {values['epoch']} | {status} |")
    comparisons = [f"| Fold {fold} | {previous[f'fold{fold}']['accuracy']:.2%} | {current[f'fold{fold}']['accuracy']:.2%} |"
                   for fold in [0, 1]]
    text = f"""# 第四轮研究结果

记录时间：{datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')}

## 真实结果

本轮跨用户验证加权准确率：{current['weighted_accuracy']:.2%}，上一版 {previous['weighted_accuracy']:.2%}。
**95% 尚未达到；官方测试集准确率未知。本轮 CSV 未上传 Kaggle。**

| 用户隔离验证 | V3 | V4 |
| --- | --- | --- |
{chr(10).join(comparisons)}

推荐研究包：`{manifest['recommended_research_bundle']}`。
选择只使用 Fold 0：新视频模型温度、预先固定的 21 点双模型权重搜索，以及固定 0.15 的旧统计专家。未拟合类别特定权重。配置冻结后原样应用 Fold 1。
为节省时间，局部视频 Fold 1 与 Fold 0 并行训练，但冻结前不查看 Fold 1 的指标；各 fold 仍在本 fold 验证集选择 checkpoint，因此不是无偏最终测试估计。是否推荐新包还使用 Fold 1 的不退化门槛。

## 改动与取舍

- 从原始 IR/Thermal 图片裁剪人体框上部 72%，不是放大旧低分辨率缓存。检测失败时保留全框；同一固定变换用于所有划分。
- 测试关节位置编码、空间注意力和时序卷积骨架模型。Fold 0 只有 50.46%，低于旧统计专家 53.07%，不纳入融合。
- 测试全景视频与骨架/几何/IMU 统计编码器联合训练；是否采用取决于固定验证结果。
- 缓存、缺失掩码、关节模型梯度、旧版 artifact 哈希与图像输入格式均已检查。

## 交付

- `outputs/final_v4/research_multimodal_v4.csv`：405 行。
- 分类器与人体检测器合计 {model_bytes / 1e6:.2f} MB，保持小于 100 MB。
- 打包后重新加载推理：CSV 完全一致，概率最大差 {manifest['probability_max_absolute_difference']}。
- 模型：{selected['final_models']}；权重：{selected['weights']}；温度：{selected['temperatures']}。
- 全量模型使用全部 3036 个训练片段；新模型训练更新数按 Fold 0 最佳 epoch 缩放。已有全量权重直接复用。
- SwanLab 私有项目 `shu_123/cuhkx-small-multimodal`，本轮前缀 `v4_`。
- V1/V2/V3 保留。完整配置、训练配方、验证类别对比见 `configs/v4_*` 与 `outputs/final_v4/`。

## 本轮实验

| 实验 | Fold Accuracy | 最佳 Epoch | 状态 |
| --- | --- | --- | --- |
{chr(10).join(experiments)}

## 尚未解决

比赛的 `no LLMs for development` 条款及预训练来源合规性尚未澄清。本项目为 AI 辅助研究实现，不宣称参赛合规，不自动上传预测。
本机已有新版 Kaggle 登录凭据，无需用户发送密钥。只有本轮预测对应的真实线上成绩才能说明官方测试表现。
"""
    (ROOT / "docs/RESULTS_V4.md").write_text(text)
    (bundle / "RESULTS.md").write_text(text)
    import swanlab
    swanlab.init(project="cuhkx-small-multimodal", name="v4-final-summary", public=False,
                 log_dir=str(ROOT / "outputs/swanlog"), config=selected)
    swanlab.log({"validation/fold0": current["fold0"]["accuracy"], "validation/fold1": current["fold1"]["accuracy"],
                 "validation/weighted": current["weighted_accuracy"], "artifact/model_bytes": model_bytes,
                 "artifact/csv_rows": len(rows), "target/achieved": 0})
    swanlab.finish()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
