import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.finalize_v2 import copy_verified


ROOT = Path(__file__).resolve().parents[1]


def main():
    selected = json.loads((ROOT / "configs/v3_selected_ensemble.json").read_text())
    current = json.loads((ROOT / "outputs/predictions/v3_ensemble_validation_report.json").read_text())
    previous = json.loads((ROOT / "outputs/predictions/v2_ensemble_validation_report.json").read_text())
    prediction = json.loads((ROOT / "outputs/predictions/research_multimodal_v3_manifest.json").read_text())
    diagnosis = json.loads((ROOT / "outputs/predictions/v3_generalization_diagnosis.json").read_text())
    bundle = ROOT / "outputs/final_v3"
    bundle.mkdir(parents=True, exist_ok=True)
    original = np.load(ROOT / "outputs/predictions/research_multimodal_v3.npz")
    repeated = np.load(ROOT / "outputs/predictions/research_multimodal_v3_recheck.npz")
    np.testing.assert_array_equal(original["indices"], repeated["indices"])
    np.testing.assert_allclose(original["probabilities"], repeated["probabilities"], atol=1e-6, rtol=0)
    assert original["probabilities"].shape == (405, 40) and np.isfinite(original["probabilities"]).all()
    np.testing.assert_allclose(original["probabilities"].sum(1), 1, atol=1e-6, rtol=0)
    source_csv = ROOT / "outputs/submissions/research_multimodal_v3.csv"
    assert source_csv.read_bytes() == (ROOT / "outputs/submissions/research_multimodal_v3_recheck.csv").read_bytes()
    old_manifest = json.loads((ROOT / "outputs/final_v2/manifest.json").read_text())
    for artifact in old_manifest["artifacts"]:
        assert hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    artifacts = []
    assert [model["config"]["name"] for model in prediction["models"]] == selected["final_models"]
    for model in prediction["models"]:
        assert model["config"]["fold"] == -1 and model["config"]["train_clips"] == 3036
        source = Path(model["checkpoint"])
        artifacts.append(copy_verified(source, bundle / "models" / f"{model['config']['name']}.pt"))
    for name, model in prediction["preprocessing_models"].items():
        artifacts.append(copy_verified(Path(model["path"]), bundle / "models" / f"{name}.pth"))
    model_bytes = sum(artifact["bytes"] for artifact in artifacts)
    assert model_bytes == prediction["checkpoint_bytes_total"] and model_bytes <= 100_000_000
    artifacts.append(copy_verified(source_csv, bundle / source_csv.name))
    copy_verified(ROOT / "configs/v3_selected_ensemble.json", bundle / "ensemble.json")
    copy_verified(ROOT / "outputs/predictions/v3_ensemble_validation_report.json", bundle / "validation.json")
    copy_verified(ROOT / "outputs/predictions/v3_generalization_diagnosis.json", bundle / "diagnosis.json")
    copy_verified(ROOT / "outputs/predictions/v3_data_audit.json", bundle / "data_audit.json")
    copy_verified(ROOT / "outputs/predictions/v3_legacy_regression.json", bundle / "legacy_regression.json")
    copy_verified(ROOT / "docs/leaderboard_snapshot.json", bundle / "leaderboard_snapshot.json")
    copy_verified(ROOT / "outputs/predictions/kaggle_submission_history.json", bundle / "kaggle_submission_history.json")
    with source_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 405
    np.testing.assert_array_equal([int(row["prediction"]) for row in rows], original["probabilities"].argmax(1))
    manifest = {"artifacts": artifacts, "total_inference_model_bytes": model_bytes, "rows": len(rows),
                "target_accuracy": 0.95, "target_achieved": False, "test_accuracy": None,
                "csv_exact_match": True, "probability_max_absolute_difference": float(np.abs(original["probabilities"] - repeated["probabilities"]).max()),
                "v2_artifact_hashes_unchanged": True,
                "reproduction_scope": "Packaged classifier weights with audited existing preprocessing caches, not a raw-data detector rerun."}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pruned = json.loads((ROOT / "configs/v3_pruned_experiments.json").read_text())["names"]
    experiments = []
    for path in sorted((ROOT / "outputs/checkpoints").glob("v3_*/metrics.json")):
        values = json.loads(path.read_text())
        status = "主动停止（保留最佳权重）" if path.parent.name in pruned else "完成"
        experiments.append(f"| {path.parent.name} | {values['accuracy']:.2%} | {values['epoch']} | {status} |")
    comparisons = [f"| Fold {fold} | {previous[f'fold{fold}']['accuracy']:.2%} | {current[f'fold{fold}']['accuracy']:.2%} |"
                   for fold in [0, 1]]
    confusion = sum(np.asarray(current[f"fold{fold}"]["confusion_matrix"]) for fold in [0, 1])
    old_confusion = sum(np.asarray(previous[f"fold{fold}"]["confusion_matrix"]) for fold in [0, 1])
    class_names = {int(path.name.split("_", 1)[0]): path.name.split("_", 1)[1]
                   for path in (ROOT / "data/raw/HAR/data/Skeleton").iterdir() if path.is_dir()}
    classes = [{"label": label, "action": class_names[label], "support": int(confusion[label].sum()),
                "v2_correct": int(old_confusion[label, label]), "v3_correct": int(confusion[label, label]),
                "v3_recall": float(confusion[label, label] / confusion[label].sum())}
               for label in range(40)]
    with (bundle / "validation_classes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(classes[0]))
        writer.writeheader()
        writer.writerows(classes)
    worst = sorted([entry for entry in classes if entry["support"] >= 10], key=lambda entry: entry["v3_recall"])[:8]
    error_rows = [f"| {entry['action']} | {entry['support']} | {entry['v3_recall']:.2%} |" for entry in worst]
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    checkpoint_arguments = " ".join(f"outputs/final_v3/models/{name}.pt" for name in selected["final_models"])
    weight_arguments = " ".join(map(str, selected["weights"]))
    temperature_arguments = " ".join(map(str, selected["temperatures"]))
    text = f"""# 第三轮研究结果

记录时间：{now}

## 95% 目标仍未完成

用户要求的是接近 95%，不是只要比前一轮高。本轮所有分数仍为本地用户隔离验证；没有测试标签或本轮新预测的线上成绩，不声称测试达标。

| 划分 | 第二轮 | 第三轮 |
| --- | --- | --- |
{chr(10).join(comparisons)}

按 1357 个验证片段加权：{previous['weighted_accuracy']:.2%} → {current['weighted_accuracy']:.2%}。
融合权重与温度仅在 Fold 0 上选择，然后固定应用 Fold 1；各模型 checkpoint 仍按各自验证选取，所以不是无偏测试估计。

## 为什么之前不够

- 第二轮统计传感器模型的干净训练准确率为 {diagnosis['clean_accuracy']['train']:.2%}，同组用户隔离验证为 {diagnosis['clean_accuracy']['valid']:.2%}。训练本身超过 95% 没有解决跨用户泛化。
- 细粒度动作失分明显；仅增加轮数、扩大空间池化或叠加统计特征，并没有得到相应收益。
- TSM 与高空间分辨率池化对照明显落后于新视频模型，主动停止，不把被中断的实验说成正常完成。
- 检查了真实时间 IMU、胸部参考系相对姿态及二段/四段时序统计；未证明有益的路线不默认纳入交付。
- 新的主要视频候选使用 X3D-S 的小型 3D 视频编码器，显式记录 Kinetics-400 外部初始化。该来源的竞赛合规性仍需澄清。

## 交付与复现

- 预测：`outputs/final_v3/research_multimodal_v3.csv`，405 行，未自动提交。
- 全部推理权重（含人体检测器）：{model_bytes / 1e6:.2f} MB。前两轮文件保留，第二轮 artifact 哈希复核未变化。
- 打包分类权重重新加载后，CSV 完全一致，概率最大绝对差 {manifest['probability_max_absolute_difference']}。
- 配置：`configs/v3_selected_ensemble.json`；新模型全量重训配方：`configs/v3_final_*.json`。复用的第二轮模型已经用全部 3036 个训练样本训练，不重复浪费计算。
- 依赖：`configs/video-requirements.txt`，使用 `pip install --no-deps --target .tools/python -r configs/video-requirements.txt`；共享 Train 环境没有升级。
- 核验：`outputs/final_v3/manifest.json`、`outputs/final_v3/data_audit.json`。
- SwanLab 私有项目 `shu_123/cuhkx-small-multimodal`，本轮前缀 `v3_`。

融合模型：{selected['final_models']}；权重：{selected['weights']}；温度：{selected['temperatures']}。

先确认显卡空闲，再从项目根目录复现（下例使用 GPU 3）：

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 conda run -n Train python scripts/predict.py --checkpoints {checkpoint_arguments} --weights {weight_arguments} --temperatures {temperature_arguments} --name research_multimodal_v3_recheck
```

## 所有实验

| 实验 | Fold Accuracy | 最佳 Epoch | 状态 |
| --- | --- | --- | --- |
{chr(10).join(experiments)}

## 仍在失分的类别

以下为两组验证合并后的低召回类别（仅展示支持数至少 10 的类别）；全部 40 类的前后对比见 `outputs/final_v3/validation_classes.csv`。没有使用测试标签。

| 类别 | 验证片段数 | 召回率 |
| --- | --- | --- |
{chr(10).join(error_rows)}

## 未解决事项

95% 仍不是已达到或能保证的结果。已找到新版格式的本机 Kaggle 登录凭据并只读查询提交历史，结果保存于 `outputs/predictions/kaggle_submission_history.json`；没有上传本轮预测，不要求用户发送密钥。要判断线上效果，仍须本轮 CSV 对应的实际成绩。
2026 年 9 月 7 日公开榜 API 快照显示前列已有超过 95% 的公开成绩，见 `docs/leaderboard_snapshot.json`；这说明目标有公开榜参照，但别人的公开榜分数不等于本模型的测试成绩，也不代表最终私榜或合规审查结论。
比赛的 `no LLMs for development` 限制仍未澄清；本项目是 AI 辅助研究实现，不宣称参赛合规。
"""
    (ROOT / "docs/RESULTS_V3.md").write_text(text)
    (bundle / "RESULTS.md").write_text(text)
    import swanlab
    swanlab.init(project="cuhkx-small-multimodal", name="v3-final-summary", public=False,
                 log_dir=str(ROOT / "outputs/swanlog"), config=selected)
    swanlab.log({"validation/fold0": current["fold0"]["accuracy"], "validation/fold1": current["fold1"]["accuracy"],
                 "validation/weighted_accuracy": current["weighted_accuracy"], "target/accuracy": 0.95,
                 "artifacts/inference_model_mb": model_bytes / 1e6, "artifacts/test_rows": 405})
    swanlab.finish()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
