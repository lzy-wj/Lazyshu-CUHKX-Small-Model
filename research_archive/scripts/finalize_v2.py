import csv
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Copy mismatch: {target}")
    return {"path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size, "sha256": expected}


def main():
    selected = json.loads((ROOT / "configs/v2_selected_ensemble.json").read_text())
    report = json.loads((ROOT / "outputs/predictions/v2_ensemble_validation_report.json").read_text())
    original = json.loads((ROOT / "outputs/predictions/ensemble_validation_report.json").read_text())
    prediction = json.loads((ROOT / "outputs/predictions/research_multimodal_v2_manifest.json").read_text())
    bundle = ROOT / "outputs/final_v2"
    bundle.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for model in prediction["models"]:
        if model["config"]["fold"] != -1:
            raise RuntimeError("Final model is not full-data training")
        source = Path(model["checkpoint"])
        artifacts.append(copy_verified(source, bundle / "models" / f"{model['config']['name']}.pt"))
    for name, model in prediction.get("preprocessing_models", {}).items():
        artifacts.append(copy_verified(Path(model["path"]), bundle / "models" / f"{name}.pth"))
    total_bytes = sum(artifact["bytes"] for artifact in artifacts)
    if total_bytes > 100_000_000 or total_bytes != prediction["checkpoint_bytes_total"]:
        raise RuntimeError("Incorrect full inference model size accounting")
    source = ROOT / "outputs/submissions/research_multimodal_v2.csv"
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    with (ROOT / "outputs/final/research_multimodal.csv").open() as handle:
        previous = list(csv.DictReader(handle))
    if len(rows) != 405 or [row["path"] for row in rows] != [row["path"] for row in previous]:
        raise RuntimeError("Submission sample identity mismatch")
    artifacts.append(copy_verified(source, bundle / source.name))
    copy_verified(ROOT / "configs/v2_selected_ensemble.json", bundle / "ensemble.json")
    audit = {"artifacts": artifacts, "total_inference_model_bytes": total_bytes,
             "rows": 405, "test_accuracy": None, "target_accuracy": 0.95, "target_achieved": False,
             "changed_predictions_from_v1": sum(left["prediction"] != right["prediction"] for left, right in zip(rows, previous))}
    (bundle / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    comparisons = []
    paired = {}
    for fold in [0, 1]:
        current = np.load(ROOT / f"outputs/predictions/v2_ensemble_fold{fold}.npz")
        old = np.load(ROOT / f"outputs/predictions/ensemble_fold{fold}.npz")
        if not np.array_equal(current["indices"], old["indices"]):
            raise RuntimeError("Cannot compare different validation samples")
        new_correct = current["probabilities"].argmax(1) == current["labels"]
        old_correct = old["probabilities"].argmax(1) == old["labels"]
        paired[f"fold{fold}"] = {"fixed_errors": int((new_correct & ~old_correct).sum()),
                                  "new_errors": int((~new_correct & old_correct).sum())}
        comparisons.append(f"| Fold {fold} | {original[f'fold{fold}']['accuracy']:.2%} | {report[f'fold{fold}']['accuracy']:.2%} | {len(new_correct)} |")
    audit["paired_comparison"] = paired
    class_names = {int(path.name.split("_", 1)[0]): path.name.split("_", 1)[1]
                   for path in (ROOT / "data/raw/HAR/data/Skeleton").iterdir() if path.is_dir()}
    confusion = sum(np.asarray(report[f"fold{fold}"]["confusion_matrix"]) for fold in [0, 1])
    class_rows = []
    for label in range(40):
        support = int(confusion[label].sum())
        class_rows.append({"label": label, "action": class_names[label], "support": support,
                           "correct": int(confusion[label, label]),
                           "recall": float(confusion[label, label] / support) if support else None})
    with (bundle / "validation_classes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    confusion_pairs = sorted([(int(confusion[actual, predicted]), actual, predicted)
                              for actual in range(40) for predicted in range(40)
                              if actual != predicted and confusion[actual, predicted] > 0], reverse=True)[:10]
    confusion_rows = [f"| {class_names[actual]} | {class_names[predicted]} | {count} |"
                      for count, actual, predicted in confusion_pairs]
    copy_verified(ROOT / "outputs/predictions/v2_data_audit.json", bundle / "data_audit.json")
    (bundle / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    experiments = []
    for path in sorted((ROOT / "outputs/checkpoints").glob("v2_*/metrics.json")):
        values = json.loads(path.read_text())
        experiments.append(f"| {path.parent.name} | {values['accuracy']:.2%} | {values['epoch']} |")
    checkpoint_arguments = " ".join(f"outputs/final_v2/models/{name}.pt" for name in selected["final_models"])
    weights = " ".join(map(str, selected["weights"]))
    temperatures = " ".join(map(str, selected["temperatures"]))
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    text = f"""# 第二轮建模结果

记录时间：{now}

## 目标状态

**没有达到约 95% 的目标。** 本轮完成了建模改进、用户隔离验证、全量重训与 405 条测试预测，但没有测试标签或新的 Kaggle 分数，不能声称测试准确率提高到某个数值。

| 验证划分 | 第一轮 | 第二轮 | 片段数 |
| --- | --- | --- | --- |
{chr(10).join(comparisons)}

第二轮两组按样本量加权：{report['weighted_accuracy']:.2%}；第一轮：{original['weighted_accuracy']:.2%}。
各组均为用户隔离，两组各覆盖 39 类，合起来覆盖全部 40 类。
结构、融合权重和温度在 Fold 0 上选择，固定后应用 Fold 1；每组模型 checkpoint 仍按本组验证选择，因此不是无偏的最终测试估计。
Fold 0 修正旧错误 {paired['fold0']['fixed_errors']} 个、引入新错误 {paired['fold0']['new_errors']} 个；Fold 1 对应 {paired['fold1']['fixed_errors']} / {paired['fold1']['new_errors']} 个。

## 改进内容

- 对 IR 作对比度增强；用 SSDLite-MobileNetV3 的人体框分别裁剪 IR、Thermal，并把 IR 框应用到 Depth。
- 人物区域保留 224×224 分辨率，IR、Thermal、Depth 独立视觉编码，结合骨架与 IMU。
- 将 IMU 加速度与角速度按四元数旋转到世界坐标，减少佩戴方向变化；对比统计时序编码和卷积编码。
- 冻结增强视觉特征没有产生收益，未默认用于最终方案；BatchNorm 重估也没有带来收益。
- 所有预处理均不使用测试活动标签。检测器、外部初始化的合规性仍需主办方确认。

## 交付

- 新预测：`outputs/final_v2/research_multimodal_v2.csv`，405 行，未自动提交。
- 第一轮文件原样保留在 `outputs/final/`，没有覆盖。
- 分类模型与检测器全部计入体积：**{total_bytes / 1e6:.2f} MB**，低于 100 MB。
- 模型哈希、大小、相对路径：`outputs/final_v2/manifest.json`。
- 概率：`outputs/predictions/research_multimodal_v2.npz`。
- 相比第一轮，405 个测试预测中有 {audit['changed_predictions_from_v1']} 个变化；变化本身不代表这些预测正确。
- SwanLab 私有项目 `shu_123/cuhkx-small-multimodal`，本轮实验均为 `v2_` 前缀。

融合模型顺序：{selected['final_models']}；权重：{selected['weights']}；温度：{selected['temperatures']}。
最终模型都使用全部 3036 个训练片段，依据验证最优 epoch 折算全量训练更新次数，具体命令在 `configs/v2_final_*.json`。

## 复现

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n Train python scripts/prepare_imu_world.py
CUDA_VISIBLE_DEVICES=2 conda run -n Train python scripts/prepare_person_crops.py --stage detect
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n Train python scripts/prepare_person_crops.py --stage crop
CUDA_VISIBLE_DEVICES=2 conda run -n Train python scripts/predict.py --checkpoints {checkpoint_arguments} --weights {weights} --temperatures {temperatures} --name research_multimodal_v2_recheck
```

检测器原始权重缓存位于 `.tools/torch/hub/checkpoints/`，副本也在本轮模型包。未从缓存直接假装省掉检测器体积。
CPU/GPU 并行执行不保证跨环境逐位复现；项目内会使用打包权重再次推理检查。

## 本轮实验

| 实验 | 本地 Accuracy | 最佳 Epoch |
| --- | --- | --- |
{chr(10).join(experiments)}

## 错误诊断

以下统计仅来自两组本地验证，共 1357 个片段，不涉及测试标签。每类支持数和召回率见 `outputs/final_v2/validation_classes.csv`。

| 真实动作 | 最常见错误预测之一 | 次数 |
| --- | --- | --- |
{chr(10).join(confusion_rows)}

缓存形状、数值有限性、用户隔离及旋转算术检查通过，记录在 `outputs/final_v2/data_audit.json`。

## 下一步需要的证据

要判断线上差距，需要具体 Kaggle 分数及其对应 CSV。当前能验证的是本地用户隔离成绩，不会以训练准确率或随机帧划分成绩代替 95% 目标。
细粒度手部动作、物体识别、稀有类别与跨用户分布偏移是后续待验证的改进方向，不能仅靠增加空闲 GPU 就承诺从当前水平达到 95%。
官方 `no LLMs for development` 限制仍未澄清；这套 AI 辅助研究实现不宣称参赛合规，也不会代为提交。
"""
    (ROOT / "docs/RESULTS_V2.md").write_text(text)
    (bundle / "RESULTS.md").write_text(text)
    import swanlab
    swanlab.init(project="cuhkx-small-multimodal", name="v2-final-summary", public=False,
                 log_dir=str(ROOT / "outputs/swanlog"), config=selected)
    swanlab.log({"validation/fold0": report["fold0"]["accuracy"], "validation/fold1": report["fold1"]["accuracy"],
                 "validation/weighted_accuracy": report["weighted_accuracy"], "target/accuracy": 0.95,
                 "artifacts/inference_model_mb": total_bytes / 1e6, "artifacts/test_rows": 405})
    swanlab.finish()
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
