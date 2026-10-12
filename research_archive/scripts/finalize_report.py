import csv
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    selected = json.loads((ROOT / "configs/selected_ensemble.json").read_text())
    validation = json.loads((ROOT / "outputs/predictions/ensemble_validation_report.json").read_text())
    prediction = json.loads((ROOT / "outputs/predictions/research_multimodal_manifest.json").read_text())
    if any(model["config"]["fold"] != -1 for model in prediction["models"]):
        raise RuntimeError("Final model was not trained on the full training set")
    probabilities = np.load(ROOT / "outputs/predictions/research_multimodal.npz")["probabilities"]
    if probabilities.shape != (405, 40) or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise RuntimeError("Invalid final probabilities")
    bundle = ROOT / "outputs/final"
    (bundle / "models").mkdir(parents=True, exist_ok=True)
    for name in ["research_multimodal", "scratch_multimodal"]:
        source = ROOT / "outputs/submissions" / f"{name}.csv"
        with source.open() as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 405:
            raise RuntimeError(f"Invalid row count: {source}")
        shutil.copy2(source, bundle / source.name)
    for model in prediction["models"]:
        source = Path(model["checkpoint"])
        target = bundle / "models" / f"{model['config']['name']}.pt"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != model["sha256"]:
            raise RuntimeError("Bundled model checksum mismatch")
    shutil.copy2(ROOT / "configs/selected_ensemble.json", bundle / "ensemble.json")
    shutil.copy2(ROOT / "outputs/predictions/research_multimodal_manifest.json", bundle / "manifest.json")
    experiment_rows = []
    for path in sorted((ROOT / "outputs/checkpoints").glob("*/metrics.json")):
        if path.parent.name.startswith("smoke"):
            continue
        values = json.loads(path.read_text())
        config = json.loads((path.parent / "config.json").read_text())
        experiment_rows.append(f"| {path.parent.name} | {config['fold']} | {values['accuracy']:.2%} | {values['balanced_accuracy']:.2%} | {values['epoch']} |")
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    report = f"""# 训练与最终推理结果

生成时间：{now}

## 结论

完整的数据预处理、跨用户训练验证、全量重训、SwanLab 记录及 405 条测试预测已完成。
这是一套可复现的研究基线，**不能据此承诺较高名次或获奖**。验证表现仍有明显提升空间。

- Fold 0：{validation['fold0']['accuracy']:.2%}，652 个 clip，用户 3/8/18/23。
- Fold 1：{validation['fold1']['accuracy']:.2%}，705 个 clip，用户 5/7/17/22。
- 两组按样本量加权准确率：{validation['weighted_accuracy']:.2%}；这不是 Kaggle 成绩。
- 融合权重及温度在 Fold 0 上选择，原样用于 Fold 1；各模型 checkpoint 仍按各自验证集选择，因此 Fold 1 也不是完全锁定的最终测试集。
- 第二组的单个六模态 MobileNet 模型为 53.76%，融合没有在该组继续提高准确率；不能把第一组的融合提升说成稳定收益。保留冻结的权重，不再针对第二组反复调权。
- 两组各缺少一个稀有类别（分别 25、26），合起来覆盖 40 类。详情在 `data/metadata/validation_protocol.json`。
- 没有测试真值，不报告测试准确率，也没有自动提交 Kaggle。

## 最终产物

- 主预测：`outputs/final/research_multimodal.csv`。
- 无外部预训练权重的六模态对照：`outputs/final/scratch_multimodal.csv`，其 Fold 0 单模型参考准确率为 47.39%，不是测试分数。
- 模型：`outputs/final/models/`，选定模型合计 {prediction['checkpoint_bytes_total'] / 1e6:.2f} MB，低于十进制 100 MB。
- 概率及分模型输出：`outputs/predictions/research_multimodal.npz`。
- 模型哈希与配置：`outputs/final/manifest.json`。
- 验证混淆矩阵、逐用户指标：`outputs/predictions/ensemble_validation_report.json`。
- SwanLab 私有项目：`shu_123/cuhkx-small-multimodal`。

主模型融合权重：{selected['weights']}；温度：{selected['temperatures']}；顺序：{selected['final_models']}。
主方案包含骨架、IMU、雷达、深度、IR、Thermal 六模态，其中深度与 IR 共享视觉分支。模态缺失通过 mask 显式处理。

## 全量重训

所有最终模型只使用提供的 3036 个训练 clip，不使用测试标签，也不对测试样本作人工标注。
MobileNet 与 EfficientNet 的全量轮数按 Fold 0 最优 epoch 的优化器更新次数折算，精确命令记录在 `configs/final_mobile.json` 和 `configs/final_efficient.json`。
骨架模型使用两组验证的最佳轮数平均更新预算，折算为全量 67 轮；无外部预训练的六模态对照为全量 31 轮。
固定随机种子，但 GPU 算子及多线程执行不保证逐位重现。

## 再次生成主预测

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n Train python scripts/predict.py --checkpoints outputs/final/models/final_mobile.pt outputs/final/models/final_pose.pt outputs/final/models/final_efficient.pt --weights {' '.join(map(str, selected['weights']))} --temperatures {' '.join(map(str, selected['temperatures']))} --name research_multimodal
```

验证已覆盖：下载哈希、7-Zip 解压 CRC、分用户隔离、数据有限性、模型输出形状、缺失模态、405 行路径顺序/唯一性、类别范围、模型体积及最终模型哈希。
79 个抽样图像不可解码，使用同模态邻近有效抽样帧补齐，整模态无有效抽样帧则关闭 mask；没有修改原始数据。

## 实验记录

| 实验 | Fold | 最优 Accuracy | Balanced Accuracy | Epoch |
| --- | --- | --- | --- | --- |
{chr(10).join(experiment_rows)}

## 规则与限制

官方规则在 2026-09-06 仍禁止 `LLMs for development`。本实现由 AI 辅助开发，不能宣称参赛合规；是否可提交需要先取得主办方明确许可。
主方案使用公开的小型 ImageNet 初始化（torchvision MobileNetV3-Small / EfficientNet-B0），是否允许该初始化也需确认。对照方案不使用外部预训练权重，但这不消除 AI 开发规则问题。
主要误差来自跨用户分布偏移、相近动作、稀有类别和不完整传感器。没有通过随机帧切分或识别采集日期来制造虚高分数。
"""
    (ROOT / "docs/RESULTS.md").write_text(report)
    (bundle / "RESULTS.md").write_text(report)
    import swanlab
    swanlab.init(project="cuhkx-small-multimodal", name="final-summary", public=False,
                 log_dir=str(ROOT / "outputs/swanlog"), config=selected)
    swanlab.log({"validation/fold0_accuracy": validation["fold0"]["accuracy"],
                 "validation/fold1_accuracy": validation["fold1"]["accuracy"],
                 "validation/weighted_accuracy": validation["weighted_accuracy"],
                 "artifacts/model_mb": prediction["checkpoint_bytes_total"] / 1e6, "artifacts/test_rows": 405})
    swanlab.finish()
    print(report)


if __name__ == "__main__":
    main()
