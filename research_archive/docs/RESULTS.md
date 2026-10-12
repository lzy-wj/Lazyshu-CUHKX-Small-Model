# 训练与最终推理结果

生成时间：2026-09-06T12:57:31+08:00

## 结论

完整的数据预处理、跨用户训练验证、全量重训、SwanLab 记录及 405 条测试预测已完成。
这是一套可复现的研究基线，**不能据此承诺较高名次或获奖**。验证表现仍有明显提升空间。

- Fold 0：54.60%，652 个 clip，用户 3/8/18/23。
- Fold 1：53.33%，705 个 clip，用户 5/7/17/22。
- 两组按样本量加权准确率：53.94%；这不是 Kaggle 成绩。
- 融合权重及温度在 Fold 0 上选择，原样用于 Fold 1；各模型 checkpoint 仍按各自验证集选择，因此 Fold 1 也不是完全锁定的最终测试集。
- 第二组的单个六模态 MobileNet 模型为 53.76%，融合没有在该组继续提高准确率；不能把第一组的融合提升说成稳定收益。保留冻结的权重，不再针对第二组反复调权。
- 两组各缺少一个稀有类别（分别 25、26），合起来覆盖 40 类。详情在 `data/metadata/validation_protocol.json`。
- 没有测试真值，不报告测试准确率，也没有自动提交 Kaggle。

## 最终产物

- 主预测：`outputs/final/research_multimodal.csv`。
- 无外部预训练权重的六模态对照：`outputs/final/scratch_multimodal.csv`，其 Fold 0 单模型参考准确率为 47.39%，不是测试分数。
- 模型：`outputs/final/models/`，选定模型合计 42.13 MB，低于十进制 100 MB。
- 概率及分模型输出：`outputs/predictions/research_multimodal.npz`。
- 模型哈希与配置：`outputs/final/manifest.json`。
- 验证混淆矩阵、逐用户指标：`outputs/predictions/ensemble_validation_report.json`。
- SwanLab 私有项目：`shu_123/cuhkx-small-multimodal`。

主模型融合权重：[0.2, 0.1, 0.7]；温度：[1.228330254512121, 1.2643666836663014, 1.1982447831783827]；顺序：['final_mobile', 'final_pose', 'final_efficient']。
主方案包含骨架、IMU、雷达、深度、IR、Thermal 六模态，其中深度与 IR 共享视觉分支。模态缺失通过 mask 显式处理。

## 全量重训

所有最终模型只使用提供的 3036 个训练 clip，不使用测试标签，也不对测试样本作人工标注。
MobileNet 与 EfficientNet 的全量轮数按 Fold 0 最优 epoch 的优化器更新次数折算，精确命令记录在 `configs/final_mobile.json` 和 `configs/final_efficient.json`。
骨架模型使用两组验证的最佳轮数平均更新预算，折算为全量 67 轮；无外部预训练的六模态对照为全量 31 轮。
固定随机种子，但 GPU 算子及多线程执行不保证逐位重现。

## 再次生成主预测

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n Train python scripts/predict.py --checkpoints outputs/final/models/final_mobile.pt outputs/final/models/final_pose.pt outputs/final/models/final_efficient.pt --weights 0.2 0.1 0.7 --temperatures 1.228330254512121 1.2643666836663014 1.1982447831783827 --name research_multimodal
```

验证已覆盖：下载哈希、7-Zip 解压 CRC、分用户隔离、数据有限性、模型输出形状、缺失模态、405 行路径顺序/唯一性、类别范围、模型体积及最终模型哈希。
79 个抽样图像不可解码，使用同模态邻近有效抽样帧补齐，整模态无有效抽样帧则关闭 mask；没有修改原始数据。

## 实验记录

| 实验 | Fold | 最优 Accuracy | Balanced Accuracy | Epoch |
| --- | --- | --- | --- | --- |
| canonical_pose_f0 | 0 | 49.39% | 45.39% | 112 |
| canonical_pose_f1 | 1 | 51.91% | 48.25% | 60 |
| efficient_f0 | 0 | 50.77% | 46.07% | 20 |
| efficient_f1 | 1 | 51.35% | 47.45% | 38 |
| geometry_f0 | 0 | 48.01% | 41.78% | 58 |
| imu_f0 | 0 | 23.47% | 21.11% | 18 |
| mobile_f0 | 0 | 50.92% | 47.70% | 50 |
| mobile_f1 | 1 | 53.76% | 49.86% | 24 |
| mobile_vision_f0 | 0 | 40.49% | 36.11% | 34 |
| multimodal_f0 | 0 | 47.39% | 44.10% | 40 |
| pose_imu_f0 | 0 | 45.25% | 38.97% | 90 |
| sensor_f0 | 0 | 46.17% | 40.46% | 32 |
| skeleton_f0 | 0 | 44.48% | 39.17% | 34 |
| stgcn_f0 | 0 | 48.77% | 45.06% | 48 |
| vision_f0 | 0 | 37.58% | 32.70% | 80 |

## 规则与限制

官方规则在 2026-09-06 仍禁止 `LLMs for development`。本实现由 AI 辅助开发，不能宣称参赛合规；是否可提交需要先取得主办方明确许可。
主方案使用公开的小型 ImageNet 初始化（torchvision MobileNetV3-Small / EfficientNet-B0），是否允许该初始化也需确认。对照方案不使用外部预训练权重，但这不消除 AI 开发规则问题。
主要误差来自跨用户分布偏移、相近动作、稀有类别和不完整传感器。没有通过随机帧切分或识别采集日期来制造虚高分数。
