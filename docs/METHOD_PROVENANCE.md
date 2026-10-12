# 方法来源与原创边界 / Method Provenance

## 1. 官方资源

| 资源 | 使用方式 | 是否属于本项目原创 |
|---|---|---|
| CUHK-X 数据与任务定义 | 训练、开发验证与比赛评测 | 否 |
| 官方 baseline 与仓库 | 理解数据格式、任务流程，并复现 official-style IMU 对照 | 否 |
| 官方复现规范 | 组织最终单入口、资产清单和提交材料 | 否 |

官方 baseline 不是最终选中模型的主干。后期复现 official-style IMU CNN-Transformer 是诊断实验，其 held-out 结果较低，没有进入最终系统。

## 2. 公开架构、方法与预训练资源

以下都是公开研究或通用工程方法，不应表述为我们发明：

- R(2+1)D-34 时空 CNN，以及 IG-65M → Kinetics-400 公开初始化。
- CNN、Conv1D、LSTM/BiLSTM、Transformer 和 attention mask。
- FiLM、residual adapter、GroupNorm、knowledge distillation。
- MixUp、label smoothing、EMA、TTA、AdaBN、伪标签和自训练。
- Model Soup、INT8 权重存储和 BF16 计算。
- YOLO11n COCO 预训练检测器，用于确定性人物裁剪，不输出行为类别。
- PyTorch、TorchVision、NumPy、pandas、SciPy、Pillow、OpenCV、SwanLab 等工具。

R(2+1)D 的主体实现调用 TorchVision `VideoResNet`、`BasicBlock`、`Conv2Plus1D` 和 `R2Plus1dStem`，并加载公开 IG-65M/Kinetics checkpoint。YOLO11n 来源与 AGPL-3.0 许可已在 `THIRD_PARTY_NOTICES.md` 中单独列明。

## 3. 社区方案与公开高分方案

赛期检索到的官方仓库主要提供 baseline，没有找到可以直接复现的 Small Model Track 公开高分完整方案。最终系统不是复制社区冠军代码，也不是从他人提交 CSV 反推得到。

我们借鉴的是公开论文中的通用思想，并在 CUHK-X 的验证协议下重新实现、组合、消融和取舍。

## 4. 本项目的任务特定工作

以下可以归为 Lazyshu 针对该任务独立设计和验证的系统工作：

1. 用户隔离双折开发验证、逐用户分析和晋级门槛。
2. 动态多模态审计、缺失模态处理、损坏帧回退和世界坐标 IMU 表示。
3. 共享视觉编码、Skeleton/IMU 专门编码和 masked token fusion 的任务组合。
4. 五源 full-data 参数汤配方与 100 MB 内的部署组织。
5. V52 domain logit residual 与 V57 round-Transformer distillation。
6. 零初始化、GroupNorm 的 protected visual adapters。
7. paired-person cross-environment adaptation 与 60/40 anchored fusion。
8. V127 supervision-aware deterministic routing 和 label-coverage protection。
9. repeat pooling、transition/template beam decoding 等任务结构实现。
10. V164 posterior-rank evidence gate，以及对 V160/V166 结构过拟合的拒绝。
11. 最终资产哈希、动态输入发现、离线推理和 byte-exact reproduction。

正确的原创性表述是：本项目的贡献在于 **针对 CUHK-X 分布偏移和容量约束，对公开基础方法进行系统化组合、任务特定设计、严格验证、压缩和工程复现**，而不是发明基础神经网络组件。

## 5. AI 辅助开发

OpenAI Codex/ChatGPT 参与了实现、调试、实验编排、文档和审阅。参赛者保存的开发会话模型标识为 `gpt-5.6-sol`。具体披露和主办方澄清依据见 `docs/AI_ASSISTED_DEVELOPMENT.md`。

AI 工具作为开发辅助致谢，不列为作者。项目作者、参赛者和维护者为刘钊洋。
