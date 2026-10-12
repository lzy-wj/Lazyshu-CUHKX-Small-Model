# Lazyshu：从 53.94% 到 CUHK-X 决赛第四名

[English](README_en.md) · [Kaggle 最终榜单](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/leaderboard) · [技术报告](docs/technical_report.pdf) · [完整版本史](docs/RESEARCH_LINEAGE_zh-CN.md)

这是我们参加 **CUHK-X Small Model Track** 的完整方案。主要研发从 9 月 6 日持续到 9 月 15 日，决定最终方案的核心迭代集中在最后一周。

我们从一个 `53.94%` 的六模态 baseline 起步，最终主提交取得 **Private 91.176%**，获得决赛 **第四名**。全部推理资产为 **84.09 MB**。

![Lazyshu 架构](docs/figures/figure_3_architecture.png)

## 我们是怎么一步步做好的

| 时间/版本 | 我们解决的问题 | 结果 |
|---|---|---:|
| V1 | 先跑通六模态训练和跨用户验证 | 53.94% |
| V2 | 人物裁剪、IR 增强、IMU 世界坐标化 | 63.45% |
| V3 | 用真正的时空视频网络替代简单 2D 特征 | 70.52% |
| V24 | 全量训练、参数汤、INT8 和伪标签适配 | 86.07% |
| V52/V57 | 把复杂教师压缩成小型残差和序列 Transformer | 88.06% |
| V114/V127 | 同人跨环境 Adapter，加上按监督条件路由 | 88.80% |
| V164 | 让 Radar 结构候选服从神经网络证据 | 89.39% |
| V166 | 本地最高 90.35%，但线上下降，所以放弃 | — |

早期最大的提升来自把输入和视觉时序做对；后期提升则来自更谨慎地使用监督、Adapter 和结构信息。

我们也走过不少弯路：扩大模型、增加多次前向、直接使用 Radar 分组，都能让本地分数变高，却没有带来更好的线上泛化。最终方案并不是本地最高分，而是我们认为更稳的一版。

## 两种阅读方式

- **只想了解最终方案：** 继续阅读本 README，再看 [`solution/`](solution/) 和 [关键代数记忆卡](docs/VERSION_CARDS_zh-CN.md)。
- **想了解完整研究过程：** 从 [研究档案导航](research_archive/INDEX_zh-CN.md) 进入，查看 V1–V197 谱系、全部现存实验分支和失败记录。

版本号表示研究批次，不是 197 次单调升级。开源树保留了 `1,196` 个历史文件和 `178` 个实验目录，但默认可运行入口始终是 `solution/`。

## 最终方案

- Depth、Thermal、IR 共享一个 R(2+1)D-34 视频编码器。
- Skeleton 使用 temporal CNN，IMU 使用 Conv1D + BiLSTM。
- 各模态编码后，由 masked Transformer 完成融合。
- 有同一自然人的源域标签时使用 paired adapter；没有时使用 decoded V57。
- Radar 原始点不进入神经网络，只在 V164 中提供时间分组候选。
- 所有权重、Adapter、检测器和状态文件合计 `84,094,248 bytes`。

## 成绩

Public 与 Private 成绩可在 [Kaggle 最终榜单](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/leaderboard) 查询。

| 提交 | Public | Private | 说明 |
|---|---:|---:|---|
| V127 | **0.92039** | **0.91176** | 主提交，最终第四名 |
| V164 | 0.91044 | 0.90686 | 结构差异备选 |

两份 CSV 都已经从原始无标签测试数据重新推理，并做到逐字节一致。

## 复现

本次只开源代码、配置和研究文档，**不公开模型权重、decoder state 或最终预测 CSV**，也不建立 Hugging Face/ModelScope 模型仓库。比赛冻结模型包继续由作者本地保存，其哈希为：

```text
SHA-256 060c126ac015135f5265d510ed6db6eaa799d515f22178fb6b12c2cbf28cc55a
```

公开仓库可以复现数据处理、训练方法、模型结构、路由与解码实现，但由于不发布冻结权重，不能仅靠公开仓库逐字节重建比赛提交。拥有合法数据和自行训练的权重后，可以使用同一入口：

```bash
bash inference.sh /path/to/testing/data /path/to/output
```

仓库不重新分发比赛数据。详细边界见 [复现说明](docs/REPRODUCIBILITY.md)。

## 项目内容

```text
solution/            最终推理、训练快照和验证代码
research_archive/    V1–V197 研究谱系与全部现存实验档案
docs/                技术报告、版本史、负结果和模型说明
```

- [项目教程](docs/TUTORIAL_zh-CN.md)
- [完整研究档案导航](research_archive/INDEX_zh-CN.md)
- [V1–V197 完整版本谱系](docs/RESEARCH_LINEAGE_zh-CN.md)
- [负结果与研究教训](docs/NEGATIVE_RESULTS.md)
- [方法来源与原创边界](docs/METHOD_PROVENANCE.md)
- [SwanLab 记录说明](docs/SWANLAB.md)

## 说明与致谢

这个项目使用了 R(2+1)D、Transformer、BiLSTM、FiLM、Adapter、MixUp、Model Soup 等公开方法，也使用了官方 baseline、公开预训练权重和 YOLO11n。我们没有发明这些基础组件；我们的工作主要是针对这个任务完成验证、组合、适配、路由、压缩和复现。

感谢 CUHK-X Challenge 主办方与 CUHK AIoT Lab，感谢 SwanLab 的实验追踪支持，感谢 PyTrio 团队提供计算资源与技术支持，也感谢 PyTorch、TorchVision、Ultralytics 和相关开源社区。

本项目由刘钊洋与 **GPT-5.6-sol** 共同开发。刘钊洋负责参赛、研究目标、计算资源、提交决策和现场演讲；GPT-5.6-sol 参与代码实现、调试、实验编排、分析、文档和审阅。完整说明见 [致谢](ACKNOWLEDGEMENTS.md) 和 [共同开发说明](docs/AI_ASSISTED_DEVELOPMENT.md)。

## License

项目自有代码采用 Apache-2.0。第三方软件、模型和数据仍遵循各自许可证，详见 `THIRD_PARTY_NOTICES.md`。
