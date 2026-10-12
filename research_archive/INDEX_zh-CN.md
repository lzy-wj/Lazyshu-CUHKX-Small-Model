# Lazyshu 完整研究档案导航

> 这里保留的不只是最终成功的代码，还包括失败实验、并行分支、验证口径调整、合规审计和冻结后诊断。它用来回答“最终方案是怎样一步步得到的”，而不是一个需要顺序运行的软件包。

## 先选一条阅读路线

### 5 分钟：只看主线

1. [关键代数记忆卡](../docs/VERSION_CARDS_zh-CN.md)
2. [完整版本谱系](../docs/RESEARCH_LINEAGE_zh-CN.md)
3. [负结果与研究教训](../docs/NEGATIVE_RESULTS.md)

建议先记住这条链：

```text
V1 → V2 → V3 → V24 → V52 → V57 → V104 → V114 → V127
                                                              └→ V164
```

### 30 分钟：理解研究决策

1. 先读 [实验时间线](../docs/EXPERIMENT_TIMELINE.md)，理解各阶段问题。
2. 再读 [方法来源与原创边界](../docs/METHOD_PROVENANCE.md)，区分公开方法、官方 baseline 和本项目贡献。
3. 按阶段查看 `experiments/` 内的 `README.md`、`RESULTS.md`、`STATUS.md` 和轻量结果文件。
4. 对照 [SwanLab 索引说明](../docs/SWANLAB.md)查找实验记录。

### 复现：只使用冻结入口

最终可运行入口是 [`solution/`](../solution/)，复现边界见 [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md)。历史实验中的绝对路径、旧配置和中间脚本只为追溯保留，不建议作为默认入口。

## 档案规模

| 内容 | 数量 | 说明 |
|---|---:|---|
| 历史文件 | 1,196 | 代码、配置、研究记录、轻量结果与导航 |
| 实验目录 | 178 | 169 个版本命名目录，9 个早期/专项命名批次 |
| 版本范围 | V1–V197 | 编号有跳号，且不代表性能单调提升 |
| 历史配置 | 42 | 训练、输入、融合、验证与部署配置 |
| 研究脚本 | 140 | 数据准备、训练、评估、诊断、压缩与打包 |

上述数字来自当前开源候选树。这些目录不等于 178 个都可直接比较的模型：其中有单变量消融、诊断、失败路线和赛后审计。

## 阶段地图

| 阶段 | 范围 | 核心问题 | 需要记住的结论 |
|---|---|---|---|
| 建立起点 | V1–V5 | 数据、跨用户验证和基础表征 | 输入稳定化与时空视觉带来早期最大提升 |
| 建立研究规范 | V6–V23 | 单变量、双折、结构解码 | 不以单折或单模型上涨作为晋级依据 |
| 全量重训与蒸馏 | V24–V71 | 如何在 100 MB 内保留复杂教师行为 | 建立 V24 锚点，再用小型 residual 和 Transformer 修正 |
| 受保护的适配 | V72–V104 | 提高视觉证据但避免遗忘 | 冻结主干、限制可训练子空间比全量解冻更稳 |
| 同人跨环境 | V105–V127 | 何时可以安全地使用 paired supervision | 按监督可用性做确定性路由，得到最终 V127 |
| 结构推理 | V128–V166 | 能否利用 Radar 时间分组继续提升 | 结构候选必须受神经后验门控；V160/V166 证明本地高分可能不泛化 |
| 冻结后审计 | V167–V197 | 分布偏移、模态对齐、IMU 和合规性 | 未找到可以替换 V127/V164 的稳定方案 |

## 关键分支直达

| 版本 | 档案 | 角色 |
|---|---|---|
| V24 | [`v24_full_refit`](experiments/v24_full_refit/) | 首个稳定部署锚点 |
| V52 | [`v52_domain_logit_residual`](experiments/v52_domain_logit_residual/) | domain-conditioned logit residual |
| V57 | [`v57_round_transformer_distillation`](experiments/v57_round_transformer_distillation/) | 序列教师压缩 |
| V97 | [`v97_protected_visual_adapters`](experiments/v97_protected_visual_adapters/) | 受保护视觉 Adapter |
| V104 | [`v104_domain_gated_adapters`](experiments/v104_domain_gated_adapters/) | 按域启用 Adapter |
| V114 | [`v114_paired_person_fusion`](experiments/v114_paired_person_fusion/) | 同人跨环境融合 |
| V127 | [`v127_supervision_routed_system`](experiments/v127_supervision_routed_system/) | 最终主提交系统 |
| V160 | [`v160_radar_batch_decoder`](experiments/v160_radar_batch_decoder/) | 本地上涨、线上退化的关键反例 |
| V164 | [`v164_evidence_gated_radar`](experiments/v164_evidence_gated_radar/) | 最终次提交 |
| V166 | [`v166_repeated_transition_consensus`](experiments/v166_repeated_transition_consensus/) | 本地最高分但被拒绝 |
| V167 | [`v167_test_distribution_audit`](experiments/v167_test_distribution_audit/) | 冻结后分布审计起点 |
| V197 | [`v197_device_attention_adapter`](experiments/v197_device_attention_adapter/) | 现存版本谱系终点 |

## 目录怎么看

| 路径 | 用途 |
|---|---|
| [`experiments/`](experiments/) | 完整实验分支，包含各版本的代码、状态和结果摘要 |
| [`configs/`](configs/) | 较早期的训练、融合和消融配置 |
| [`docs/`](docs/) | 当时写下的协议、诊断、阶段结论和提交计划 |
| [`scripts/`](scripts/) | 早期通用数据、训练、评估和打包脚本 |
| [`src/`](src/) | 早期共享数据集、模型和验证划分实现 |

## 研究档案的诚实边界

- 版本号是研究批次，不是性能排名。
- 很多实验只回答一个局部问题，不具备独立部署价值。
- 部分高本地分方案已被线上结果否定，它们仍然被保留，因为这些反例构成了最重要的泛化证据。
- 生成的 checkpoint、预测、cache、原始 SwanLab 日志和公榜探测产物没有进入公开仓库。
- 最终系统只继承了少数经验证的分支；具体依赖关系见 [完整版本谱系](../docs/RESEARCH_LINEAGE_zh-CN.md)。
