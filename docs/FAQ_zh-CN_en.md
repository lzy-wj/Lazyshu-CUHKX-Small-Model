# Lazyshu 决赛 Q&A 问答库（中英）

## 1. 为什么不用更大的模型？

**中文：** 我们确实测试过更宽的视觉分支和更大范围的解冻，暴露了 1.17 亿到 1.78 亿可训练高层参数，但它们很快在 held-out 用户上遗忘。瓶颈是人员和环境分布偏移，而不是训练容量不足。因此我们选择冻结可迁移主干，只训练零初始化的小残差与适配器。

**English:** We tested wider visual branches and broader unfreezing, exposing 117 to 178 million trainable high-level parameters. They quickly forgot held-out users. The bottleneck was conditional shift, not insufficient fitting capacity, so we froze the transferable backbone and trained only zero-initialized residuals and adapters.

## 2. Parameter soup 是否等于五模型集成？

**中文：** 不是。五个训练配方只在参数空间加权平均，推理时加载一个融合后的 checkpoint，只进行一次基础主干前向。

**English:** No. The five training recipes are averaged in parameter space. Inference loads one merged checkpoint and performs one base-backbone forward, not five model forwards.

## 3. 你们如何处理缺失模态？

**中文：** 每种模态独立预处理和编码，无法使用的模态通过显式 padding mask 从 Transformer 中移除。我们不生成虚假模态，也不强制不同传感器逐时刻对齐。

**English:** Each modality is preprocessed and encoded independently. Unusable streams are removed by an explicit Transformer padding mask. We neither hallucinate missing modalities nor force artificial frame-level alignment across sensors.

## 4. Radar 为什么不进入神经网络？

**中文：** 原始雷达点云稀疏且设备差异大，直接神经编码在本地很强但线上泛化较差。最终系统只在 V164 中使用雷达时间戳形成结构挑战者，而且修改必须得到原多模态 posterior 支持。

**English:** Raw radar points are sparse and device-dependent. Direct neural radar models were locally strong but generalized poorly online. In V164, we use only radar timestamps to form a structural challenger, and any edit must be supported by the original multimodal posterior.

## 5. “跨受试者”与“同人源域监督”是否矛盾？

**中文：** 不矛盾，但必须分开解释。基础模型验证时排除目标自然人；paired adapter 是后续跨环境适配，使用该人在另一个已发布训练环境中的标签，目标环境标签不可见。无同人源域分支则完全没有该自然人的标签。我们分别报告这两种情况。

**English:** They are different evaluation strata. The base model excludes the target natural person. The paired adapter is a later cross-environment adaptation step using that person's released source-environment labels, never target-environment labels. The unmatched branch has no labels from that person, and we report the two strata separately.

## 6. V71 的 FiLM 最终没有直接贡献，为什么保留 4.08 MB？

**中文：** 在最终路由中，paired 样本走 V114；unmatched 样本的 FiLM gate 关闭，输出精确等于 V57。因此 FiLM 修正不改变 selected labels。但冻结提交代码仍加载并执行该模块。为了不在赛后改变提交路径，我们保留它，并完整计入 100 MB 预算。

**English:** In the selected routing, paired rows use V114, while unmatched rows have the FiLM gate off and exactly reproduce V57. So the FiLM correction does not change selected labels. However, the frozen submitted code still loads and executes this module. We retained it for exact post-freeze reproducibility and fully charged its 4.08 megabytes.

## 7. 结构解码的真实执行顺序是什么？

**中文：** 每个候选专家先分别进行 round grouping、repeat pooling 和 transition/template beam search；之后 V127 根据同人监督可用性选择已经解码的结果，再做 coverage protection 和日期 rollback。V164 最后构建独立雷达挑战者并应用 evidence gate。

**English:** Each candidate expert is first decoded independently using round grouping, repeat pooling, and transition or template beam search. V127 then selects an already-decoded route, applies coverage protection and the frozen date rollback. V164 finally builds a separate radar-timed challenger and applies the evidence gate.

## 8. 为什么不选择本地最高的 V166？

**中文：** V166 本地达到 90.35%，但 public score 从主方案的 0.92039 降到 0.88557，说明它过度依赖本地 acquisition script。我们选择了跨 fold、线上证据和复现稳定性更一致的 V127。

**English:** V166 reached 90.35 percent locally, but its public score dropped from 0.92039 for the primary to 0.88557. This indicated over-specialization to local acquisition scripts. We selected V127 because cross-fold validation, online evidence, and reproducibility were more consistent.

## 9. 公榜反馈有没有用于训练？

**中文：** 没有。我们未读取主办方测试标签文件、未人工标注测试样本，也未用公榜推断标签训练模型。四行日期 rollback 来自聚合公榜 ablation，并改变了最终 CSV；单行诊断只用于保留 incumbent 或否决 challenger，没有修改模型、伪标签或解码规则。这些均在报告中披露。

**English:** No leaderboard-inferred label was used to train a model. We did not access an organizer test-label file or manually annotate test clips. Aggregate leaderboard ablation selected the disclosed four-row rollback, while one-row diagnostics only retained incumbents or rejected challengers; they changed no model, pseudo-label, or decoder rule. All uses are disclosed in the report.

## 10. INT8 是否意味着 INT8 推理？

**中文：** 不是。INT8 用于主干 checkpoint 的存储压缩。加载后权重解量化，神经网络前向使用 BF16 autocast。100 MB 是运行资产文件预算，不是显存或实时预算。

**English:** No. INT8 is the storage format of the base checkpoint. We dequantize it for BF16 neural computation. The one-hundred-megabyte rule is an on-disk runtime-asset budget, not a GPU-memory or real-time guarantee.

## 11. 运行时间为什么不是 76.21 + 311.17 秒？

**中文：** 76.21 秒是预处理，311.17 秒是神经网络推理，剩余 20.39 秒来自结构解码和 I/O，因此总耗时为 407.77 秒。

**English:** Preprocessing takes 76.21 seconds and neural inference takes 311.17 seconds. The remaining 20.39 seconds cover structural decoding and I/O, giving 407.77 seconds end to end.

## 12. 如何证明可复现？

**中文：** 包中只有一个离线入口，会检查每个资产的 byte size 和 SHA-256，并验证输入与输出 schema。我们在隔离工作区从原始无标签 Testing 数据重新执行两个 Selected Submission ID，生成的 CSV 与保留提交逐字节一致。

**English:** The package exposes one offline entry point, verifies every asset's byte size and SHA-256, and checks input and output schemas. In an isolated workspace, we reran both selected submission IDs from raw label-free test data, and both CSV files matched the retained submissions byte for byte.

## 13. 你们最核心的创新是什么？

**中文：** 单个模块本身并不是新发明。核心贡献是针对这个任务验证出一套可靠组合：模态合适的编码器、诚实的跨人验证、可逆的小容量适配、监督可用性路由、证据约束的结构解码，以及完整的容量和复现审计。

**English:** We do not claim that each component is new. Our contribution is a validated task-specific composition: modality-appropriate encoders, honest cross-person validation, reversible low-capacity adaptation, supervision-aware routing, evidence-constrained structural decoding, and complete size and reproduction audits.

## 14. 如果再做一次，最优先改进什么？

**中文：** 第一，做更严格的同人监督控制实验；第二，学习更稳健但仍受证据约束的结构解码器；第三，减少多视角和适配器重复前向，降低 13.49 GB 峰值显存和总体延迟。

**English:** First, I would add a label-source-matched control for same-person supervision. Second, I would study a more robust but still evidence-constrained structural decoder. Third, I would reduce repeated multi-view and adapter forwards to improve the 13.49-gigabyte peak memory and overall latency.

