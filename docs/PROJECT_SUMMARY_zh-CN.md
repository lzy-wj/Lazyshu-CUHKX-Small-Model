# Lazyshu 项目完整总结

## 1. 一句话概括

我们构建了一套面向跨受试者分布偏移的轻量级多模态行为识别系统：用共享时空 CNN、结构化传感器时序编码器和 masked Transformer 得到基础证据，再通过受保护的适配器、确定性路由和保守结构解码提升泛化；最终模型资产为 84.09 MB，Kaggle 私榜准确率 91.176%，排名第 4，并能从原始无标签测试数据完整复现提交结果。

## 2. 比赛任务与真正难点

### 任务

- 40 类日常行为识别。
- 可用模态：Depth、Thermal、Infrared、Skeleton、IMU、mmWave Radar。
- 训练集为 3,036 个有标签片段；公开 Testing 为 405 个无标签片段。
- 测试使用 held-out user IDs，核心评测是跨受试者泛化。
- 允许的模型架构为 CNN、RNN、Transformer；所有运行时模型资产合计不超过十进制 100 MB。

### 难点

1. **人的分布偏移**：训练拟合很高并不代表能泛化到新的人；部分传感器模型训练准确率接近 99%，held-out 人员上却只有 23%–53%。
2. **模态异构**：视频、骨架、IMU 和雷达采样频率、时间支持、缺失方式完全不同，不能强行逐帧对齐。
3. **细粒度动作**：翻书与阅读、扫地与拖地等类别依赖微小的手部、物体和时序差异。
4. **严格容量预算**：检测器、主干、适配器、残差模块和解码状态全部计入 100 MB。
5. **本地验证与线上分布不同**：后期最强本地结构规则并没有带来更好的 Kaggle 泛化，因此不能只追求本地最高分。

## 3. 验证设计

我们没有使用随机 clip 切分，因为它会泄露人员与环境特征，而是建立两个 user-isolated development folds：

- Fold 0：652 个片段，用于主要架构选择。
- Fold 1：705 个片段，最初用于迁移确认，后期也参与多次方案筛选。

因此两个 fold 都明确称为**开发验证**，不包装成独立测试集。方案冻结后，才使用全部 3,036 个有标签片段重训部署模型；全量训练 checkpoint 不拥有 held-out 分数。

这里还要区分两个概念：

- **user ID**：数据集中的记录身份。
- **natural person**：真实自然人；同一自然人在不同环境中可能对应不同 user ID。

基础模型验证时排除目标自然人。在有同人源域录像时，后续适配器可以使用该自然人在另一环境中的训练标签；目标环境标签始终不可见。没有同人源域时，则是真正的无同人监督分支。最终路由在开发集上的描述性分层结果为：

- 有同人源域监督：645/699，92.27%。
- 无同人源域监督：560/658，85.11%。

这两个数字是开发分层结果，不是独立测试估计。

## 4. 数据预处理

### 视觉模态

- 使用打包的 YOLO11n 做确定性人体区域检测；检测器不预测行为类别。
- 对整个 clip 使用联合人体框，避免每帧框抖动。
- 均匀采样 32 帧，生成 128 和 160 两种分辨率缓存。
- Depth 与 Thermal 使用三通道表示，IR 在共享编码器入口扩展到三通道。
- 损坏帧回退到邻近有效帧；整条模态失效时通过 mask 排除。

### Skeleton

- 输入 17 个三维关节。
- 在坐标之外加入骨向量和时间速度。
- 使用 temporal CNN 编码局部运动模式。

### IMU

- 最多五个穿戴设备。
- 使用四元数将加速度与角速度转换到世界坐标系。
- 构成 85 维时序特征。
- 使用两层 Conv1D 与双向 LSTM 编码。

### Radar

- 原始雷达点不进入神经网络主干。
- 雷达时间戳只用于可选的 V164 结构挑战者进行时间分组。

## 5. 基础多模态网络

### 视觉编码器

- Depth、Thermal、IR 共享一个 R(2+1)D-34 时空 CNN。
- 共享权重减少容量，并迫使三种隐私视觉模态学习可迁移的运动表示。
- 视觉特征投影为 192 维 token，同时保留视觉分类 logit skip。

### 结构化传感器编码器

- Skeleton：temporal CNN。
- IMU：Conv1D + BiLSTM。

### 融合

- 为每种模态加入 learned modality embedding。
- 两层、六头、192 维的 pre-norm Transformer 处理 `[CLS]` 和可用模态 token。
- padding mask 显式移除缺失模态。
- Transformer 输出作为残差，加到聚合视觉 logits 上。

基础分类器共 64,953,671 个参数。

## 6. 训练与压缩

### 五个全量训练来源

冻结架构后，训练五个互补配方：基础、跨用户、progressive-160、scratch masked-distillation、warm masked-distillation。它们在参数空间按固定权重融合：

`0.082416 / 0.079504 / 0.148508 / 0.089572 / 0.600000`

这是**参数汤**，不是推理时跑五个模型；推理只加载一个融合后的主干。

### 训练技术

- BF16 autocast，FP32 优化。
- AdamW、warmup + cosine decay。
- EMA、MixUp、label smoothing。
- clip-consistent horizontal flip。
- masked-modality distillation。

### 存储

- 主干以 INT8 形式存储，运行时解量化并使用 BF16 前向。
- INT8 解决的是文件体积，不等同于 INT8 计算，也不等同于低显存保证。

## 7. 小残差与适配器

### 紧凑残差蒸馏

- Domain logit residual：用小型 Conv1D 修正 adapted/base log probabilities。
- Round Transformer：按采集轮次组织片段，用 92,392 参数学生蒸馏序列教师。
- V71 gated FiLM 是开发阶段学生。它只在一个 paired-source domain 打开修正；其余域精确返回 V57 anchor。

重要事实：在最终选中的 V127/V164 路径里，V71 的 FiLM 修正并不会直接影响最终标签。有配对源的样本走 paired adapter；无配对源时 FiLM 关闭，输出等于 V57。由于冻结复现代码仍加载并执行该模块，我们保留并完整计入其 4.08 MB，而没有在赛后修改提交路径。

### 受保护视觉适配器

- 冻结完整源模型。
- 在视觉 layer 4 前后插入零初始化的时空 residual adapter。
- GroupNorm 避免目标域 running statistics 漂移。
- paired-person adapter 使用同一自然人在源环境中的真实标签，并保留目标伪标签监督。
- Anchored fusion 使用 60% paired-adapter evidence + 40% protected source evidence。

由于没有做 label-source-matched 的严格控制实验，我们只把收益归因于“完整的 paired adaptation + anchored fusion 方案”，不单独声称同人标签本身造成全部提升。

## 8. 路由与结构解码

真实执行顺序是：

1. 分别计算并结构解码 V24 incumbent、无配对序列 anchor、paired adapter。
2. 有同人源域时选 paired adapter；否则选精确 V57 anchor。
3. 应用 matched-source coverage protection；必要时恢复 decoded V24。
4. 对满足固定日期与模型分歧条件的样本执行四行 rollback。
5. V164 再使用雷达时间戳构造独立结构挑战者，仅在原多模态 posterior 支持时接受修改。

### 结构解码细节

- 通过时间戳形成 acquisition rounds。
- 相似且长度一致的重复轮次进行概率池化。
- 每轮用 beam search 综合 clip log probability 和冻结 transition score。
- 约束轮内标签不重复。
- 有同人源时，可比较源序列模板与自由分配；只有惩罚后的模板分数更高时才替换。

核心原则是：**结构信息只能提出或限制修正，不能无条件覆盖多模态证据。**

## 9. 实验演进

| 阶段 | 主要变化 | 开发结果 |
|---|---|---:|
| V1 | 初始 2D-CNN / 统计模型 | 53.94% |
| V2 | 人体裁剪、IR 增强、世界坐标 IMU | 63.45% |
| V3 | X3D-S 时空视觉专家 | 70.52% |
| V24 | INT8 参数汤 + 结构解码 | 86.07% |
| V52 | domain logit residual | 87.91% |
| V57 | round Transformer distillation | 88.06% |
| V127 | supervision-aware routing | 88.80% |
| V164 | evidence-gated radar hedge | 89.39% |
| V166 | repeated-transition consensus | 90.35%，线上拒绝 |

V166 虽然本地最高，但 public score 只有 0.88557，显著低于主方案。因此最终选择不是本地分数最高的系统，而是跨受试者验证、线上证据和复现稳定性共同支持的系统。

## 10. 最终提交与效率

### Primary：V127

- Public：0.92039。
- Private：0.91176。
- 私榜第 4。

### Secondary：V164

- Public：0.91044。
- Private：0.90686。
- 相对 primary 仅改变 7/405 个标签，作为结构多样性 hedge。

### 资源

- 14 个运行资产，总计 84,094,248 bytes。
- 距离 100,000,000 bytes 限制还有 15,905,752 bytes。
- A800 上 405 个片段：
  - 预处理 76.21 秒。
  - 神经网络推理 311.17 秒。
  - 结构解码与 I/O 20.39 秒。
  - 总耗时 407.77 秒。
  - 峰值 CUDA allocation 13.49 GB。

## 11. 公榜反馈、合规与披露

- 未读取主办方测试标签文件。
- 未人工标注测试片段。
- 未把公榜推断出的样本标签用于训练模型。
- 模型生成的无标签测试集伪标签用于获准的自训练/适配。
- 四行日期 rollback 来自聚合公榜 ablation，并实际改变了 primary CSV。
- 11 次单行诊断中，4 次确认 incumbent；7 次分数不变，不能确认候选标签。
- 单行诊断只用于保留 incumbent 或否决 challenger，没有修改训练、伪标签、transition、threshold、decoder 或 radar hedge 规则。
- 报告没有声称单行探测获得主办方专项批准，而是完整披露其信息来源与使用边界。

## 12. 复现

- 单一入口自动发现数据根目录、样本数和模态。
- 保留官方 CSV 顺序。
- 对所有运行资产检查 byte size 与 SHA-256。
- 检查 submission ID、schema、顺序、重复路径、空值和标签范围。
- 推理全程离线，无随机采样。
- 两个 Selected Submission ID 均从原始无标签 Testing 数据重新执行。
- 生成 CSV 与保留提交逐字节一致，包括 SHA-256。

## 13. 哪些是公共技术，哪些是我们的工作

### 公共或已有技术

- R(2+1)D、Transformer、LSTM、MixUp、model soup、FiLM、adapter。
- YOLO11n 预训练检测器。
- IG-65M / Kinetics 初始化的公开视频 CNN checkpoint。

### 我们的任务特定贡献

- 六模态数据审计、动态预处理和缺失模态策略。
- 适合异构模态的 CNN/RNN/Transformer 组合方式。
- 世界坐标 IMU 表示与共享视觉主干。
- 两折 user-isolated 验证协议与严格的 promotion gate。
- 五模型参数汤的具体训练、权重选择与 INT8 部署。
- 小残差蒸馏、paired-person protected adapters 与 anchored fusion。
- supervision-aware deterministic routing。
- repeat pooling、transition、template alignment 和 evidence-gated radar challenger 的完整结构解码链。
- 100 MB 资产审计、单入口复现与字节级提交验证。

## 14. 最值得在演讲中强调的三点

1. **我们不是靠堆大模型取胜，而是靠验证设计、保守适配和结构化决策提升跨人泛化。**
2. **参数汤把五个训练来源压成一次前向；整个部署只有 84.09 MB。**
3. **我们主动拒绝本地最高分方案，并选择线上与跨受试者证据更一致、可以完整复现的系统。**

