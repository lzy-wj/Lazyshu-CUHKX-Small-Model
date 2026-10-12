# Lazyshu 版本演进图谱：从 V1 到 V197

> 这份文档回答两个问题：**这么多版本是怎样发展出来的？哪些版本真的进入了最终方案？**

## 1. 先理解版本号

### 1.1 它不是一条连续产品线

`V1 → V2 → ... → V197` 不是 197 次都成功升级。版本号记录的是研究批次，其中包含：

- 主线里程碑；
- 单变量消融；
- 并行架构分支；
- 压缩或量化实验；
- 失败实验；
- 数据与合规审计；
- 最终冻结之后的继续研究。

因此，不能说“V166 一定比 V127 好”，更不能说“V197 是最终模型”。最终获奖主提交来自 **V127 系统的冻结提交版本**，V164 是第二提交；V167 以后主要是赛后/冻结后的诊断，不改变最终选择。

### 1.2 为什么有跳号

部分编号没有独立目录，可能是临时实验、合并进其他批次、没有完整产物或未正式立项。记版本时只记有明确研究问题和结果的节点，不需要补齐每个整数。

### 1.3 三种状态

| 状态 | 含义 |
|---|---|
| 晋级 / promoted | 成为下一阶段 baseline 或进入最终依赖链 |
| 保留证据 / retained | 方法有局部价值，但不是最终独立路线 |
| 拒绝 / rejected | 在开发折、线上、容量或稳定性上未通过门槛 |

---

## 2. 一张图看懂主线

```text
V1 六模态初始基线 53.94%
  ↓ 输入稳定化
V2 人物裁剪 + 世界坐标 IMU 63.45%
  ↓ 更强时空视觉表示
V3 X3D-S 70.52%
  ↓ 统一 CNN/RNN/Transformer 主干、结构解码、全量重训
V23 稳定训练配方
  ↓ 五源参数汤 + INT8 + 测试伪标签适配
V24 1168/1357，Public 0.91044
  ↓ 压缩复杂教师行为
V52 domain logit residual 1193/1357
  ↓ 学习采集 round 的序列修正
V57 round Transformer 1195/1357
  ↓ gated FiLM 开发学生
V71 1201/1357
  ↓ 多尺度与受保护 Adapter
V89 → V104
  ↓ 同人跨环境 paired adaptation + anchored fusion
V114 1180/1357
  ↓ 按监督是否可用做确定性路由
V127 1205/1357，最终 Primary
  ↓ Radar 只提出结构 challenger
V164 1213/1357，最终 Secondary

旁支警示：
V160 1222/1357，但 Public 0.84079 → 拒绝
V166 1226/1357，但 Public 0.88557 → 拒绝
```

---

## 3. 第一阶段：建立可信起点（V1–V5）

### V1：自建六模态 baseline

- 目标：先跑通数据、训练、跨用户验证、全量重训和提交生成。
- 方法：2D CNN、骨架/IMU/Radar 统计与时序分支、概率融合、缺失模态 mask。
- 结果：Fold 0 `54.60%`，Fold 1 `53.33%`，加权 `53.94%`。
- 结论：六模态简单拼接不等于强模型；跨用户泛化是首要问题。

### V2：先把输入处理正确

- 新增：人物裁剪、IR 对比度增强、Depth/Thermal/IR 独立裁剪、IMU 世界坐标化。
- 结果：`63.45%`，相对 V1 提升 `9.51` 个百分点。
- 结论：输入规范化比盲目加深网络更有效。

### V3：引入真正的时空视觉编码

- 新增：X3D-S 小型 3D 视频编码器，保留其他传感器专家。
- 结果：`70.52%`，是早期最大架构跃迁。
- 结论：细粒度行为的主信息源是人物区域中的时空视觉变化。

### V4：局部上涨但总体不涨

- 探索：上半身细节、骨架注意力、联合统计等。
- 结果：Fold 0 略升，Fold 1 略降，加权仍为 `70.52%`。
- 决策：不替换 V3。
- 结论：单折上涨不是可接受的泛化证据。

### V5：系统保存负结果

- 探索：M 初始化、树模型、全景视频、BatchNorm 适配。
- 结果：没有候选超过预设确认门槛。
- 决策：继续以 V3 为推荐研究包。
- 结论：从这一阶段开始形成“失败也记录、门槛先冻结”的实验文化。

---

## 4. 第二阶段：泛化、结构与稳定训练（V6–V23）

V6–V10 是早期训练、时序和解码迭代，后续由 V11 的固定参考协议统一审计。真正需要理解的是以下节点：

### V11：跨用户泛化实验协议

在不改变推理架构和参数量的情况下，逐项测试：

- structured modality dropout；
- cross-user supervised contrastive loss；
- action-conditioned user adversary；
- BatchNorm 冻结或 momentum；
- class balancing；
- EMA 0.999；
- synchronized speed jitter。

价值不在某个单点分数，而在于建立单变量、固定参考、双折晋级的研究规范。

### V12：blend-aware 选择

研究候选模型是否只在单模型上上涨，还是在固定融合系统中带来互补增益。它提醒我们：一个强但高度相关的模型，未必能改善最终系统。

### V13：pair-aware baseline

- 把采集轮次、重复结构和配对关系纳入结构解码。
- 本地 `1161/1357 = 85.56%`。
- Public `0.89552`。
- 这是从“单 clip 分类”转向“系统级联合预测”的关键节点。

### V14：epoch soup 与短周期伪标签适配

- epoch 10/14/18 参数平均；
- 两个 epoch 的小头部伪标签适配；
- 本地最高 `1185/1357 = 87.32%`，但 public `0.89054`。
- 结论：本地提升不保证线上提升；full-train soup 本身可能带来分布偏移。

### V23：稳定的统一多模态训练配方

- 形成共享 R(2+1)D、Skeleton temporal CNN、IMU 时序编码和 masked Transformer 的稳定主干。
- 固定 BF16、EMA、MixUp、label smoothing、mask distillation 等训练设置。
- 它更像“工程与训练配方里程碑”，是 V24 full-data refit 的直接基础。

---

## 5. 第三阶段：全量重训与紧凑蒸馏（V24–V71）

### V24：第一个真正稳定的部署锚点

- 五个 full-data fits 做固定 parameter soup。
- 主干 INT8 存储；YOLO 与主干合计约 `70.94 MB`。
- 未适配 soup public `0.87064`。
- V13 hard-pseudo adaptation 后，本地 `1168/1357 = 86.07%`，public `0.91044`。
- 角色：后续所有复杂系统的 **base/incumbent anchor**。

### V52：domain-conditioned logit residual

- 输入 adapted/base 共 80 维 log probability。
- 按 domain 学习零初始化 `80→40` Conv1D 残差。
- 测试侧只有 `9,720` 个参数。
- 结果：`1193/1357 = 87.91%`。
- 贡献：证明“小型输出修正器”可以保留复杂教师的域特定行为。

### V57：round Transformer distillation

- 按 acquisition round 组织 clip。
- 用 `92,392` 参数 Transformer 蒸馏序列教师。
- 本地 `1195/1357 = 88.06%`。
- 精确复现教师 `1357/1357` 个本地决策和 `405/405` 个测试决策。
- 贡献：把复杂 sequence teacher 压缩成约 `192 KB` 的部署模块。

### V71：gated FiLM development student

- 三层、256 维 Transformer，带 FiLM-style domain gate。
- 开发结果：`1201/1357 = 88.50%`。
- 内部门控只在 domain 2 打开，其他域精确返回 V57。
- 最终事实：在选中的 V127/V164 中，domain-2 样本改走 V114；无配对样本的 gate 关闭。因此 V71 FiLM 不直接改变最终标签，但冻结包仍保留、执行并完整计入 `4.08 MB`。

### V25–V71 分支索引

| 版本 | 研究问题 | 最终去向 |
|---|---|---|
| V25 | structured teacher distillation | 未晋级 |
| V26 | system distillation | 压缩思路被保留，完整分支未直接部署 |
| V27 | targeted distillation | 未晋级 |
| V28 | wider center crop | 有局部改善，后被更强路线超过 |
| V29 | crop-system fusion | 未晋级 |
| V30 | hard pseudo-label sweep | 未找到更稳健设置 |
| V31 | adapted decoder calibration | 为后续解码校准提供基础 |
| V32 | domain adaptation gate | 门控思想进入后续版本 |
| V33 | domain gate + calibrated decoder | 组合实验，未成为最终锚点 |
| V34 | confidence residual gate | 置信度门控没有稳定价值 |
| V35 | Adapter seed ensemble | 未晋级 |
| V36 | auxiliary rescue | 提供局部救援证据 |
| V37 | auxiliary + domain + calibration | 组合分支未成为最终方案 |
| V38 | X3D-L diversity | 更大视觉模型仅作多样性研究 |
| V39 | diverse compound | 多分支组合未通过最终门槛 |
| V40 | INT4 compression | 压缩过强，未采用 |
| V41 | INT6 compression | 未优于最终 INT8 取舍 |
| V42 | student initialization sweep | 确定学生初始化敏感性 |
| V43 | progressive student mix | 训练轨迹研究 |
| V44 | progressive compound | 组合研究，未晋级 |
| V45 | adapted sequence decoder | 未优于后续 round 路线 |
| V46 | temporal TTA | 未带来稳定泛化增益 |
| V47 | environment AdaBN | 域统计适配未成为最终组件 |
| V49 | fusion-last adaptation | 探索只调融合末层 |
| V50 | per-user pseudo adaptation | 用户级适配思路的前身 |
| V51 | global logit residual | V52 的无 domain 前身 |
| **V52** | **domain logit residual** | **晋级，进入最终依赖链** |
| V54 | recovered repeats | 测试输出与旧方案相同 |
| V55 | modality FiLM expert | V71 路线前身 |
| V56 | retrieved-script rescue | 未晋级 |
| **V57** | **round Transformer distillation** | **晋级，最终 unmatched anchor** |
| V58 | post-teacher repeat pooling | Fold 1 退化 |
| V59 | teacher-sequence hybrid | 本地重要中间版本，未最终选择 |
| V60 | V52/V57 predecode blend | 没有产生新信息 |
| V61 | FiLM seed 43 | 运行不完整，不作结论 |
| V62 | FiLM high-resolution 160 | 运行不完整，不作结论 |
| V63 | FiLM mask distillation | FiLM 训练分支 |
| V64 | FiLM invariant skeleton | 运行不完整，不作结论 |
| V65 | FiLM per-visual auxiliary | 模态辅助监督研究 |
| V66 | supervised expert Transformer | 专家式序列学生研究 |
| V67 | paired-domain expert gate | 明确 paired domain 门控 |
| V68 | FiLM full-data refit | 全量部署研究 |
| V69 | FiLM expert Transformer | 专家 Transformer 研究 |
| V70 | FiLM margin distillation | _margin_ 蒸馏研究 |
| **V71** | **gated high-fidelity FiLM student** | **开发里程碑；资产保留，最终 gate 不改标签** |

---

## 6. 第四阶段：提高视觉证据但保护泛化（V72–V104）

### V89：固定多尺度边际化

- 128px 与 160px 两个小 head delta 固定等权平均。
- 不搜索融合权重。
- 本地 `1170/1357`，两折目标域加权指标都改善。
- 贡献：多尺度确实增加了一点独立证据，但收益不大。

### V97：受保护视觉 Adapter

- 直接解冻大量 CNN 高层参数会遗忘跨用户表示。
- 改为冻结主干，只训练零初始化、GroupNorm 的时空 Adapter。
- 贡献：建立“限制可训练子空间 + 原模型安全锚点”的后期范式。

### V98：cross-modal target adapter

- 用融合教师监督视觉 Adapter。
- raw 从 V89 的 `1077` 提升到 `1090`，但 decoded 只有 `1154`。
- 结论：视觉证据变好，不代表现有 decoder 一定能利用；不直接晋级。

### V104：domain-gated adapters

- 只在两折都有证据的 domain 启用 Adapter，其他域回退。
- 结果：`1175/1357`，两折分别比 V89 `+2/+3`。
- 贡献：把 Adapter 与安全 domain routing 结合。

### V72–V104 分支索引

| 版本 | 研究问题 | 最终去向 |
|---|---|---|
| V72 | modality-BN FiLM | 未晋级 |
| V73 | confidence-gated FiLM | 置信度门控未成为最终规则 |
| V74 | FiLM per-visual soup | 未晋级 |
| V75 | per-visual full refit | 视觉流全量重训分支 |
| V76 | high-res per-visual FiLM | 高分辨率 FiLM 分支 |
| V77 | seed-43 per-visual FiLM | 随机种子稳定性分支 |
| V78 | wide iterative fusion | 更宽循环融合未通过门槛 |
| V79 | wide loop trust region | trust-region 循环未通过门槛 |
| V80 | spatial masked encoder | 空间 masked encoder 有可迁移证据，未成最终主线 |
| V81 | validation rebuild | 重建/审计验证口径 |
| V82 | V24 conservative rescue | 保守回退研究 |
| V83 | synchronized multimodal flip | 同步增强未稳定增益 |
| V84 | residual-invariant skeleton | 未晋级 |
| V85 | skeleton scale randomization | 未晋级 |
| V86 | high-resolution spatial encoder | 高分辨率视觉改善 |
| V87 | V24 high-res transfer | 把高分辨率信息迁回 V24 |
| V88 | high-res pseudo validation | raw 改善，但最终 gate 未通过 |
| **V89** | **fixed pseudo multiscale** | **晋级为后续 Adapter 锚点** |
| V90 | spatial pseudo adaptation | 未晋级 |
| V91 | error attribution | 分解错误来源，不是部署模型 |
| V92 | cross-scale consensus | 伪标签更干净，但最终结果未提高 |
| V93 | upper-body multiview | 未晋级 |
| V94 | upper-body consistency | 未晋级 |
| V95 | phase-aligned label experts | 未晋级 |
| V96 | modality-specialized large backbone | 117.2M–178.4M 高层可训练参数导致遗忘 |
| **V97** | **protected visual adapters** | **Adapter 设计被保留** |
| V98 | cross-modal target adapter | raw 涨、decoded 降，未直接晋级 |
| V99 | refreshed sequence teacher | 未晋级 |
| V100 | diverse multiforward | 多次前向没有新增益 |
| V101 | native 224 multiscale | 未晋级 |
| V102 | native 192 scale | 未优于 160px 取舍 |
| V103 | full protected adapters | V104 的全域前身 |
| **V104** | **domain-gated protected adapters** | **晋级，paired 路线锚点** |

---

## 7. 第五阶段：同人跨环境监督与最终路由（V105–V127）

### V113：paired-person adapter

- 使用同一自然人的 source-environment 真实标签。
- target 环境仍只使用伪标签。
- equal fusion 后 decoded `1176/1357`。
- 贡献：证明 paired adaptation 路线值得继续。

### V114：anchored paired fusion

- `60%` paired-adapter evidence + `40%` V104 protected evidence。
- raw `1091/1357`，decoded `1180/1357`。
- 相比 V104 增加 5 行，且不是纯 decoder 假提升。

### V127：supervision-aware routing

- 有同人 source：使用 decoded V114。
- 无同人 source或未知日期：使用 V71 代码路径，但 gate 关闭，实际为 decoded V57。
- source 标签不支持 V24 incumbent：恢复 decoded V24。
- 本地 `1205/1357 = 88.80%`。
- 冻结的日期 rollback 形成最终 primary CSV。
- Public `0.92039`，Private `0.91176`，私榜第 4。

### V105–V127 分支索引

| 版本 | 研究问题 | 最终去向 |
|---|---|---|
| V105 | PoseC3D heatmaps | 未晋级 |
| V106 | isolated target adapters | 目标域 Adapter 对照 |
| V107 | sqrt class balance | 类别平衡对照 |
| V108 | center-detail expert | 中心细节视觉专家未晋级 |
| V109 | cross-fit round Transformer | V104 从 1175 降到 1144，拒绝 |
| V110 | object-token Transformer | 物体 token 分支未晋级 |
| V111 | robust adapter consensus | 无增益 |
| **V113** | **paired-person adapter, equal fusion** | **paired 路线成立** |
| **V114** | **retention-heavy anchored fusion** | **paired 专家晋级** |
| V115 | confidence-anchored adapters | 置信度加权未晋级 |
| V116 | iterative paired adapters | 迭代适配未晋级 |
| V117 | longer paired training | raw 涨但 decoder 退化 |
| V118 | multimodal paired adapters | 没有额外增益 |
| V120 | retrieved donor adapters | donor 检索未晋级 |
| V121 | multi-donor adapters | 多 donor 未晋级 |
| V123 | paired/person-specific heads | 人员专属头未晋级 |
| V124 | system-aware adapters | 系统级适配未晋级 |
| V125 | domain complement | V127 路由的直接前身 |
| V126 | shared MViT | 作为多样性教师研究，不部署 |
| **V127** | **supervision-routed system** | **最终 Primary 主体** |

---

## 8. 第六阶段：寻找结构大突破并控制风险（V128–V166）

### V160：Radar batch decoder

- 用 Radar 文件时间戳替换分组 key，不改变模型 posterior。
- 本地 `1222/1357 = 90.05%`。
- Public `0.84079`。
- 结论：直接结构分组严重过拟合测试之外的采集规律。

### V164：evidence-gated radar hedge

- 保留 V127 primary 作为 incumbent。
- Radar decoder 只产生 challenger。
- challenger posterior rank `<=2` 且 incumbent rank `>=3` 才替换。
- 本地 `1213/1357 = 89.39%`，两折 `+8/-0`。
- Public `0.91044`，Private `0.90686`。
- 最终作为 Secondary，提供有限结构多样性。

### V166：repeated-transition consensus

- 同一目标人物中，相同 incumbent→challenger 转移至少重复两次才接受。
- 本地项目最高 `1226/1357 = 90.35%`，共 `+21/-0`。
- 60 个 partial-script stress views 均无新增错误。
- Public 只有 `0.88557`，因此拒绝。
- 研究结论：开发折中的一致性不等于未知测试脚本中的因果规律。

### V128–V166 分支索引

| 版本 | 研究问题 | 最终去向 |
|---|---|---|
| V128 | MViT domain adaptation | 保留为多样性教师，不部署 |
| V129 | full MViT refit | 未部署 |
| V130 | MViT consensus rescue | 未晋级 |
| V131 | hand trajectory expert | 未晋级 |
| V132 | decoder acceptance gate | 未优于后续证据门控 |
| V133 | shared Swin3D | 未晋级 |
| V134 | DINOv2 temporal | 直接路线未晋级 |
| V135 | end-to-end label experts | 中止/拒绝 |
| V136 | retained label experts | 未晋级 |
| V137 | session-matched decoder | 未晋级 |
| V138 | soft session prior | 无增益 |
| V139 | environment-paired strength | 仅微小或不迁移增益 |
| V140 | sensor-style randomization | 未晋级 |
| V141 | paired embedding prototypes | 未晋级 |
| V142 | cross-environment metric learning | 未晋级 |
| V143 | global template matching | 两折净损失，拒绝 |
| V145 | motion-saliency crop | 未晋级 |
| V146 | balanced warm refinement | 平衡训练分支 |
| V147 | class-user balanced sampling | 退化 |
| V148 | environment-specific teachers | 未晋级 |
| V149 | multimodal person windows | 未晋级 |
| V150 | Group DRO | 泛化稳健性实验 |
| V151 | hierarchical activity loss | 层级标签实验 |
| V152 | balanced softmax | 类别不平衡实验 |
| V153 | hierarchical decoder | 未晋级 |
| V154 | triplicate template matching | Fold 0 持平、Fold 1 下降 |
| V155 | source subsequence decoder | 1181，低于 V127 的 1205 |
| V156 | cross-environment contrastive | 表征对齐实验 |
| V157 | partial-script stress | 压力测试工具，不是部署模型 |
| V158 | Group DRO + balanced softmax | 组合泛化实验 |
| V159 | ordered source alignment | 顺序对齐实验 |
| **V160** | **direct radar batch decoder** | **本地强、线上崩，拒绝** |
| **V164** | **evidence-gated radar challenger** | **最终 Secondary** |
| V165 | balanced-cycle radar | V166 前的更强本地结构分支 |
| **V166** | **repeated-transition consensus** | **本地最高、线上拒绝** |

---

## 9. 第七阶段：冻结后的泛化审计（V167–V176）

这些版本用于回答“还能否找到真正的大泛化提升”，但没有替换已选 V127/V164。

| 版本 | 研究问题 | 结论 |
|---|---|---|
| V167 | test-distribution audit | 95% 目标理论可想象，但现有证据不能证明 |
| V168 | modality distribution alignment | 未晋级 |
| V169 | leave-one-user supervision | 未晋级 |
| V170 | raw radar-point expert | 未晋级；Radar 原始点未进入最终主干 |
| V171 | full-modality high resolution | 未晋级 |
| V172 | thermal frame ResNet | 未晋级 |
| V173 | TSN visual expert | 未晋级 |
| V174 | paired-source frame expert | 未晋级 |
| V175 | paired-source multimodal | 未晋级 |
| V176 | full-scene multimodal | 未晋级 |

这组实验强化了一个结论：缺少的不是更多视觉模型，而是对未知人员/环境条件更可靠的适应机制。

---

## 10. 第八阶段：合规、身份与 IMU 专项审计（V177–V197）

这些版本主要是冻结之后的可解释性与传感器研究，不属于最终获奖提交的代际升级。

| 版本 | 研究问题 | 结论/角色 |
|---|---|---|
| V177 | compliance reset | 重置并核对合规边界 |
| V178 | identity mapping audit | 核对 user ID 与自然人关系 |
| V179 | exact source-round matching | 精确 source round 对齐实验 |
| V180 | combined relative IMU | 相对 IMU 表示实验 |
| V181 | relative IMU experts | 多 IMU 专家实验 |
| V182 | magnetic IMU | 磁力计诊断 |
| V183 | statistical IMU | 统计传感器对照 |
| V184 | world + magnetic main | 世界坐标与磁力融合实验 |
| V185 | statistical residual | 统计残差实验 |
| V186 | official-style IMU Transformer | 训练近 99%，held-out 约 23%–25% |
| V187 | statistics-only adapter | 统计 Adapter 对照 |
| V188 | linear IMU | 线性可分性对照 |
| V189 | low-LR statistics adapter | 低学习率适配实验 |
| V190 | statistics logit adapter | logit 级统计适配 |
| V191 | low-LR statistics logit | 低学习率 logit 适配 |
| V192 | clean statistics logit | 清洁输入审计 |
| V193 | clean statistics token | token 级统计适配 |
| V194 | statistics token seed 43 | 随机种子复核 |
| V195 | exact official IMU reproduction | 未晋级 |
| V196 | adapter audit | Adapter 行为与资产审计 |
| V197 | device-attention IMU adapter | 两折退化，拒绝 |

---

## 11. 最终系统到底继承了哪些版本

不要把最终包理解为把所有版本都叠起来。可用下面的依赖关系理解：

| 最终组件 | 主要来源 |
|---|---|
| 共享 CNN/RNN/Transformer 基础主干 | V23/V24 |
| 五源参数汤与 INT8 主干 | V24 |
| 小型 domain logit correction | V52 |
| unmatched sequence anchor | V57 |
| 冻结保留的 gated FiLM asset | V71；最终 gate 不改标签 |
| 多尺度 head delta | V89 |
| protected/domain-gated adapter | V97/V104 |
| paired-person anchored expert | V113/V114 |
| supervision-aware router | V125/V127 |
| optional Radar challenger | V160 思路，经 V164 证据门控后保留 |

最终没有继承：V96 的大规模解冻、V109 的跨折序列学习、V160 的直接 Radar 覆盖、V166 的重复转移共识，以及 V167–V197 的后续研究分支。

---

## 12. 用四句话复述完整研发史

1. **V1–V3：先把输入和时空表示做对。** 人物裁剪、世界坐标 IMU 和 3D 视频编码把基线从 53.94% 提升到 70.52%。
2. **V23–V71：建立统一主干并压缩复杂教师。** 参数汤、伪标签、logit residual 和 round Transformer 把系统推进到 88% 左右，同时控制在 100 MB 内。
3. **V89–V127：从统一模型转向条件化适配。** 受保护 Adapter、同人跨环境监督和按监督可用性路由形成最终主方案。
4. **V160–V166：结构先验很强但易过拟合。** 最终只保留由神经 posterior 约束的 V164 hedge，主提交仍选择更稳健的 V127。
