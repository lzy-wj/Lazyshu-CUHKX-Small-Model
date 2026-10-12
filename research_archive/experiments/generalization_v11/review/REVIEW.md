# V11 R(2+1)D 跨主体泛化独立审阅

## 审阅结论

当前三项实验**不应按现有代码直接并行启动**。

1. **A 当前不是计划中的“真正视觉/传感器 structured modality dropout”**：它只在 RGB/IR 都存在时将其中一组归零，既不丢弃整个视觉分支，也不将被丢弃的 skeleton/IMU 输入归零；非 shared V9 还完全忽略 `visual_mask`。
2. **B 的基本损失方向正确，修复成本最低，建议最高优先级**：same-class/different-user 正样本定义和同用户同类排除均正确，但全局 FIFO queue 未做类/用户均衡，2048 项偏陈旧，且新代码在 B 关闭视觉 dropout 时仍额外消耗随机数，无法与历史 baseline 严格配对。
3. **C 当前不具备因果可解释性，不能作为有效 adversarial removal 消融**：训练用户与动作标签明显混杂，无条件用户分类器会利用动作类别预测用户，GRL 随后会把动作信息本身从表示中删除。必须改为 label-conditioned user adversary，并做条件均衡。

建议执行顺序为：**先修复并复跑同代码 control，再 B，再 A，最后 C**。若强调工程成本，可在 control 后并行 B/A；C 应等待条件化实现审阅通过。

## 审阅快照

审阅期间主训练脚本被另一并行任务更新过，因此本报告锁定以下最终快照（2026-09-08 12:52 CST），不评价更早的中间版本：

- `src/r2plus1d_models.py`: `f221b31a0de6bc11f5dc883c7a14fead1429e48d4e60bea5f69a701b67b8aef8`
- `scripts/train_r2plus1d.py`: `c56d1820a0b3702719b1bc5dd4da5bcbf18d20aca6e0d9dc80075034afc0ba6b`
- `src/r2plus1d_data.py`: `416af7e139e05226e448735e883e8ff531118c8e954334b1e04d218effad7263`
- `scripts/evaluate_single_ablation.py`: `6d828a2e02604938036492911de11e3e7ca213b9ac7f24f97a27502c7946bae2`

`experiments/generalization_v11/audit_methods.py` 在该快照上通过，但它只验证局部张量机制，不足以证明 A 的 mask 真正影响非 shared 模型，也没有覆盖 B 的 queue 均衡、C 的标签混杂或 control 随机数等问题。

## 当前泛化问题

### 1. 明显的跨主体拟合差距

- fold0 staged 右端在所选 epoch 3 的在线增强训练准确率为 93.46%，clean EMA 验证准确率为 65.34%，表面差距约 28.1 个百分点。两者使用 online/EMA 和 augmented/clean 两套条件，不能视作严格 gap，但足以说明主体外泛化是主要问题。
- 最终 R(2+1)D soup 的 fold0 用户准确率为：user3 65.03%、user8 66.86%、user18 72.07%、user23 66.67%。问题不是单个灾难性用户，而是广泛的跨主体退化。
- 622 个视觉、骨架、IMU 均可用样本的 standalone 准确率为 70.58%；13 个 IMU 缺失样本仅 23.08%，16 个三分支均缺失样本为 0%。缺失模态是困难尾部，但只覆盖 30/652，不能单独解释整体上限。

### 2. 类别、用户和采集环境混杂

- fold0 训练集有 14 个用户、2,384 个样本；每类样本数从 12 到 298，默认 `class_balance_power=0`，主损失和选择指标都偏向高频类。
- 14 x 40 个 user-class 单元中有 147 个为空；单个训练用户只覆盖 23–34 类。
- 每个用户相对全局类别分布的 total-variation distance 为 0.179–0.350。用户分类器可仅凭动作类别先验获得大量信号，因此无条件 GRL 会惩罚真实动作表征。
- 用户编号还编码两个采集环境。移除环境风格可能有益，但必须在给定动作类别后移除 residual user information，而不是直接删除所有可预测用户信息。

### 3. 视觉路径占主导

- 非 shared 模型将 RGB+IR 四通道送入同一个 R(2+1)D，只有一个视觉 token 流；最终输出是 `video_logits + fusion_residual`（`src/r2plus1d_models.py:139`、`src/r2plus1d_models.py:175`）。
- fusion head 最后一层零初始化（`src/r2plus1d_models.py:123`），staged 开始时严格退化为视觉分类器；视频分支还同时接受主 CE 和 video auxiliary CE，进一步加强视觉主导。
- B/C 目前只约束 `LayerNorm(fused CLS)`；梯度会经 visual token 回到视频编码器，但不会直接约束绕过 fusion 的 `video_head` 路径。因此即使辅助指标改善，也可能对最终视频 residual 的主体信息影响不足。
- R(2+1)D 的 `BatchNorm3d(momentum=0.9)` 使用 batch=8 的高更新率统计（`src/r2plus1d_models.py:25`），可能放大用户混合批次的统计波动。三项消融必须保持它不变，但应记录为残余泛化风险。

## 现有 Mask 行为

### 数据端

- `visual_mask` 在归一化前通过整个 clip 是否存在任意非零像素得到，RGB 和 IR 各一个 clip-level bit（`src/r2plus1d_data.py:79`）。
- 原始缺失视觉流是全零 uint8，之后仍执行 Kinetics 标准化，所以 CNN 实际看到的是固定负常数，而不是中性零输入（`src/r2plus1d_data.py:87`）。
- 缺失 skeleton/IMU 在数据增强时仍会加高斯噪声（`src/r2plus1d_data.py:95`），之后依靠 mask 阻断融合和辅助损失。
- fold0 训练可用率：视觉 96.35%、骨架 96.31%、IMU 94.00%；验证可用率分别为 97.55%、97.39%、95.40%。RGB/IR 几乎总是同时存在。

### 训练端

- skeleton 与 IMU 的 mask 各自以 10% 概率清零（`scripts/train_r2plus1d.py:95`），但输入张量本身不归零。
- 另一个 10% `visual_drop` 只修改 `visual_mask`（`scripts/train_r2plus1d.py:97`）。在当前 V9 的非 shared `multimodal` 模式中，模型把视觉 padding mask硬编码为全 false（`src/r2plus1d_models.py:165`），因此这部分 visual dropout 是 **no-op**。
- sensor mask 会屏蔽对应 token 作为 Transformer key/value（`src/r2plus1d_models.py:167`），并屏蔽 skeleton/IMU auxiliary CE（`scripts/train_r2plus1d.py:462`），所以 sensor dropout 在 fusion 上有效，但不是完整的输入级 structured dropout。
- video auxiliary CE 的 mask 永远为 1（`scripts/train_r2plus1d.py:459`）；自然视觉缺失或未来整段视觉 dropout 时仍会强制视觉支路分类。
- skeleton/IMU auxiliary loss 对每个 batch 的可用样本重新求均值。即使某支路只剩一个有效样本，其整体权重仍约为 `0.15/3=0.05`，会产生高方差梯度。

## A：Structured Modality Dropout

### 对当前实现的判定

**不通过计划一致性审阅。** `--visual-channel-dropout 0.15` 的实际定义是：15% eligible clip 随机归零 RGB 或 IR 之一（`scripts/train_r2plus1d.py:102`），且只在二者均可用时执行。它更准确的名称应是 `visual_stream_dropout`，不是视觉/骨架/IMU 三分支 structured modality dropout。

额外问题：

- 它永远保留至少一个视觉流，不能训练无视觉条件下的 skeleton/IMU 融合能力。
- 更新后的 `visual_mask` 在非 shared 模型仍被忽略；真正产生效果的只有视频通道归零。
- skeleton/IMU 仍只改 mask、不归零输入，三分支处理不对称。
- 即使参数为 0，代码仍执行两次额外随机采样（`scripts/train_r2plus1d.py:104`、`scripts/train_r2plus1d.py:107`），会改变随后 Transformer/dropout 的 CUDA RNG 序列。B、C 和新 control 因此不能与历史 baseline 做严格同随机轨迹比较。

### 建议的确认性实现

把物理输入映射为模型的三个推理分支：`video=(RGB+IR joint)`、`skeleton`、`IMU`。

1. 从自然 mask 构造 `[visual_mask.any(), skeleton_mask, imu_mask]`。
2. 对每个当前可用分支独立 Bernoulli dropout，主实验保持原概率 **p=0.10**，避免同时改变既有概率和机制。
3. 若所有自然可用分支都被丢弃，均匀恢复其中一个；自然全缺失样本不能伪恢复。
4. 被丢弃的 normalized video、skeleton、IMU 输入全部置零。
5. 为非 shared 模型增加 joint visual token mask；视觉不可用时同时令 `video_logits=0`，并将 video auxiliary CE mask 设为 0。
6. skeleton/IMU auxiliary CE 使用 post-drop mask；不要求被丢弃的单支路从空输入预测动作。
7. 只在 `p>0` 时执行 A 新增随机采样。B、C 的 `p=0` 路径必须不消耗额外 RNG。

如果团队只想测试 RGB/IR 互补性，应保留现实现但将实验重命名为“visual-stream dropout”，不能把它解释为完整 A。

### 建议超参

- 确认性 A：branch dropout `p=0.10`，无 warmup，每样本保证至少一个自然可用分支。
- 仅在 A 通过后做探索性 follow-up：`p=0.15`；不要在同一 fold0 上同时比较 0.10/0.15 后挑最好者。
- 记录三分支各自 drop fraction、有效组合频率、按自然 availability pattern 的准确率。

## B：Same-Class Different-User SupCon Queue

### 对当前实现的判定

**核心数学实现基本正确，修复公平性后可作为第一项训练。**

- anchor 与 candidate 均做 FP32 L2 normalize（`scripts/train_r2plus1d.py:142`）。
- positive 定义为 same label 且 different user（`scripts/train_r2plus1d.py:146`）。
- same-label/same-user 项被从 denominator 排除，而不是错误地作为负样本（`scripts/train_r2plus1d.py:149`）。
- 无正样本 anchor 被跳过；queue 特征 detach 后入队，不会跨 batch 反传（`scripts/train_r2plus1d.py:153`、`scripts/train_r2plus1d.py:169`）。
- queue 仅由 `train_indices` 映射出的 user target 更新，当前没有 fold0 validation/test 标签进入 queue。

需要修正或补强：

1. **先修复 A 的 p=0 RNG 消耗**，否则 B 与 frozen baseline 的差异包含随机轨迹变化。
2. 当前全局 FIFO=2048，约覆盖 0.86 个训练 epoch；未限制 class/user 占比。样本频次 12–298，queue 会偏向高频类和高频用户。
3. 当前 batch=8 时，随机 batch 内仅约 22.2% anchor 有 same-class/different-user positive；queue 确有必要，但应记录 queue 稳定后的 active fraction，而非只验证 toy batch。
4. queue 表示来自持续变化的 online encoder，2048 长度会引入较陈旧 key；不建议在首轮再增加 momentum encoder，因为那会引入第二个科学变量。
5. loss 日志按 active-anchor mean 得到后又乘整个 batch size，最后除总样本数（`scripts/train_r2plus1d.py:505`），当 active fraction <1 时记录值被放大；训练梯度不受影响，但日志解释错误。
6. SupCon 仅约束 fused CLS，而最终分类有直接 `video_logits` bypass。必须明确这是“fusion representation regularization”，不能声称完全移除了视频主体特征。

### 建议损失与 queue 约束

- 正样本：`label_i == label_j && user_i != user_j`。
- 忽略项：`label_i == label_j && user_i == user_j`，包括跨 epoch 的同一样本。
- 负样本：所有不同动作类别。
- 无 positive 的 anchor 不进入平均；分母仅包含 positive 与合法 negatives。
- queue 只收训练样本；每个 fold/run 开始时清空，验证前不更新，不写入推理 checkpoint。
- 推荐全局 queue **1024**，并设每类最多 32、每个 `(class,user)` 最多 8；超限时淘汰最旧项。这样降低陈旧度并避免高频类占满 queue。
- queue key 保存 FP32 或 FP16 均可，但相似度、mask、`logsumexp` 必须 FP32。

### 建议超参

- `temperature=0.10`。
- `contrastive_weight=0.05`，而不是当前 0.10。当前 staged task loss 已很低，0.10 乘一个约为对数候选数尺度的 SupCon loss 容易与主 CE 同量级。
- 前 512 个 queue 项只入队不反传 SupCon；达到 512 后将权重在一个 epoch 内线性升到 0.05。
- 不增加 projection MLP，首轮直接使用现有 192 维 normalized fused CLS，以保证推理模型和参数严格不变。
- 必须记录 active-anchor fraction、每 anchor positive 数、queue class/user entropy、queue 最大年龄和未加权 task CE。

## C：Gradient-Reversal User Removal

### 对当前实现的判定

**阻断，不建议按当前无条件线性 user head 训练。**

GRL 本身的符号和 schedule 正确（`scripts/train_r2plus1d.py:122`、`scripts/train_r2plus1d.py:475`）。`subject_loss` 不乘权重而将权重放入 GRL coefficient，意味着 discriminator 始终以完整 CE 学习、encoder 接收缩放后的反向梯度，这是合法参数化，不是损失漏乘。

但当前实现存在以下关键风险：

1. 用户分类器只读取 fused CLS，不知道动作标签。由于 147/560 个 user-class 单元为空，它可通过动作内容预测用户，GRL 会反向删除动作判别信息。
2. user CE 未按用户或 `(class,user)` 平衡，训练样本数 133–203 的用户以及高频动作主导 adversarial gradient。
3. `train/subject_accuracy` 是 micro accuracy；在用户和类别条件不均衡时，`1/14` 不是有效 chance baseline。
4. 最终分类的 `video_logits` 绕过 fused CLS，单独约束 CLS 可能既伤害融合动作信息，又未充分移除主视频路径的用户特征。
5. gradient clipping 只覆盖 `model.parameters()`，不包含外置 `subject_head`（`scripts/train_r2plus1d.py:487`）。这不改变 encoder 梯度，但 discriminator 本身可能出现不受控更新。

### 建议的条件 adversary

采用训练标签条件化：

`subject_logits = D(GRL(z, lambda), stop_gradient(one_hot(action)))`

其中 `z` 是现有 normalized fused CLS，动作 one-hot 只进入 discriminator，不经过 GRL，user head 在推理时完全删除。这样目标变为移除给定动作后的 residual user information，而不是移除动作本身。

- user CE 使用 training-only `(action,user)` 频数的 inverse-sqrt 权重并归一化；不创建不存在的 user-class 组合。
- 主报告使用按动作条件宏平均的 user accuracy，而非普通 micro user accuracy。
- 首轮仍只约束 fused CLS；若它有效，再单独研究 video feature adversary，不能在同一次 C 中同时加两个 adversarial attach point。
- subject head 保持训练脚本外置，不进入 EMA 和最终 checkpoint；但 optimizer 和 gradient clipping 都应显式包含它。

### 建议超参

- 线性 head：输入 `192 + 40`，输出 14 个训练用户；首轮不加隐藏层。
- `lambda_max=0.05`。
- epoch 1 只训练 discriminator，GRL lambda=0；epoch 2–20 使用 DANN sigmoid schedule 升到 0.05。
- discriminator LR 与 fusion head 相同 `3e-4`，weight decay `1e-4`。
- 记录 task-only loss、conditional subject CE、conditional macro accuracy、GRL lambda 和 model/head 各自 gradient norm。

## 数据泄漏与公平比较

### 当前安全项

- `get_indices` 按用户完全隔离 fold0，并显式检查训练/验证用户无交集（`src/data.py:17`）。
- B 的动作标签和用户 ID、C 的用户 target 均可仅从 fold0 training records 构造；这属于训练监督，不是测试泄漏。
- 固定 evaluator 的 transition matrix 排除了当前 fold 用户，且 A/B/C 共用相同 base、X3D-L gate、ensemble 和 decoder 参数。

### 必须阻止的泄漏

- queue 不得加入 validation embedding、validation label、fold0 user target 或 test embedding；也不能使用 validation active fraction 调 queue 容量。
- adversary 的 user vocabulary 只能由 `train_indices` 构造；不得将 held-out/test 用户映射加入 user head，也不得按 held-out user accuracy 选 lambda。
- 所有 class/user 权重、queue cap 和正样本统计只能由训练 split 计算。
- 不得用 fold0 final soup/TTA/decoder 指标逐 epoch 选 checkpoint；checkpoint 仍按 clean single-view EMA raw accuracy选择。

### 不公平比较风险

1. **必须增加同代码 control。** 当前训练脚本和历史 baseline 的 source SHA 不同；新增 p=0 随机采样还改变模型 dropout RNG。先在同一脚本运行所有新参数为 0 的 fold0 control，确认 endpoint 与历史基线处于合理随机波动范围。
2. control、A、B、C 必须加载相同 visual initializer SHA，使用相同 train/valid indices、seed=42、batch order、epoch/schedule、EMA、patience 和 checkpoint 规则。
3. 每项只训练新的 32f staged 右端，再与固定 16f 左端按 0.25/0.75 soup；不得为某项重调 soup weight、temperature、gate threshold 或 decoder。
4. B/C 的训练专用对象不得进入 inference state dict。需报告训练额外参数量，但部署 checkpoint 必须仍可被原 `R2Plus1DClassifier` strict-load。
5. 当前 CLI 允许 A/B/C 同时非零。run wrapper 虽然一次只启用一个，但训练入口最好拒绝超过一个 generalization flag，以防误跑组合实验。
6. 三项方法和多个超参都复用 fold0，会产生多重比较偏差。确认性阶段每项只允许一个预注册配置；任何 grid 必须在训练用户内部再划 inner subject split，不能直接在 fold0 挑最佳。

## 选择指标审计

- 主选择指标应保持 standalone raw accuracy；V9 blend raw 和 decoder 只作为固定次级指标。decoder 使用重复轮次、唯一标签和 transition prior，非独立样本指标，可能掩盖 raw 回退。
- `scripts/evaluate_single_ablation.py` 的 “available-only” 只读取视觉 cache 的 `available.npy`（`scripts/evaluate_single_ablation.py:94`），实际含义是 **visual-available accuracy**，不是所有输入模态均可用。A 必须额外按 `[visual,skeleton,imu]` 八种自然 mask pattern 报告。
- fold0 每个样本约等于 0.1534 个百分点；decoder +2 样本仅约 +0.307 pp，且三项并行比较时容易由方差触发。晋级后 fold1 只能原样确认，不能再调参。
- micro accuracy 受 12–298 的类别频数影响。保留既定 gate 的同时，应附加 macro-class accuracy、user-macro、worst-user 和 user std 作为诊断，不得用这些指标事后替换主 gate。
- 固定 temperature=2 和 confidence gate 对概率校准敏感。B/C 可能改变 entropy，故应附加 NLL/ECE、gated row count；但不得为单个候选重调 temperature 或 0.15 gate。
- 当前训练 history 的 `train/loss` 在 C 中包含完整 subject CE，在 B 中包含 weighted SupCon，不能跨方法直接比较。必须另记 `task_loss`、`auxiliary_loss` 和正则损失。

## 可执行优先级

### P0：训练前阻断项

1. 将 `visual_channel_dropout=0` 路径改为不执行新增随机采样，并完成同代码 control。
2. 明确 A 是三分支 structured dropout，或将当前方案重命名为 visual-stream dropout。
3. 将 C 改为 action-conditioned、条件均衡 adversary。
4. 为训练入口增加“最多一个 generalization method 非零”的断言。
5. 增加真实模型级 mask 测试：同一输入改变非 shared `visual_mask` 时当前 logits 应不变；归零视觉分支并启用 joint visual mask 后 logits 应变化且 video residual 应为零。

### P1：确认性运行顺序

| 顺序 | 实验 | 预注册配置 | 当前可运行性 |
|---:|---|---|---|
| 0 | control | A/B/C 全 0；修复后脚本 | 必须先跑 |
| 1 | B | weight 0.05, temperature 0.10, queue 1024 + class/user cap | 修复 RNG/queue 后可跑 |
| 2 | A | 三分支 p=0.10，保证至少一个自然可用分支 | 需重写当前 A |
| 3 | C | conditional GRL, lambda max 0.05, epoch1 warmup | 条件化前禁止 |

B 优先于 A 的原因是其目标直接对应 same-action 跨用户聚合，且 40 类均至少分布在 3 个训练用户，监督条件成立；A 的上限受当前视觉 joint encoder 和极少的真实缺失样本限制。C 最后执行，因为 label-user support 不完整使其最容易产生负迁移。

### P2：统一交付

每个候选必须输出：

- 32f staged endpoint clean single-view raw 指标与所选 epoch。
- 固定 0.25/0.75 soup 的 4-view candidate NPZ，且 indices/labels 与 baseline 逐项一致。
- standalone raw、visual-available、固定 V9 blend raw、固定 decoder correct/accuracy。
- macro-class、每用户、worst-user、user std、自然 availability pattern、NLL/ECE。
- 初始 checkpoint SHA、源文件 SHA、训练专用参数量、最终 inference 参数量、SwanLab URL。
- 明确的 gate 结果；不通过则停止，不看 fold1。

## 最终建议

- **B：有条件放行，最高优先级。** 先修复 p=0 RNG 和 queue 均衡/日志，再按 `0.05 / 0.10 / 1024` 单配置运行。
- **A：退回重构。** 当前只是 RGB/IR stream dropout，不能回答真正视觉/传感器 structured dropout 的问题。
- **C：阻断。** 无条件 user GRL 在当前稀疏 user-class support 下高度可能删除动作信息；完成 action conditioning 后才可训练。
- **任何方法前先跑 control。** 否则无法区分方法收益与训练脚本漂移、随机序列变化。

本审阅未修改任何主文件，也未启动训练或 Kaggle 提交。
