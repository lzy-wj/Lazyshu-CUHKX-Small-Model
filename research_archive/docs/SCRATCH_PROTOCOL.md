# 从零训练：CNN / RNN / Transformer 白名单实验

日期：2026-09-07。用户确认可使用 AI 辅助开发，按用户确认的口径继续本地实验；架构限制不放宽，不自动上传 Kaggle。

## 实现和硬限制

- 新训练命令必须带 `--strict-scratch`，调度器强制添加该选项。模型入口是 `src/scratch_models.py`，不实例化旧模型类。
- 只使用常规 CNN、LSTM 和 Transformer 编码器。网络内的线性分类层、归一化、池化、激活不作为独立专家模型；没有独立 MLP、图网络、PointNet、树模型或 SVM 分支。
- 拒绝外部预训练、任意 checkpoint 初始化、预训练检测裁剪以及缓存的预训练视觉特征；所有分支随机初始化。旧代码和结果只为复核保留，不进入这轮实验。

后续用户批准的编码器研究增加了受严格审计的 `--encoder-sources` 专用入口，仅加载同折、独立从零训练的单模态编码器；通用 `--init` 仍被禁止。这个扩展不改变首轮八个实验的随机初始化事实，详见 `docs/ENCODER_STUDY.md`。
- CNN / RNN / Transformer 同样受到实际推理权重文件合计 100 MB 的预算限制。本轮模型先维持小容量，以后才能在固定输入下做容量对照。

## 六个单模态与两个融合

| 输入 | 编码器 | 参数量（含分类头） |
| --- | --- | --- |
| Skeleton | 关节、骨向量、运动三流 1D CNN | 401168 |
| IMU | 1D CNN + 双向 LSTM | 334640 |
| Radar | 现有逐帧点云统计 + 时间 CNN | 292304 |
| IR / Thermal / Depth_Color | 每模态独立的残差 2D CNN + 时间 CNN | 各 1381520 |
| 全部六路，拼接分类 | 相同六路编码器，池化拼接、缺失标记、分类头 | 5089912 |
| 全部六路，Transformer 融合 | 相同六路编码器，16 token/模态，2 层 Transformer，4 头 | 5317816 |

模型序列化 state_dict 约为 1.18–21.39 MB；训练结束还会测量含配置的 `best.pt` 实际大小。训练使用 BF16 autocast，当前权重仍保存 FP32，不能将训练精度当成导出精度。

输入复用未经过预训练检测器的全景缓存 `video_24_160`，从中均匀采样 16 帧。新增 `scene_depth` 只读取深度三通道，不和 IR 混为一个模态。

这批是数据基线，不声称已完成前期调研中的全部预处理：雷达仍用 28 维帧统计，不是空间网格 CNN；骨架仍用现有位置/骨向量/差分缓存，未重新提取置信度；IMU 仍用已有缓存。先量化现有信号，再逐项改变数据表示，避免同时改变数据和架构。

## 可比性与缺失数据

- 固定 Fold 0：验证用户 3、8、18、23，训练 2384 个片段、验证 652 个片段；官方测试数据不用于选择模型。
- 每个候选训练 60 epoch、batch size 24、学习率 0.001、随机种子 42，每 2 epoch 验证。不同单模态的有效训练集大小不同，记录覆盖率，不声称其优化步数一致。
- 两个融合模型的编码器初始权重逐张量核验一致。独立固定 DataLoader 随机生成器，并按 epoch/batch 固定 GPU 增强种子，避免融合头初始化影响数据顺序。
- 不做混合样本、左右翻转或强增强。沿用片段内一致的轻量亮度/平移及传感器噪声，融合时使用缺失模态 dropout。
- 训练剔除所选模态全部缺失的样本，但验证保留完整 652 个样本；完全缺失时使用仅由本 fold 训练标签计算的平滑类别先验。
- 每次保存最佳模型时报告完整验证、输入有效子集、六模态共同有效子集、每用户、每类、宏平均 F1、混淆矩阵及覆盖率。宏平均 F1 默认包含标签或预测中出现的类别，另存固定 40 类版本；Fold 0 验证集缺少类别 25，不能把无样本类误判为该模型识别失败。
- 时间 token 仅代表归一化片段位置，不声称各传感器的物理时间戳已经严格同步。
- 全部缺失时的输出、部分模态缺失时对该输入内容的不变性都经过检查；CNN 使用 GroupNorm，避免缺失样本改变其他样本的 BatchNorm 统计。
- Fold 0 已经参与历史研究，本轮也是探索验证，不是无偏最终测试。筛选之后再决定 Fold 1 复核和重训；不自动以这轮分数替换旧结果。

## 运行与产物

配置：`configs/scratch_ablation.json`。只调度 GPU 2、3、7，每次检查显存并由原启动器取得 GPU 文件锁；不触碰其他进程。

```bash
conda run --no-capture-output -n Train python scripts/audit_scratch.py
conda run --no-capture-output -n Train python scripts/run_scratch_study.py
```

CPU 审计检查八种配置、非法配置拒绝、梯度、掩码、重载一致性、划分和旧产物哈希；另有两次 CUDA BF16 的双 batch 冒烟训练，名称为 `scratch_smoke_concat` / `scratch_smoke_transformer`。冒烟分数没有研究意义，不纳入实验排名。

- SwanLab 沿用私有项目 `cuhkx-small-multimodal`，正式实验名称前缀 `scratch_v1_`，配置字段 `study=scratch_ablation`。
- 审计与动态汇总：`outputs/study_scratch/audit.json`、`outputs/study_scratch/report.json`。
- 每个实验独立保存 `config.json`、`split.json`、`history.jsonl`、`best.pt`、`validation.npz`、`ablation_metrics.json` 和成功时才写入的 `completion.json`。
- 失败不静默覆盖或重试；调度器停止启动新任务，等待已运行任务结束并写失败状态。
- 初批共八个配置；未自动增加容量、做63种模态组合、全量重训或提交。
