# 空间 token 受控实验协议

## 问题与对照

研究问题不是“Transformer 是否更高级”，而是提前空间池化是否限制了局部动作表示，以及直接 patch 输入能否替代 CNN。

| 编码方式 | 每帧局部 token | 视频 token | 总参数 | 空间/时间 FFN 宽度 |
| --- | ---: | ---: | ---: | ---: |
| cnn_pooled | 4 | 64 | 3,018,496 | 384 |
| cnn_tokens | 100 | 1600 | 3,018,496 | 384 |
| patch_tokens | 100 | 1600 | 3,029,248 | 1344 |

CNN 两组使用完全相同的可训练模块，唯一结构差异为是否在 Transformer 前池化到 2×2。保留空间组使用 CNN 自身输出的 10×10 特征，不做额外空间池化。

Patch 组使用 kernel=stride=16 的卷积实现非重叠图像块线性投影，没有 CNN 骨干。FFN 更宽以匹配总参数，差异 0.36%；因此该组是容量匹配的架构对照，不是只替换一个层的单因素实验。

三个视觉分支均进行空间、时间分解的 Transformer 编码。局部 token 保留至跨模态 Transformer：16 个查询直接关注全部视觉、骨架和 IMU token，最后才汇聚分类。不对 1600 个视觉 token 再做全局二次方自注意力。该查询式融合是共同的计算控制，不是逐像素无损处理。

cnn_pooled 是本轮匹配对照，不是旧的 CNN+TCN 编码器；不能把两者的差异只归因于池化。

## 固定条件

- Skeleton、世界坐标 IMU、运动区域 Thermal，三个模态与输入缓存完全相同。
- 16 帧、160×160、batch 24、60 epoch、AdamW、学习率 0.001、5 epoch warmup、BF16；不提前停止。
- 相同数据增强、损失、EMA、CNN/RNN 传感器分支、查询式跨模态 Transformer。
- 所有权重重新初始化，无外部预训练、无旧编码器初始化。
- 各折 seed42；参数与优化步数匹配，但 FLOPs 和实际时间不匹配。单种子不能证明稳定优越性。
- 架构仅 CNN/RNN/Transformer；BF16 用于计算，FP32 权重实际序列化约 12.2MB，远低于 100MB。最终仍以 checkpoint 和整个提交包实测大小为准。

## 用户隔离与选型

折 0：训练 1693、内层 691、外层 652 片段。折 1：训练 1647、内层 684、外层 705 片段。

每折内三组数据一致，train/inner/outer 的用户互不重叠。内层用户通过固定随机种子、仅使用标签覆盖和片段数量条件选取；没有读取模型分数。每个 train/inner 分区均覆盖 40 类。

每两个 epoch 仅评估内层。训练结束后，加载内层选定的 checkpoint，才评估一次干净训练和外层。不让外层曲线影响训练或早停。

每折单独按其内层 accuracy 选架构，平分时按 macro-F1，再按配置顺序。外层汇总使用各折的局部选择。跨折之间用户有复用，因此不能先汇总所有内层选一个共同赢家，再宣称它在两折外层上的结果无偏。

此前外层折已暴露于历史研究；本轮防止新增的 epoch 选择泄漏，并不把历史使用抹掉。真实官方测试集成绩仍未知。

## 检查与输出

训练前检查分区、错误划分拒绝、真实输入、BF16 前反向、缺失模态 NaN 屏蔽、全缺失先验回退、视觉参数梯度以及序列化大小。

GPU 只允许 2/3/7；启动前同时检查显存、进程与项目文件锁，不干扰其他用户任务。通过 tmux 运行，SwanLab 在线记录。

- 冻结配置：`configs/spatial_token_study.json`
- 实现：`src/spatial_tokens.py`、`src/scratch_models.py`
- 验证划分：`configs/spatial_grouped_f0.json`、`configs/spatial_grouped_f1.json`
- 审计：`outputs/study_spatial/audit.json`
- 队列：`outputs/study_spatial/report.json`
- 汇总脚本：`scripts/summarize_spatial_study.py`

除整体 accuracy/macro-F1，还报告逐用户、全身动作与其余类别表现，以及相对池化对照的配对用户聚类区间。区间仅基于八个外层用户、固定训练种子，不能替代重复训练。

本轮不自动覆盖旧最终模型或提交 Kaggle。保留所有失败结果，若保留 token 没有稳定收益，就不能继续把提前池化当作已证实的主要瓶颈。

## 固定复验

在查看首轮外层汇总之前，追加 seed43 对全部三组、两个划分的复验，不根据首轮优劣删减架构。保持全部设置和划分不变。复验配置 `configs/spatial_token_study_s43.json`，队列 `outputs/study_spatial_s43/report.json`。

两个种子的单模型结果与概率平均结果分别报告。最终嵌套选型仍在每折自己的内层完成，不能从外层选择种子或架构。复验后也不把两个种子当作充分的统计保证。

## 表示与融合的诊断边界

六个训练侧示例的原图/缓存对照保存在 `outputs/model_analysis/spatial_input_contact.png`。其中阅读/翻页示例的手部和纸张在热成像输入中很小；这提示应检查可观测细节，但六个中间帧不能证明全数据的细节已经不可恢复，也不能替代动作视频检查。

保留 token 还改变了跨模态 attention 的 token 数量比例：视觉占全部 memory 的比例从 64/97 变成 1600/1633。可学习的注意力并不一定服从这个比例，但这提示不能把所有效果都归于空间分辨率。

补充的 `scripts/diagnose_spatial_fusion.py` 在内层验证上检查模态屏蔽敏感性和最后一层 attention 分配；不据此修改已冻结的主实验。Attention 权重不是因果贡献，推理屏蔽也不是重新训练消融。
