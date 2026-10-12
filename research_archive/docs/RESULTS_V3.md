# 第三轮研究结果

记录时间：2026-09-07T09:34:48+08:00

## 95% 目标仍未完成

用户要求的是接近 95%，不是只要比前一轮高。本轮所有分数仍为本地用户隔离验证；没有测试标签或本轮新预测的线上成绩，不声称测试达标。

| 划分 | 第二轮 | 第三轮 |
| --- | --- | --- |
| Fold 0 | 60.58% | 66.87% |
| Fold 1 | 66.10% | 73.90% |

按 1357 个验证片段加权：63.45% → 70.52%。
融合权重与温度仅在 Fold 0 上选择，然后固定应用 Fold 1；各模型 checkpoint 仍按各自验证选取，所以不是无偏测试估计。

## 为什么之前不够

- 第二轮统计传感器模型的干净训练准确率为 96.14%，同组用户隔离验证为 53.07%。训练本身超过 95% 没有解决跨用户泛化。
- 细粒度动作失分明显；仅增加轮数、扩大空间池化或叠加统计特征，并没有得到相应收益。
- TSM 与高空间分辨率池化对照明显落后于新视频模型，主动停止，不把被中断的实验说成正常完成。
- 检查了真实时间 IMU、胸部参考系相对姿态及二段/四段时序统计；未证明有益的路线不默认纳入交付。
- 新的主要视频候选使用 X3D-S 的小型 3D 视频编码器，显式记录 Kinetics-400 外部初始化。该来源的竞赛合规性仍需澄清。

## 交付与复现

- 预测：`outputs/final_v3/research_multimodal_v3.csv`，405 行，未自动提交。
- 全部推理权重（含人体检测器）：93.90 MB。前两轮文件保留，第二轮 artifact 哈希复核未变化。
- 打包分类权重重新加载后，CSV 完全一致，概率最大绝对差 0.0。
- 配置：`configs/v3_selected_ensemble.json`；新模型全量重训配方：`configs/v3_final_*.json`。复用的第二轮模型已经用全部 3036 个训练样本训练，不重复浪费计算。
- 依赖：`configs/video-requirements.txt`，使用 `pip install --no-deps --target .tools/python -r configs/video-requirements.txt`；共享 Train 环境没有升级。
- 核验：`outputs/final_v3/manifest.json`、`outputs/final_v3/data_audit.json`。
- SwanLab 私有项目 `shu_123/cuhkx-small-multimodal`，本轮前缀 `v3_`。

融合模型：['v3_final_x3d', 'v2_final_person_multi', 'v2_final_person_ir', 'v2_final_statistics']；权重：[0.6000000000000001, 0.19999999999999996, 0.07999999999999999, 0.11999999999999997]；温度：[0.9736012920941253, 1.156757778331949, 1.2146341263961762, 1.1338148770999141]。

先确认显卡空闲，再从项目根目录复现（下例使用 GPU 3）：

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 conda run -n Train python scripts/predict.py --checkpoints outputs/final_v3/models/v3_final_x3d.pt outputs/final_v3/models/v2_final_person_multi.pt outputs/final_v3/models/v2_final_person_ir.pt outputs/final_v3/models/v2_final_statistics.pt --weights 0.6000000000000001 0.19999999999999996 0.07999999999999999 0.11999999999999997 --temperatures 0.9736012920941253 1.156757778331949 1.2146341263961762 1.1338148770999141 --name research_multimodal_v3_recheck
```

## 所有实验

| 实验 | Fold Accuracy | 最佳 Epoch | 状态 |
| --- | --- | --- | --- |
| v3_pyramid_f0 | 52.30% | 110 | 完成 |
| v3_relative_f0 | 49.39% | 50 | 完成 |
| v3_relative_pyramid_f0 | 50.31% | 42 | 完成 |
| v3_sensor_both_f0 | 52.30% | 116 | 完成 |
| v3_shift_f0 | 51.69% | 32 | 主动停止（保留最佳权重） |
| v3_spatial_f0 | 53.53% | 22 | 主动停止（保留最佳权重） |
| v3_timed_f0 | 50.77% | 16 | 完成 |
| v3_timed_pyramid_f0 | 50.15% | 24 | 完成 |
| v3_x3d_f0 | 66.10% | 44 | 完成 |
| v3_x3d_f1 | 72.62% | 32 | 完成 |

## 仍在失分的类别

以下为两组验证合并后的低召回类别（仅展示支持数至少 10 的类别）；全部 40 类的前后对比见 `outputs/final_v3/validation_classes.csv`。没有使用测试标签。

| 类别 | 验证片段数 | 召回率 |
| --- | --- | --- |
| Write | 22 | 18.18% |
| Use_a_mobile_phone | 10 | 30.00% |
| Take_medicine | 36 | 30.56% |
| Massage_oneself | 21 | 33.33% |
| Take_body_temperature | 24 | 33.33% |
| Make_a_phone_call | 26 | 38.46% |
| Read_documents | 44 | 40.91% |
| Turn_pages | 24 | 41.67% |

## 未解决事项

95% 仍不是已达到或能保证的结果。已找到新版格式的本机 Kaggle 登录凭据并只读查询提交历史，结果保存于 `outputs/predictions/kaggle_submission_history.json`；没有上传本轮预测，不要求用户发送密钥。要判断线上效果，仍须本轮 CSV 对应的实际成绩。
2026 年 9 月 7 日公开榜 API 快照显示前列已有超过 95% 的公开成绩，见 `docs/leaderboard_snapshot.json`；这说明目标有公开榜参照，但别人的公开榜分数不等于本模型的测试成绩，也不代表最终私榜或合规审查结论。
比赛的 `no LLMs for development` 限制仍未澄清；本项目是 AI 辅助研究实现，不宣称参赛合规。
