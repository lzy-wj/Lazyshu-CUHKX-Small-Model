# 第二轮建模结果

记录时间：2026-09-07T00:26:12+08:00

## 目标状态

**没有达到约 95% 的目标。** 本轮完成了建模改进、用户隔离验证、全量重训与 405 条测试预测，但没有测试标签或新的 Kaggle 分数，不能声称测试准确率提高到某个数值。

| 验证划分 | 第一轮 | 第二轮 | 片段数 |
| --- | --- | --- | --- |
| Fold 0 | 54.60% | 60.58% | 652 |
| Fold 1 | 53.33% | 66.10% | 705 |

第二轮两组按样本量加权：63.45%；第一轮：53.94%。
各组均为用户隔离，两组各覆盖 39 类，合起来覆盖全部 40 类。
结构、融合权重和温度在 Fold 0 上选择，固定后应用 Fold 1；每组模型 checkpoint 仍按本组验证选择，因此不是无偏的最终测试估计。
Fold 0 修正旧错误 72 个、引入新错误 33 个；Fold 1 对应 129 / 39 个。

## 改进内容

- 对 IR 作对比度增强；用 SSDLite-MobileNetV3 的人体框分别裁剪 IR、Thermal，并把 IR 框应用到 Depth。
- 人物区域保留 224×224 分辨率，IR、Thermal、Depth 独立视觉编码，结合骨架与 IMU。
- 将 IMU 加速度与角速度按四元数旋转到世界坐标，减少佩戴方向变化；对比统计时序编码和卷积编码。
- 冻结增强视觉特征没有产生收益，未默认用于最终方案；BatchNorm 重估也没有带来收益。
- 所有预处理均不使用测试活动标签。检测器、外部初始化的合规性仍需主办方确认。

## 交付

- 新预测：`outputs/final_v2/research_multimodal_v2.csv`，405 行，未自动提交。
- 第一轮文件原样保留在 `outputs/final/`，没有覆盖。
- 分类模型与检测器全部计入体积：**62.63 MB**，低于 100 MB。
- 模型哈希、大小、相对路径：`outputs/final_v2/manifest.json`。
- 概率：`outputs/predictions/research_multimodal_v2.npz`。
- 相比第一轮，405 个测试预测中有 154 个变化；变化本身不代表这些预测正确。
- SwanLab 私有项目 `shu_123/cuhkx-small-multimodal`，本轮实验均为 `v2_` 前缀。

融合模型顺序：['v2_final_person_multi', 'v2_final_person_ir', 'v2_final_statistics']；权重：[0.5, 0.2, 0.3]；温度：[1.156757778331949, 1.2146341263961762, 1.1338148770999141]。
最终模型都使用全部 3036 个训练片段，依据验证最优 epoch 折算全量训练更新次数，具体命令在 `configs/v2_final_*.json`。

## 复现

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n Train python scripts/prepare_imu_world.py
CUDA_VISIBLE_DEVICES=2 conda run -n Train python scripts/prepare_person_crops.py --stage detect
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n Train python scripts/prepare_person_crops.py --stage crop
CUDA_VISIBLE_DEVICES=2 conda run -n Train python scripts/predict.py --checkpoints outputs/final_v2/models/v2_final_person_multi.pt outputs/final_v2/models/v2_final_person_ir.pt outputs/final_v2/models/v2_final_statistics.pt --weights 0.5 0.2 0.3 --temperatures 1.156757778331949 1.2146341263961762 1.1338148770999141 --name research_multimodal_v2_recheck
```

检测器原始权重缓存位于 `.tools/torch/hub/checkpoints/`，副本也在本轮模型包。未从缓存直接假装省掉检测器体积。
CPU/GPU 并行执行不保证跨环境逐位复现；项目内会使用打包权重再次推理检查。

## 本轮实验

| 实验 | 本地 Accuracy | 最佳 Epoch |
| --- | --- | --- |
| v2_features_f0 | 38.50% | 18 |
| v2_imu_world_f0 | 51.38% | 94 |
| v2_multimodal_f0 | 45.86% | 4 |
| v2_person_ir_f0 | 55.67% | 30 |
| v2_person_ir_f1 | 56.60% | 46 |
| v2_person_multi_f0 | 57.82% | 12 |
| v2_person_multi_f1 | 59.57% | 48 |
| v2_person_thermal_f0 | 53.07% | 46 |
| v2_statistics_f0 | 53.07% | 106 |
| v2_statistics_f1 | 52.34% | 174 |

## 错误诊断

以下统计仅来自两组本地验证，共 1357 个片段，不涉及测试标签。每类支持数和召回率见 `outputs/final_v2/validation_classes.csv`。

| 真实动作 | 最常见错误预测之一 | 次数 |
| --- | --- | --- |
| Read_documents | Turn_pages | 19 |
| Mop_the_floor | Sweep_the_floor | 8 |
| Sweep_the_floor | Mop_the_floor | 7 |
| Pour_drinks | Take_and_use_tableware | 7 |
| Take_and_use_tableware | Pour_drinks | 7 |
| Take_off_clothes | Put_on_clothes | 7 |
| Stir_drinks | Pour_drinks | 6 |
| Eat_food | Play_games | 6 |
| Drink_water | Eat_food | 6 |
| Do_stretching_exercises | Do_jumping_jacks | 5 |

缓存形状、数值有限性、用户隔离及旋转算术检查通过，记录在 `outputs/final_v2/data_audit.json`。

## 下一步需要的证据

要判断线上差距，需要具体 Kaggle 分数及其对应 CSV。当前能验证的是本地用户隔离成绩，不会以训练准确率或随机帧划分成绩代替 95% 目标。
细粒度手部动作、物体识别、稀有类别与跨用户分布偏移是后续待验证的改进方向，不能仅靠增加空闲 GPU 就承诺从当前水平达到 95%。
官方 `no LLMs for development` 限制仍未澄清；这套 AI 辅助研究实现不宣称参赛合规，也不会代为提交。
