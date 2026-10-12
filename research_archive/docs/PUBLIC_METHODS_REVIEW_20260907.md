# CUHK-X 小模型赛：公开方案调研与最小实验设计

日期：2026-09-07。状态：调研与实验提案，尚未运行本文的新实验。

## 1. 结论与证据边界

- 在本轮能访问的官方仓库、公开 Notebook、论文及实现中，**尚未找到同时具备可运行代码、可核验比赛高分、明确跨用户评估协议的高分开源方案**。这不等于不存在；Kaggle 部分页面抓取受限，检索不能覆盖所有讨论。
- 官方 CUHK-X 仓库提供单模态基线，包含 CNN、CNN-LSTM、CNN-GRU、ViT 等代码。README 中单模态多数超过 90% 的描述对应 **cross-trial**，不能当成本次 **cross-subject** 小模型赛的预期成绩。RGB 基线代码也不是本赛可直接使用的 RGB 输入路线。
- 检索到的 `avikds/Kaggle-Notebooks-Avik/cuhk-x-har-3dcnn-bigru.ipynb` 有 3DCNN + BiGRU 结构，但保存的执行输出为 “Mock data loaded.”，输入来自随机数，验证准确率 0.0250。它可作为结构示例，**不是已验证的强比赛方案**。
- 不采用其他参赛者的成品权重。论文和官方实现用于理解结构，不把论文在其他数据集的精度移植为 CUHK-X 的预期收益。

## 2. 当前项目到底证明了什么

现有 V3/V4 的跨用户验证加权准确率均为 70.52%，官方测试准确率未知，95% 尚未达到。旧流程包含 ImageNet/Kinetics 预训练分类器及 COCO 人体检测器，**不是用户现在要求的全流程从零训练基线**。详见 `docs/RESULTS_V4.md`、`docs/RESULTS_V5.md`。

已有负结果：骨架注意力版本、局部画面、全景版本、树模型及若干 IMU 时间处理没有形成稳定收益。TSM 也做过旧预训练路线的对照，未胜出；不能包装成全新、必涨的方向。部分历史对照同时改变了缓存、增强或输入，不能当成严格单因素消融。没有完成六个单模态与逐模态移除的系统实验，因此还不能声称“某模态无用”或“Transformer 不行”。

本轮检查 `data/metadata/samples.json`、`scripts/prepare_data.py` 和缓存：训练片段 3036 个；骨架、IMU、雷达的缓存有效数分别为 2931、2863、1409，即约 96.5%、94.3%、46.4%。这是当前解析器的有效掩码，不等于全部原始文件已人工确认可用；缺失原因须区分未采集、空文件与解析失败。

## 3. 按实际数据选择编码器

以下为待验证假设，不是已取得的实验结论。所有编码器随机初始化。

### 架构限制复核（2026-09-07）

重新检查了本地保存的 Kaggle 小赛道官方概览（`docs/competition_pages.json`）：明确写有 `architectures limited to CNN / RNN / Transformer`，模型大小不超过 100 MB，禁止大型预训练骨干。本次在线抓取未返回可核验正文，主办方官网直连另遇 TLS 错误，因此不声称已确认最新修订。不能仅因模型小、从零训练或某算子可用，就认定整套架构获准。

- 实验白名单收紧为常规 1D/2D/3D CNN、LSTM/GRU 以及 Transformer；TCN 使用时间卷积实现，DeepConvLSTM 使用 CNN + LSTM 实现。
- 不把 CatBoost、随机森林、SVM、独立 MLP 专家加入正式候选；GCN/ST-GCN 和 PointNet 类点集网络未获得主办方明确确认，暂不纳入。此处是保守排除，不声称主办方已逐一判定这些网络违规；尤其不能把图卷积等同于规则中的常规 CNN。
- 雷达第一版改为固定空间网格统计（占据、速度、信号质量及有效掩码）后送入常规 CNN + 时间卷积，网格范围等参数只由训练划分确定。PointNet 仅保留为研究参考，不通过改名或改写成 1×1 卷积来规避分类问题。
- 随机初始化与架构限制分别审计；所有分支及融合头都纳入检查。量化、BF16、压缩不自动豁免架构或预训练来源限制。
- 缓存概览明确禁止 `LLMs for development`，不只是禁止 LLM 标注；架构收紧不能解决本项目 AI 辅助开发的资格问题。主办方未书面澄清前，不宣称正式参赛合规。

| 模态 | 数据与处理边界 | 第一优先级编码器 | 有条件追加的对照 |
| --- | --- | --- | --- |
| IR | 灰度图；片段级一致的缩放、裁剪；先保留全景，检验是否丢失手部和物体线索 | 小型 2D 残差 CNN + TCN，保留时序输出 | 同一 CNN 换 GRU；再考虑 TSM |
| Thermal | 当前为三通道伪彩图，不假设像素值就是温度；先保留三通道，不做可能破坏含义的任意色彩增强 | 与 IR 相同的 CNN + TCN 架构，但独立训练权重 | 仅验证有动机的输入归一化，避免一次改多个因素 |
| Depth_Color | 当前是彩色深度图，不能假定存在可恢复的米制深度或相机标定 | 与另两种视觉模态一致的 CNN + TCN | 胜出的视觉模态再做轻量 R(2+1)D 对照；不直接套 HHA |
| Skeleton | JSON 为 17×3 关节坐标并带置信度；现有父节点拓扑不是 COCO17，须检查可视化。中心化、体型归一化，保留关节、骨向量、运动及置信度/缺失信息 | 多流 1D CNN/TCN，流之间晚融合 | PoseC3D 思路：明确坐标投影后生成关节热图，以轻量 3D CNN 编码；仅在预算允许时做 |
| IMU | 五个身体位置，含加速度、角速度、角度、磁场、四元数。按相对时间对齐，归一化仅拟合训练用户，保留设备有效掩码 | DeepConvLSTM 风格 CNN + LSTM；以多尺度 1D CNN 作便宜对照 | 原始坐标 vs 世界坐标的严格对照；已有相关负结果，不能盲目重做 |
| mmWave | CSV 为 timestamp/frame/点编号及 x,y,z,v,snr,noise，是点云而非原始 ADC/IQ。先检查空帧、离群点，生成固定空间网格统计并带掩码 | 常规 2D/3D CNN + TCN | 在同一 CNN 下比较网格分辨率或输入统计通道；PointNet 暂不纳入候选 |

注意：IMU 抽样约 10 Hz，不能凭重采样制造高频信息；骨架在均匀片段网格上的差分不能直接称为物理速度；点云不能直接当作原始雷达信号生成标准微多普勒谱。

所有视觉分支使用同一时间采样策略，视频内增强一致；空间/左右翻转须先检查动作标签语义。当前预训练人体检测框不能继续作为“端到端从零训练”的默认输入，应回到全景或固定、与标签无关的裁剪。给定骨架属于比赛输入，不等于自行使用预训练权重。

## 4. 最小实验矩阵：先回答问题，再增加参数

### 阶段 A：六个单模态，找出有效信号

固定已有的用户隔离划分、训练预算、采样、增强与种子，训练上述六个单模态编码器。缺失模态样本不参与该编码器的有效输入训练；单模态指标须报告覆盖率、有效子集分数及统一验证样本上的分数（缺失时采用仅由训练标签确定的固定先验）。另外报告共同有效子集结果，不能将不同验证子集的准确率直接排名。

先统一用小容量网络跑通和计时，不同时搜索容量、分辨率、帧数、损失、优化器。先测 1 个 epoch 的实际吞吐，再按剩余时间分配预算；不承诺尚未测量的完成小时数。

### 阶段 B：两个融合对照

1. 六路编码器 + 池化拼接 + 缺失掩码，建立完整但简单的参考。
2. 使用相同编码器初始化、训练数据及训练预算，只将融合头换成小型时序 token Transformer。统一片段时间区间、明确模态/时间标识和缺失 mask；控制头部参数规模并报告差异，不同时扩大编码器。

先检查同步元数据；不能把各模态独立插值后的同序号 token 自动称为严格同步。融合头比较使用相同可用时间信息。预训练各分支仅使用当前 fold 的训练样本，不加载其他 fold 或全量模型。

### 阶段 C：只对胜出方案做针对性消融

- 第一轮共 **6 个单模态 + 2 个融合 = 8 个配置**。随后先做已训练模型的推理遮蔽检查，仅作为敏感度诊断。
- 若时间允许，对最有信息价值的模态进行“移除后重新训练”；这才估计训练时的模态贡献。优先检验雷达是否提供互补信号、视觉三路是否冗余，而非穷举 63 个非空模态组合。
- CNN+TCN vs CNN+GRU/TSM/R(2+1)D，只在胜出的视觉输入上追加一个有明确假设的对照；PoseC3D 也是后续候选，不同时全开。
- 筛选以 Fold 0 为主，配方冻结后再进行 Fold 1 复核；现有两折已被多轮使用，结果不称为无偏最终测试。短名单才做 3 个种子，报告均值/波动、每用户及每类指标、宏平均 F1、混淆矩阵。
- SwanLab 按 `scratch_ablation` 分组记录配置、数据/划分版本、有效覆盖率、随机种子、验证指标、耗时、峰值显存、参数量和实际导出字节数。失败与负结果保留。

## 5. 精度、容量和参赛边界

训练脚本已经使用 BF16 autocast；这不等于 checkpoint 已按 BF16 保存。纯 BF16 参数的理论开销是每参数 2 字节：100 MB 大约只容纳 5000 万个参数，且尚未计算 buffer、序列化开销和其他推理模型。建议第一版总参数控制在约 2000 万至 3000 万，导出逐项实测并留余量；这是工程预算，不是最优容量结论。

先建立可靠小模型，再决定是否需要训练更大的、同样从零开始的教师进行蒸馏。不要一开始就把“大网络 + 压缩 + 全部消融”变成必做链条。最终以全部推理所需模型文件的实际大小核验，BF16 转换前后也要比较预测。

2026-09-07 复核本地规则快照：Kaggle 小赛道概览限制 CNN/RNN/Transformer，并禁止 LLM 参与开发；本次未成功获取在线正文，最新条款仍待核实。旧规则页标题存在大小赛道不一致的问题，不能因此忽略概览中的限制。当前为 AI 辅助研究，不宣称具备参赛资格；正式提交前应向主办方确认适用规则，不自动上传模型或预测。

## 6. 一手来源

以下链接供复核方法或公开执行记录，不构成在本赛高分的证据。

1. 官方数据集及基线：https://github.com/openaiotlab/CUHK-X
2. 官方小模型代码目录：https://github.com/openaiotlab/CUHK-X/tree/main/SM/Code
3. 官方 CNN/RNN/ViT 示例：https://raw.githubusercontent.com/openaiotlab/CUHK-X/main/SM/Code/rgb/train_models_cross_multi.py
4. 公开 3DCNN + BiGRU Notebook（已检查随机数据执行输出）：https://raw.githubusercontent.com/avikds/Kaggle-Notebooks-Avik/main/cuhk-x-har-3dcnn-bigru.ipynb
5. TSM，ICCV 2019：https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html
6. TSM 作者实现：https://github.com/mit-han-lab/temporal-shift-module
7. R(2+1)D，CVPR 2018：https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html
8. PoseC3D，CVPR 2022：https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html
9. DeepConvLSTM，Sensors 2016：https://www.mdpi.com/1424-8220/16/1/115
10. InceptionTime 作者实现：https://github.com/hfawaz/InceptionTime
11. PointNet，CVPR 2017：https://openaccess.thecvf.com/content_cvpr_2017/html/Qi_PointNet_Deep_Learning_CVPR_2017_paper.html
12. 比赛：https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track （本地规则快照：`docs/competition_pages.json`）
13. 主办方官网（本次在线正文未成功获取，供后续核实）：https://openaiotlab.github.io/CUHK-X-Challenge/
