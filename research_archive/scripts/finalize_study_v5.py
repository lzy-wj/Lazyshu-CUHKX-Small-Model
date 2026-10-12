import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from finalize_v2 import copy_verified


ROOT = Path(__file__).resolve().parents[1]


def main():
    selection = json.loads((ROOT / "outputs/predictions/v5_selection.json").read_text())
    if selection["promote_to_confirmation"]:
        raise RuntimeError("A V5 candidate passed the gate; finish its confirmation/refit instead of closing the study")
    names = ["v5_minit_f0", "v5_tree_f0", "v5_scene_f0", "v5_norm_f0"]
    results = {}
    for name in names:
        assert (ROOT / f"outputs/logs/{name}.exit").read_text().strip() == "0"
        results[name] = json.loads((ROOT / f"outputs/checkpoints/{name}/metrics.json").read_text())
    source_accuracy = json.loads((ROOT / "outputs/checkpoints/v3_x3d_f0/metrics.json").read_text())["accuracy"]
    if results["v5_norm_f0"]["accuracy"] >= source_accuracy + 0.01:
        raise RuntimeError("Normalization fine-tuning passed its gate; run corresponding fold-1/full-data continuation")
    normalization = json.loads((ROOT / "outputs/checkpoints/v5_norm_f0/config.json").read_text())
    assert normalization["init_source_fold"] == normalization["fold"] == 0
    assert normalization["init_source_name"] == "v3_x3d_f0"
    assert normalization["init_loaded_tensors"] == normalization["init_total_tensors"]
    source_train = json.loads((ROOT / "outputs/checkpoints/v3_x3d_f0/split.json").read_text())["train"]
    target_train = json.loads((ROOT / "outputs/checkpoints/v5_norm_f0/split.json").read_text())["train"]
    assert set(source_train).issubset(target_train)
    tree_metadata = json.loads((ROOT / "data/metadata/tree_features.json").read_text())
    feature_path = ROOT / "data/processed/tree_features.npy"
    assert hashlib.sha256(feature_path.read_bytes()).hexdigest() == tree_metadata["sha256"]
    features = np.load(feature_path, mmap_mode="r")
    assert features.shape == (3441, 8776) and not np.isinf(features).any()
    assert np.isin(features[:, -4:], [0, 1]).all()
    offset = 0
    for mask_index, channels in enumerate([153, 408, 85, 85]):
        block = features[:, offset:offset + channels * 12]
        present = features[:, -4 + mask_index] > 0
        assert np.isfinite(block[present]).all() and np.isnan(block[~present]).all()
        offset += channels * 12
    for version in ["v3", "v4"]:
        manifest = json.loads((ROOT / f"outputs/final_{version}/manifest.json").read_text())
        for artifact in manifest["artifacts"]:
            assert hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    v4_manifest = json.loads((ROOT / "outputs/final_v4/manifest.json").read_text())
    recommended = v4_manifest["recommended_research_bundle"]
    baseline = json.loads((ROOT / "outputs/predictions/v3_ensemble_validation_report.json").read_text())
    v4 = json.loads((ROOT / "outputs/predictions/v4_ensemble_validation_report.json").read_text())
    bundle = ROOT / "outputs/study_v5"
    bundle.mkdir(parents=True, exist_ok=True)
    for source, target in [("configs/v5_protocol.json", "protocol.json"),
                           ("outputs/predictions/v5_selection.json", "selection.json"),
                           ("outputs/predictions/v5_tree_selection_check.json", "tree_selection.json"),
                           ("data/metadata/tree_features.json", "tree_features.json")]:
        copy_verified(ROOT / source, bundle / target)
    report = {"completed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
              "experiments": results, "recommended_research_bundle": recommended,
              "target_achieved": False, "test_accuracy": None, "new_csv_created": False,
              "new_candidates_passed_confirmation_gate": False,
              "audit": {"tree_features_missing_masks_and_hash": "passed", "warm_start_same_fold_and_complete_state": "passed",
                        "v3_v4_artifact_hashes_unchanged": True},
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ["src/data.py", "src/models.py", "scripts/train.py", "scripts/train_from_recipe.py",
                                             "scripts/prepare_tree_features.py", "scripts/train_tree.py", "scripts/run_v4_stage2.py"]},
              "protocol": "Negative candidates stop after fold-0 screening; no invented fold-1 or official test scores."}
    (bundle / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    table = [f"| {name} | {values['accuracy']:.2%} | {values['epoch']} |" for name, values in results.items()]
    text = f"""# 本轮追加研究结论

记录时间：{report['completed_at']}

## 结论

**本轮没有得到稳定超过旧版的方案。推荐仍按 `{recommended}` 执行，95% 目标未达到，官方测试准确率未知。**

| 已完整复核的融合 | Fold 0 | Fold 1 | 按样本数加权 |
| --- | --- | --- | --- |
| V3 | {baseline['fold0']['accuracy']:.2%} | {baseline['fold1']['accuracy']:.2%} | {baseline['weighted_accuracy']:.2%} |
| V4 | {v4['fold0']['accuracy']:.2%} | {v4['fold1']['accuracy']:.2%} | {v4['weighted_accuracy']:.2%} |

V4 选择集上升，但确认组下降；没有把局部涨分或训练高分冒充总体进步。V4 已全量重训、生成 405 行候选预测并完成打包权重复核，但不替换推荐的旧版。
V3 在扩展后的代码下重新预测，CSV 与概率均保持一致；两版模型及 CSV 的 SHA256 再次校验通过。

## 追加对照

| 候选 | Fold 0 Accuracy | 最佳 Epoch/树轮数 |
| --- | --- | --- |
{chr(10).join(table)}

- M 初始化：目标网络仍是 13 帧、160px 输入的自适应池化编码器。该结果不能代表原生 16 帧、224px X3D-M 的表现。
- 树模型：逐片段提取规范化骨架、几何、原始/世界 IMU 的统计和相对频谱特征。没有按全数据拟合归一化，也没有使用用户编号或测试标签。替换旧统计专家的九点融合检查最高只有 66.41%，未超过已有融合。
- 全景视频：保留整张 IR/Thermal 画面，以检验桌面/道具上下文。但其缓存帧数和 IR 增强也与人物裁剪路线不同，因此不是只改变裁剪区域的严格单因素实验。
- 归一化适配：仅从同一 Fold 0 的 V3 权重开始，低学习率微调 12 轮并开放视频 BatchNorm；初始化样本集合和全部权重加载经过核对。没有使用其他 fold 或全数据模型初始化验证模型。

视频对照只在 Fold 0 做预先限定的双模型 21 点权重搜索，旧统计专家固定 0.15。最高融合为 {selection['candidates'][0]['accuracy']:.2%}，未达到预先设定的 {selection['baseline_fold0_accuracy'] + selection['minimum_improvement']:.2%} 确认门槛，故不继续消耗资源进行这些候选的 Fold 1 / 全量重训，也不生成名不副实的 V5 最终 CSV。

## 文件与复核

- 当前推荐指针：`configs/recommended_research.json`。
- V4 完整报告：`docs/RESULTS_V4.md`；V5 负结果、门槛与审计：`outputs/study_v5/`。
- 各候选权重、配置、验证预测与退出码仍在 `outputs/checkpoints/`、`outputs/logs/`；没有删除失败路线来美化结果。
- 调度器曾在 CUDA 显存尚未分配时误判空闲，但 GPU 文件锁拦截了冲突。一个尚未完成第 1 轮的全量任务被连带中断，记录保存在 `outputs/interrupted/v4_final_jointstats_20260907T1042/`；修复后干净重训成功。
- SwanLab 原私有项目继续保留所有研究记录。模型、数据和预测均未自动上传 Kaggle。

规则中的 `no LLMs for development` 及外部预训练许可范围仍需澄清；这是 AI 辅助研究，不宣称比赛合规。公开榜其他队伍的分数不能当成我们的测试成绩。
"""
    (ROOT / "docs/RESULTS_V5.md").write_text(text)
    (bundle / "RESULTS.md").write_text(text)
    recommended_manifest = json.loads((ROOT / recommended / "manifest.json").read_text())
    csvs = list((ROOT / recommended).glob("*.csv"))
    submission = next(path for path in csvs if path.name.startswith("research_multimodal"))
    pointer = {"bundle": recommended, "prediction_csv": str(submission.relative_to(ROOT)),
               "csv_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
               "model_bytes": recommended_manifest["total_inference_model_bytes"], "target_achieved": False,
               "test_accuracy": None, "reason": "V4 failed the fold-1 non-regression gate; additional bounded studies failed fold-0 promotion gates.",
               "updated_at": report["completed_at"]}
    (ROOT / "configs/recommended_research.json").write_text(json.dumps(pointer, indent=2) + "\n")
    import swanlab
    swanlab.init(project="cuhkx-small-multimodal", name="v5-study-summary", public=False,
                 log_dir=str(ROOT / "outputs/swanlog"), config=pointer)
    swanlab.log({f"fold0/{name}": values["accuracy"] for name, values in results.items()} |
                {"retained/weighted_validation": baseline["weighted_accuracy"], "target/achieved": 0})
    swanlab.finish()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
