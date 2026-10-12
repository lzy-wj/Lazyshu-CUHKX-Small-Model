# Complete Research Archive Guide

This archive preserves the full available development record: successful branches, rejected ideas, diagnostics, validation revisions, compliance audits, and post-freeze studies. It explains how the final system emerged; it is not intended to be executed sequentially.

## Recommended reading paths

- **Fast overview:** [Experiment Timeline](../docs/EXPERIMENT_TIMELINE.md) and [Version Cards](../docs/VERSION_CARDS_zh-CN.md).
- **Full lineage:** [V1–V197 Research Lineage](../docs/RESEARCH_LINEAGE_zh-CN.md).
- **What failed:** [Negative Results](../docs/NEGATIVE_RESULTS.md).
- **Reproduction:** use [`solution/`](../solution/) and read [Reproducibility](../docs/REPRODUCIBILITY.md); historical scripts are retained for traceability rather than as the default entry point.

## Archive inventory

| Item | Count | Description |
|---|---:|---|
| Historical files | 1,196 | Source, configurations, research notes, lightweight results, and navigation |
| Experiment directories | 178 | 169 version-named directories and 9 early/special named batches |
| Version range | V1–V197 | Numbers contain gaps and do not imply monotonic improvement |
| Historical configurations | 42 | Training, input, fusion, validation, and deployment studies |
| Research scripts | 140 | Preparation, training, evaluation, diagnosis, compression, and packaging |

## Stage map

| Stage | Range | Central question | Main conclusion |
|---|---|---|---|
| Reliable starting point | V1–V5 | Which representation and validation setup works? | Input stabilization and spatiotemporal vision drove the largest early gains. |
| Research protocol | V6–V23 | How should candidates be promoted? | Single-fold or standalone gains were insufficient evidence. |
| Refit and distillation | V24–V71 | How can teacher behavior fit under 100 MB? | V24 became the anchor; compact residual and Transformer students added corrections. |
| Protected adaptation | V72–V104 | How can visual evidence improve without forgetting? | Frozen backbones and constrained adapters were safer than broad fine-tuning. |
| Paired transfer | V105–V127 | When is same-person source supervision reliable? | Deterministic routing by supervision availability produced V127. |
| Structural reasoning | V128–V166 | Can radar timing safely improve decoding? | Structural challengers required neural-posterior gating; local score alone was misleading. |
| Post-freeze audit | V167–V197 | What still limits generalization? | No audited branch reliably replaced V127/V164. |

## Directory map

- [`experiments/`](experiments/): version branches, status notes, and lightweight result summaries.
- [`configs/`](configs/): historical training and ablation configurations.
- [`docs/`](docs/): protocols, diagnoses, stage reports, and submission plans written during development.
- [`scripts/`](scripts/): historical preparation, training, evaluation, and packaging utilities.
- [`src/`](src/): shared early data, model, and validation implementations.

Generated checkpoints, predictions, caches, raw SwanLab logs, and public-probe artifacts are intentionally excluded from the public repository.
