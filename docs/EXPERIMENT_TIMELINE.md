# Experiment Timeline

The version numbers record research batches, not a monotonic model sequence. Many versions are parallel ablations, negative results, diagnostics, or post-freeze studies.

## Representation and Validation

| Version | Main change | Development result | Decision |
|---|---|---:|---|
| V1 | Initial six-modality 2D-CNN/statistical baseline | 53.94% | Starting point |
| V2 | Person crops, IR enhancement, world-frame IMU | 63.45% | Promoted |
| V3 | X3D-S spatiotemporal visual expert | 70.52% | Promoted |
| V4–V5 | Detail crops, skeleton attention, scene and tree baselines | No stable gain | Rejected |

The early gains came from fixing the representation, not from increasing classifier capacity.

## Compact System Distillation

| Version | Main change | Development result | Decision |
|---|---|---:|---|
| V24 | Full-data INT8 parameter soup + pseudo-label adaptation | 1168/1357 | Deployment anchor |
| V52 | Domain-conditioned logit residual | 1193/1357 | Promoted |
| V57 | Round-Transformer distillation | 1195/1357 | Promoted |
| V71 | Gated FiLM development student | 1201/1357 | Development milestone |

V57 compressed a sequence teacher into a 92,392-parameter Transformer and reproduced all teacher decisions. V71's FiLM correction is retained for frozen reproducibility but has no direct effect on selected V127/V164 labels: paired rows use V114, while unmatched rows use the gate-off V71 path that returns the exact V57 anchor.

## Protected Adaptation

| Version | Main change | Decoded correct | Decision |
|---|---|---:|---|
| V89 | Fixed 128/160 multi-scale adaptation | 1170 | Visual anchor |
| V97 | Zero-initialized protected visual adapters | 1173 | Adapter design retained |
| V98 | Target-only cross-modal adapter | 1154 | Raw gain, decoded loss |
| V104 | Domain-gated protected adapters | 1175 | Promoted |
| V113 | Paired-person adapter, equal fusion | 1176 | Precursor |
| V114 | 60/40 anchored paired fusion | 1180 | Paired expert |
| V127 | Route by same-person-source availability | 1205 | Primary system |

The key transition was from “one adaptation for every row” to “adapt only where the supervision condition supports it.”

## Structural Reasoning and Final Selection

| Version | Main change | Local | Public | Decision |
|---|---|---:|---:|---|
| V127 | Supervision-aware router | 88.80% | 0.92039 | Primary |
| V160 | Direct radar-time grouping | 90.05% | 0.84079 | Rejected |
| V164 | Posterior-gated radar challenger | 89.39% | 0.91044 | Secondary |
| V166 | Repeated-transition consensus | 90.35% | 0.88557 | Rejected |

The final selection prioritized shifted-distribution evidence and conservative behavior over the highest development score.

## Post-freeze Research

V167–V197 audited test distribution, modality alignment, high-resolution experts, raw radar, and multiple IMU formulations. None replaced the selected V127/V164 submissions. These studies support the diagnosis that conditional person/environment shift, rather than insufficient fitting capacity, was the dominant bottleneck.

