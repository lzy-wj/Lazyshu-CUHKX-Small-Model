# V96 Results

| Variant | Parameters | Raw | Pair-aware | Target-domain weighted pair |
|---|---:|---:|---:|---:|
| V23 high-resolution source | 65.0M | 1057/1357 | 1160/1357 | 87.32% |
| V89 local best | deployable 71.3 MB | 1077/1357 | **1170/1357** | **88.81%** |
| Independent layer4 | 143.1M | 1055/1357 | 1148/1357 | 87.48% |
| Independent layer3+4 | 183.9M | 1026/1357 | 1138/1357 | 86.79% |
| Independent layer4 / source equal average | - | 1057/1357 | 1167/1357 | 88.13% |
| Independent layer4 / V89 equal average | - | 1072/1357 | 1169/1357 | 88.73% |

Both capacities regress sequence-level accuracy despite exact source
initialization, BF16 training, source-logit retention, per-stream auxiliary
supervision, and EMA. The deeper split is consistently worse. This directly
rejects insufficient trainable CNN parameter count as the active bottleneck:
unprotecting 117M-178M visual parameters mainly causes cross-user forgetting.

No V96 checkpoint is promoted, distilled, or submitted. V97 follows the causal
result by freezing the entire source path and learning only zero-initialized,
normalization-safe visual residual adapters.
