# V146 Balanced Warm Refinement

The CUHK-X paper reports a large gain from class-balanced resampling, whereas
the previous local V107 experiment changed loss weights and the old V18
resampling run also enabled the weak Radar branch. V146 isolates actual
resampling on the proven no-Radar V23 graph.

Both folds warm-start from their V23 checkpoints and use the same five-epoch
schedule, mixup, masked-modality distillation, and BF16 path as V140. The only
new training variable is inverse-frequency sampling with replacement.

