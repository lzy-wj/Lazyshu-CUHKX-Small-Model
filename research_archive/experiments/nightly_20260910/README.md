# Nightly FiLM Generalization Matrix

All runs are controlled Fold-0 experiments derived from `v55_modality_film_f0`.
They keep the no-radar five-modality model, 32 frames, BF16 training, MixUp 0.3,
and the zero-initialized modality FiLM visual encoder. Each run changes one
scientific factor.

| Run | GPU | Changed factor | Hypothesis |
|---|---:|---|---|
| V61 | stopped | seed 43 | Aborted when the GPU policy changed to devices 4–7 only |
| V62 | stopped | 160 px cache | Aborted when the GPU policy changed to devices 4–7 only |
| V63 | 4 | masked-modality distillation 0.5 | Improves robustness to unavailable or weak modalities |
| V64 | stopped | invariant skeleton representation | Aborted to move V68 away from another user's GPU6 job |
| V65 | 7 | per-visual auxiliary supervision | Prevents one visual modality dominating the shared encoder |

The active GPU policy is devices 4–7 only. V61, V62, and V64 partial runs are
not eligible for analysis or promotion.

Promotion is not based on standalone accuracy alone. Every completed candidate
is evaluated against frozen V18 and V52 probabilities for raw accuracy, NLL,
pair-aware accuracy, correction overlap, and low-weight blend transfer. Fold 1
is only run for candidates that clear the Fold-0 gate.
