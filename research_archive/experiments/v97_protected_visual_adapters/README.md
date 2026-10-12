# V97 Protected Visual Adapters

V97 follows the causal result from V96: independently fine-tuning 117M-178M
high-level CNN parameters causes immediate cross-user forgetting. Instead,
V97 keeps the complete V23 encoder and fusion path frozen and inserts one
modality-specific spatiotemporal CNN residual adapter per visual stream.

The adapters operate before shared layer4, where they can alter the actual
features consumed by the original classifier, and optionally after layer4.
Their final projections are zero initialized, so the model begins at the exact
V23 function. GroupNorm avoids introducing modality-specific running-statistic
drift. Per-stream supervision and source-logit retention remain unchanged.
