# Final Reproducibility Check

This directory verifies the two selected final submissions without modifying the
frozen candidate files.

Run from the repository root:

```bash
conda run -n Train python \
  experiments/final_reproducibility_20260915/rebuild_selected.py \
  --output-dir /tmp/final_reproducibility_20260915
```

The check performs all of the following:

- Re-hashes every required inference asset from the V127 asset audit.
- Confirms the combined model assets occupy `83,900,083` bytes, below the
  `100,000,000` byte limit.
- Rebuilds `v127_without_0531.csv` from the full V127 candidate by rolling the
  four changed `2025-05-31` rows back to the frozen V24 incumbent.
- Rebuilds `v164_evidence_gated_radar.csv` by applying the fixed top-2/outside-
  top-2 evidence gate to V127 and the Radar-batch challenger.
- Requires both rebuilt files to be byte-identical to the frozen submissions.

Expected SHA256 values:

- `v127_without_0531.csv`:
  `76210e13f1f63a3d63c356f2462aa09917a6d9c0c919a0e8025098ab4408acb0`
- `v164_evidence_gated_radar.csv`:
  `aaf3ac9dc0fe71fe5a21b78d2243ac330dee3bec0aad69ebb52f11acb7989639`

The verification targets deployment reproducibility from the frozen trained
weights and deterministic inference/post-processing chain. Bit-identical model
weights are not promised for a new GPU retraining run.
