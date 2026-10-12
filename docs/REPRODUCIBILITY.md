# Reproducibility Contract

## Frozen bundle status

The authoritative competition artifact is retained privately and is not part of the public release:

```text
SHA-256 060c126ac015135f5265d510ed6db6eaa799d515f22178fb6b12c2cbf28cc55a
```

It contains the final runtime weights, inference source, environment pins, training snapshot, submission metadata, licenses, and verification scripts. The public repository releases the source/configuration side only.

The public code supports method inspection and independent retraining, but does not promise byte-exact regeneration of the original competition CSV without the private frozen artifacts.

## Runtime behavior

- Discovers the Testing root and sample count dynamically.
- Preserves official CSV path order.
- Handles missing modalities and unknown dates.
- Verifies every runtime artifact's byte size and SHA-256.
- Rejects unsupported IDs, duplicate paths, nulls, schema/order errors, and labels outside `[0, 39]`.
- Performs inference without network access.

## Frozen prediction order

1. Compute and decode the V24 incumbent, unmatched sequence anchor, and paired adapter separately.
2. Select decoded V114 when a labeled same-person source exists; otherwise use decoded V57 through the gate-off V71 path.
3. Apply matched-source label-coverage protection.
4. Apply the disclosed date-and-disagreement rollback.
5. For V164 only, build a radar-timed challenger and apply the posterior-rank gate.

## Measured runtime

- Preprocessing: 76.21 s.
- Neural inference: 311.17 s.
- Structural decoding and I/O: 20.39 s.
- Total for 405 clips: 407.77 s.
- Peak allocated CUDA memory: 13.49 GB.

Measured on one NVIDIA A800-SXM4-80GB.
