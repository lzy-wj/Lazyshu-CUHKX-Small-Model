# Status

Complete and rejected.

- Four domain-1-only jobs completed: old and refreshed pseudo teachers on both
  folds.
- The routing and fusion weights are identical to V104; only adapter training
  scope and AdaBN estimation change.
- V106 reaches 1,171/1,357 after pair decoding, versus 1,175 for V104.
- Target-domain weighted pair accuracy falls from 89.2466% to 89.0168%.
- Multi-domain adaptation followed by inference-time routing remains the
  stronger regularized solution.
