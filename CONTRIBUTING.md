# Contributing

Thanks for your interest in Lazyshu.

## Scope

Good contributions include:

- Reproduction fixes that preserve the frozen V127/V164 outputs.
- Portable data-path handling and environment improvements.
- Tests for preprocessing, missing modalities, routing, and decoding.
- Clearer documentation or independent benchmark results.

The `research_archive/` directory is historical. New development should target `solution/` or a clearly separated experimental module rather than rewriting archived records.

## Before Opening a Pull Request

1. Do not include CUHK-X data, derived video caches, model checkpoints, credentials, or raw SwanLab logs.
2. Run `python tools/validate_repository.py`.
3. Run `python -m compileall -q solution research_archive/src research_archive/scripts`.
4. Describe whether the change affects model bytes, predictions, or only documentation.
5. Preserve third-party attributions and licenses.

## Reproducibility Claims

Do not claim an independent test result without identifying the split and artifact. Both published folds are development validation. A full-data checkpoint does not have a held-out score.

