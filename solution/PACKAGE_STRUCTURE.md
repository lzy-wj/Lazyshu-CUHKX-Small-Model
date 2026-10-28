# Lazyshu Small-Track Verification Package

This tree documents the original private verification package. The public
checkout preserves the source, configurations, environment pins, and verification
records. Runtime assets, final submission CSVs, submission metadata, and signed
declarations shown below are retained privately and are not included here. See
[the public source guide](README.md) and [reproduction notes](../docs/REPRODUCIBILITY.md).

```text
small_Lazyshu_verification_v1/
├── README.md                       # organizer start point and exact commands
├── manifest.yaml                   # complete machine-readable inventory
├── inference.sh                    # required single entry point
├── LICENSE                         # Apache-2.0 for participant-authored source
├── THIRD_PARTY_NOTICES.md
├── src/                            # standalone dynamic inference implementation
├── configs/                        # frozen runtime and hash inventory
├── artifacts/                      # all 84,094,248 bytes of runtime model assets
├── environment/                    # pinned Python/CUDA environment
├── training/                       # training lineage, configs, and source snapshot
├── submission/                     # both exact Selected CSVs and Kaggle evidence
├── verification/                   # local end-to-end reproduction evidence
├── licenses/                       # bundled third-party licence text
└── declarations/                   # organizer honor declaration; team must sign
```

The package contains no CUHK-X raw data, participant cache, credential, API key, or virtual environment. All paths used by inference are relative to the package root or supplied through the required command-line arguments.
