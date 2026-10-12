# Local verification evidence

`original_test_report.json` records a clean end-to-end invocation of the exact official entry point against the published raw label-free Testing directory. It rebuilt sensors and both video resolutions, ran all neural branches, and wrote a CSV whose SHA-256 exactly equals the retained primary Kaggle CSV.

`model_branch_report.json` records the fresh-forward comparison of every model branch and both selected outputs. The one intermediate V114-21 top-1 difference was numerically absorbed by the downstream fixed fusion/decoder; both final outputs have zero label differences and exact reference hashes.

Additional internal checks compared freshly generated preprocessing arrays to the original frozen caches (exact sensor arrays and exact resized video pixels). The content-fingerprint cache hardening was then rerun and remained byte-identical for both 128/160 video memmaps and array-identical for every sensor feature. No cache or competition data from those checks is included in this package.
