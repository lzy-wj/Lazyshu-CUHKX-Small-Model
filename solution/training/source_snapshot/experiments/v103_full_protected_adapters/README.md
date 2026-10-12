# V103 Full Protected Adapters

V103 performs the fixed six-epoch full-data refit justified by V97. It trains
the intermediate and dual zero-initialized CNN adapter variants on all 3036
labeled clips from the full V24 FP32 soup. The complete base network remains
frozen, and only compact adapter states are saved.

This is candidate construction, not new architecture selection. Kaggle is not
called automatically.
