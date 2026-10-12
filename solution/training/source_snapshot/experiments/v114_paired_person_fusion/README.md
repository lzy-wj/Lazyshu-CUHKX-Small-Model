# V114 Paired-Person Fusion

V114 conservatively fuses the V113 target-specific visual adapters into V104.
Only the adapter weight changes. A coarse predeclared grid of `0.2, 0.4, 0.6,
0.8` is ranked on Fold0 pair-aware accuracy; the selected weight is then used
unchanged on Fold1. This preserves the legal counterpart-source supervision
protocol and keeps deployment comfortably below 100 MB.
