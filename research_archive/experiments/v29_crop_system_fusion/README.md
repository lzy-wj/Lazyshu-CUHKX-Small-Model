# V29 Crop-System Fusion

V29 averages two predictions from the same deployable V24 model family:

- `0.40` V26 system-distilled head-adapter probabilities with crop `0.75`;
- `0.60` unadapted V24 probabilities with the wider crop `0.875`.

The weight was selected from a stable local plateau: both `0.375` and `0.40`
score `1177/1357` after the fixed pair-aware decoder. This is a multi-forward
inference method; it adds no second backbone asset and remains under 100 MB.
