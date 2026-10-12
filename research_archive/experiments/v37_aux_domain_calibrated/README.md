# V37 Auxiliary + Domain + Decoder

V37 composes three independently selected, low-complexity changes over the V24
hard-pseudo public best:

- V32 residual domain gate for unmatched environment-1 rows;
- V36 5% skeleton-auxiliary blend on visual-missing rows;
- V31 calibrated gap penalty `1.0` and maximum edits `6`.

Fold 0 improves from `550` to `552`; blind Fold 1 improves from `618` to `624`.
The model assets remain the same V24 INT8 checkpoint, head adapter, and YOLO.
