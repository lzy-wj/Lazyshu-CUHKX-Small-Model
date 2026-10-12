# V40 Results

Hybrid INT4 improves Fold 0 raw top-1 from `493` to `500`, but the fixed
pair-aware decoder falls from `550` to `547`. It fails the predeclared Fold 0
retention gate, so Fold 1 and the INT4 dual-backbone path are not run.
