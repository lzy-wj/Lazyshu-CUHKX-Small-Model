# Results

Global replacement of the historical IR-first crop reduces frozen V23 from
479 to 460 raw-correct Fold-0 rows and is rejected.

The surgical fallback preserves the old crop unless IR detection quality is
below 0.1 or 0.2. Both thresholds produce the same four top-1 changes: zero
corrected errors and one newly wrong row, reducing raw accuracy from 479/652
to 478/652. The remaining changes are wrong-to-wrong.

The visual audit found real detector failures, but replacing their crops with
an independently detected Thermal/Depth crop creates a larger train/inference
view mismatch than the frozen encoder can tolerate. V149 is rejected before
Fold 1 by the preregistered frozen-model gate.
