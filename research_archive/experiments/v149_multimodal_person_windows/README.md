# V149 Multimodal Person Windows

The current cache detects a person in IR first and consults Depth only when IR
returns no box. Visual inspection found dark-IR false positives that crop the
actual person out of all three streams. V149 detects independently in Depth,
IR, and Thermal, estimates temporal consistency, and chooses the strongest
modality-specific track for each clip.

The first model gate keeps V23 frozen and changes only preprocessing. A low
fusion weight is selected on Fold 0 and transferred to Fold 1. Training on the
new cache is allowed only if this frozen test establishes useful information.

