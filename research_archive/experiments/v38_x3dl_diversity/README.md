# V38 X3D-L Diversity

V38 adds the existing compact X3D-L visual encoder as a low-weight auxiliary
expert to the V24 hard-pseudo model. The projected inference assets, including
YOLO and the person detector used by X3D-L preprocessing, remain below 90 MB.

Fold 0 searches temperature, weight, and a simple confidence/missingness scope.
Candidates are shortlisted by raw accuracy, selected by the fixed pair-aware
decoder, and transferred once to Fold 1.
