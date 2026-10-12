# V66 Supervised Expert Round Transformer

V66 uses the remaining deployment budget for the existing compact X3D-L visual
expert and trains a small permitted Transformer to combine six probability
streams across each acquisition round:

- V52 domain residual anchor
- V24 adapted prediction
- V24 base prediction
- compact X3D-L prediction
- skeleton auxiliary prediction
- IMU auxiliary prediction

Six modality-presence flags are appended per sample. No sample ID, user ID,
trial name, or label is used as a model input. Acquisition timestamps are used
only to group and order samples within a round.

The retention coefficient is selected by leave-one-user-out validation within
Fold 0. The selected recipe is trained on Fold 0 and evaluated once on Fold 1.
The reverse Fold-1-to-Fold-0 model creates leakage-free cross-fitted local
predictions. A blend weight against V57 is selected on Fold 0 and transferred
unchanged to Fold 1.

The final test Transformer is trained on all cross-validation rows. Deployment
assets include V59, compact X3D-L, its compact person detector, and the V66
Transformer; the total must remain below decimal 100 MB.

