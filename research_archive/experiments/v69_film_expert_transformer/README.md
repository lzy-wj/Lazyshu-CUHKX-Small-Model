# V69 FiLM Expert Transformer

V69 extends V66 with one additional expert: Fold-specific V55 modality-FiLM
probabilities. The deployment counterpart is the full-data V68 checkpoint.

Retention is selected by Fold-0 leave-one-user-out validation. The model is
then trained on Fold 0 and evaluated once on Fold 1. Reverse fitting creates
cross-fitted Fold-0 predictions. Four environment/paired-source blend gates
against V57 are selected on Fold 0 with a conservative lower-weight tie break
and transferred unchanged to Fold 1.

