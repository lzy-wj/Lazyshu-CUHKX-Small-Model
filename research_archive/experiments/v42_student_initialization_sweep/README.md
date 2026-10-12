# V42 Student Initialization Sweep

V42 keeps the V13 pseudo teacher, hard-label objective, adaptation parameters,
TTA, and decoder fixed. Fold 0 compares four INT8 student initializations:
equal five-way parameter soup, warm masked-distillation, progressive high-
resolution masked-distillation, and longer no-radar training.

Only a student exceeding the warm-60 soup reference `550/652` direct pair-aware
score may proceed to Fold 1.
