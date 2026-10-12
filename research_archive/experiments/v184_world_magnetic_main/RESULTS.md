# Results

The clean rerun verifies that direct magnetometer channels can move a small
number of raw V23 decisions. Freezing the visual branch selects epoch 1 on both
folds and gives `480/652 + 566/705 = 1046/1357`, versus `1044/1357` for V23.

This raw gain is not a final-system gain. The candidate makes gains and losses,
the V166 decoder already fixes one gain in each fold, and conservative routing
does not change unmatched predictions while paired routing regresses Fold 0.
The candidate is therefore retained as positive representation evidence only.

Full-model fine-tuning is unstable across folds: its first epoch scores
`472/652` and `568/705`, and by epoch 3 scores `478/652` and `547/705`.
It is stopped and rejected.
