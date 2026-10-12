# V79 Wide Loop Trust Region

V78 showed that unconstrained fine-tuning of an 89M-parameter wide iterative
fusion model destroys the strong `487/652` initialization. V79 isolates the
new capacity:

- freeze every V23 base-model parameter;
- train only the 24.6M-parameter wide fusion residual;
- scale its residual logits by `0.25`;
- retain the frozen base distribution with KL weight `0.5`;
- preserve epoch 0 as a valid checkpoint;
- compare equal-parameter `K=1` and `K=2` on Fold 0.

The promotion target remains `>=490/652` with NLL no worse than `1.17912`.

