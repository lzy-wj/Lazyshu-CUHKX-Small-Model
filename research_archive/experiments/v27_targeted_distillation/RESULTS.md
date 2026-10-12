# V27 Results

Fold-0 direct pair-aware accuracy is `543/652`, below the V26 reference of
`550/652`. Fold 1 was not trained.

Although the selected disagreement labels had high audit accuracy, repeatedly
optimizing only 123 rows shifted the classification heads too aggressively and
hurt unchanged classes/users. The candidate is rejected. V26's all-row system
distillation remains the promoted compression recipe.

