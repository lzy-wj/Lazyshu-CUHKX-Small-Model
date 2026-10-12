# V56 Results

V56 retrieved one same-environment training-script source for every validation
user using aggregate model/template margin only. Known paired users retained
their fixed source, while unmatched users used the retrieved source. A Fold-0
grid varied edit strictness and minimum acceptance margin on frozen V52
probabilities.

The selected setting (`template_strength=1.4`, `gap_penalty=0.6`,
`maximum_edits=4`, `minimum_margin=4`) tied V52 at `568/652` on Fold 0 but
fell from `625/705` to `621/705` on blind Fold 1. Combined accuracy is
`1189/1357`, four rows below V52. Retrieved-script rescue is rejected and no
test prediction or Kaggle submission is produced.
