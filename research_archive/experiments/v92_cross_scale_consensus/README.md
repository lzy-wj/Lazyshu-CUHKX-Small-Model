# V92 Cross-Scale Consensus Pseudo Labels

V92 addresses confirmation bias in V24/V88 self-training. A pseudo target is
kept only when the original 128-pixel pair-aware teacher and the frozen
160-pixel pair-aware view predict the same class. The original V24 selection
rule is applied first; disagreement can only remove a pseudo target, never add
one or change its class.

This label-free rule removes 42 selected OOF targets. Audit-only labels show
that the retained pseudo-label accuracy rises from about 91.6% to 93.5--94.3%
on both held-out folds. V92 then reruns the exact V88 adaptation recipe.
