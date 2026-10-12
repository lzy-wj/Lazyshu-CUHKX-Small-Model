# V27 Targeted System Distillation

V26 trains on every system-teacher label. V27 changes one variable: only rows
where the frozen system teacher disagrees with the unadapted V23 student receive
hard pseudo-label loss. All rows still receive retention KL through the frozen
student teacher.

The disagreement labels are highly selective in the frozen audit:

- Fold 0: system teacher `91/123`, student `11/123`.
- Fold 1: system teacher `77/106`, student `16/106`.

The inherited two-epoch adaptation recipe is unchanged. Fold 1 runs only if
Fold 0 beats V26's direct pair-aware `550/652`. Promotion requires beating
V26's combined `1170/1357`.

