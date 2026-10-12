# V148 Environment-Specific Teachers

The CUHK-X paper reports that excluding cross-domain RGB data gives its best
cross-subject result. V148 tests that causal observation in the current model:
one warm-started no-Radar teacher trains only on users from physical
environment 0, and another only on environment 1.

Each checkpoint is selected by validation accuracy in its own environment.
At inference the known sensor environment chooses the corresponding teacher.
Fold 0 selects only one shared retention weight against V127; Fold 1 receives
that frozen setting.

