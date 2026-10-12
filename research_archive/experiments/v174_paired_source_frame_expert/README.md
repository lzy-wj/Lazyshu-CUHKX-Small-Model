# V174 Paired-Source Frame Expert

The historical folds exclude both aliases of the same person (`8/18` and
`7/17`). That is conservative for unseen-person generalization but does not
match 329 of 405 test clips: test users 10/11/26 have labeled counterparts
20/21/16 in the other environment.

V174 holds out exactly one target alias while retaining its legal labeled
counterpart. It runs the official-style 224px Thermal ResNet on targets 8, 18,
7, and 17. This directly measures whether full-data training can exploit the
same-person source without using target labels during optimization.
