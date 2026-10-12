# V67 Paired-Domain Expert Gate

V67 is a preregistered domain gate over V57 and the cross-fitted V66 supervised
expert Transformer. The four domains are the Cartesian product of acquisition
environment and whether a labeled paired source user exists. Each domain's
blend weight is selected independently on Fold 0 by decoded accuracy with a
lower-weight tie break, then transferred unchanged to Fold 1.

Only domain 2 selects a nonzero V66 contribution: weight 0.3 for early-environment
paired users. This corresponds to user 8 on Fold 0 and transfers to unseen user
7 on Fold 1. No user identifier is a model input; the gate uses only the
paired-source availability and environment category already used by V52/V57.

