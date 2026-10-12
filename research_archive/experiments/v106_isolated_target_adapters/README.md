# V106 Isolated Target Adapters

V106 changes only target-adaptation scope. V98/V99 train and estimate AdaBN
over all four validation users, although V104 routes their predictions only to
the unmatched high-environment domain. V106 instead trains and estimates BN
statistics only on domain 1 (`user23` on Fold 0 and `user22` on Fold 1).

Old V13 and refreshed V89 pseudo teachers are both retained because V104's
fixed equal fusion uses their diversity. Four jobs run the exact V98 cross-modal
adapter recipe; no weight or epoch search is introduced.
