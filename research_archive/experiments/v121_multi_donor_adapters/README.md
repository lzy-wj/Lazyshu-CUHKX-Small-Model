# V121 Multi-Donor Adapters

V121 is the robust counterpart to V120. Instead of trusting one retrieved
script donor, it uses the top two prior label-free retrieval matches: users
20+17 for Fold0 target 23, and users 21+18 for Fold1 target 22. The larger
clean source pool should retain action coverage while target pseudo labels
provide domain adaptation. All other V113 settings remain fixed.
