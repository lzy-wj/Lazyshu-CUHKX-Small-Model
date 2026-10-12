# V129 Full-Data MViT Refit

V129 refits both complementary MViT recipes on all 3,036 labeled clips. The
endpoint is epoch 14, the independently selected best epoch shared by V126 and
V128 at launch time. The conservative model keeps V126's 24-epoch schedule;
the domain-adapted model keeps V128's 18-epoch schedule. Their test predictions
are training-time teacher evidence; any final multi-model correction must be
distilled back under the 100MB deployment limit.

