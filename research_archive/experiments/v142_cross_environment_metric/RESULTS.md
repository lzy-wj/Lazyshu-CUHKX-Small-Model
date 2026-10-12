# Results

| Fold-0 paired retrieval | Correct |
| --- | ---: |
| Fused prototype / nearest | 147 / 131 of 348 |
| Visual prototype / nearest | **223 / 225 of 348** |
| All-modal prototype / nearest | 210 / 208 of 348 |

The residual metric is trained on three known cross-environment pairs while
excluding both validation users. It improves visual nearest retrieval from
220 to 225 rows over V141, confirming transfer beyond memorized identities.
However, all nonzero additions to V114 damage the final Fold-0 sequence result;
the selected zero-weight control remains V127 at 1,205/1,357.

