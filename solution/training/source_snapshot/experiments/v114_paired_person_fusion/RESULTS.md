# Results

| Candidate | Fold 0 pair | Fold 1 pair | Combined | Target-weighted mean |
|---|---:|---:|---:|---:|
| V104 | 550 | 625 | 1175/1357 | 89.25% |
| V114 | **552** | **628** | **1180/1357** | **89.80%** |

The Fold0-selected adapter weight is `0.6`. Raw accuracy also rises from
1076/1357 to 1091/1357. The gain is therefore not a decoder-only artifact:
paired-person true-label adaptation improves clip evidence, while V104 retains
40% weight to protect the original cross-user representation.
