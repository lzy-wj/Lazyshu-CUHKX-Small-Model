# V59 Results

| Metric | V52 | V57 direct | V59 |
|---|---:|---:|---:|
| Fold 0 | 568/652 | 567/652 | **569/652** |
| Fold 1 | 625/705 | 628/705 | **628/705** |
| Combined | 1193/1357 | 1195/1357 | **1197/1357** |

V59 adds no parameters beyond V57. It applies the inherited label-free repeat
recovery and frozen pair-aware decoder to the compressed teacher probabilities.
The final asset set remains `71,324,325` bytes.

The 405-row test CSV changes three labels relative to V52 and two relative to
V57, so it is an informative hybrid rather than a duplicate submission.

Candidate submission:
`experiments/v24_full_refit/outputs/submissions/v59_teacher_sequence_hybrid.csv`

SHA256: `001139324764176f2f0bab0dea1255dca1526c853034f36495de67002ef5a418`
