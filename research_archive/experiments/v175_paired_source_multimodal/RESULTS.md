# Results

All four paired-source multimodal runs were stopped after epoch one. Retaining
the same-person opposite-environment source did not approach V166:

| Target | Candidate | V166 | Oracle gain over V166 |
|---|---:|---:|---:|
| user 8 | 123/169 | 158/169 | 0 |
| user 18 | 137/179 | 169/179 | 0 |
| user 7 | 140/185 | 176/185 | 1 |
| user 17 | 139/166 | 160/166 | 2 |

Only three rows across all targets are theoretically recoverable. Decision:
reject direct paired-source fine-tuning and preserve V166.
