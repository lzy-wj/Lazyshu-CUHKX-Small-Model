# Results

Retaining the same-person opposite-environment source raises standalone Thermal
accuracy, but not enough to challenge the multimodal system:

| Target | Thermal | V166 | Oracle gain over V166 |
|---|---:|---:|---:|
| user 8 | 73/163 | 155/163 | 1 |
| user 18 | 86/177 | 167/177 | 1 |
| user 7 | 91/182 | 173/182 | 2 |
| user 17 | 81/163 | 157/163 | 2 |

Only six rows across all four targets are theoretically recoverable by direct
replacement. Decision: do not deploy the frame expert; test the same supervision
condition with the full multimodal model in V175.
