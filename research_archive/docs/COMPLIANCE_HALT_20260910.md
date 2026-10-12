# Competition Compliance Halt

## Resolution

The ambiguity is resolved by an organizer reply in the competition's dedicated Kaggle discussion, `Clarification on “No LLMs for development”`:

- https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/724942
- Organizer message ID: `3498056`, posted `2026-07-15T02:14:10.313Z`.

The organizer explicitly states that AI coding assistants including ChatGPT, Codex, Copilot and Cursor are allowed for writing and debugging code. The restriction applies to the submitted inference model: it must use a permitted small architecture, remain under 100 MB on disk, and must not contain a large model.

This is the written clarification required by the resume condition below. Competition-oriented work may therefore resume while retaining the final-model architecture and size checks.

## Reason

The official Kaggle overview for the CUHK-X Small Model Track states that LLMs and closed-source APIs are not allowed for development, in addition to prohibiting LLM-based labeling. The official UbiComp challenge page repeats that restriction for the Small Model Track.

- Kaggle overview: https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track
- Official challenge page: https://www.ubicomp.org/ubicomp-iswc-2026/cuhk-x-competition/

Because this repository has been developed with an LLM coding agent, further competition-oriented model development or automatic Kaggle submission cannot currently be represented as compliant with the published rules.

## Action Taken

- Stopped all `v24-*` tmux training and watcher sessions at approximately `2026-09-10 13:22 Asia/Shanghai`.
- Stopped only this project's processes; unrelated GPU2 and GPU6 processes were not touched.
- Preserved all source files, logs, SwanLab records and completed checkpoints.
- Did not submit V24 to Kaggle.

## Preserved Completed Artifacts

- `experiments/v24_full_refit/outputs/checkpoints/v24_full_noradar_e13/`
- `experiments/v24_full_refit/outputs/checkpoints/v24_full_warm_maskdistill_e9/`
- `experiments/v24_iterative_loop/selection.json`
- `experiments/v24_iterative_loop/outputs/v24_iterative_k1_f0/`
- `experiments/v24_iterative_loop/outputs/v24_iterative_k2_f0/`
- `experiments/v24_iterative_loop/outputs/v24_iterative_k3_f0/`

## Resume Condition

Competition-oriented work may resume only after written clarification from the organizer that LLM coding/development assistance is permitted for the Small Model Track despite the currently published wording. Organizer message `3498056` satisfies this condition; work resumed after preserving the interrupted-run audit trail.
