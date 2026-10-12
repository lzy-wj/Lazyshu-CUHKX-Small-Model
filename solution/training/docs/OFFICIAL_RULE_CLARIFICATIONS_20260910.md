# Official Rule Clarifications

The following organizer replies were retrieved from the authenticated Kaggle competition discussion API on `2026-09-10`. They resolve rules that directly affect the current pipeline.

## AI Coding Assistants

- Topic: `Clarification on “No LLMs for development”`
- URL: https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/724942
- Organizer message: `3498056`
- Decision: ChatGPT, Codex, Copilot, Cursor and similar AI coding assistants are allowed for code writing and debugging. The final inference checkpoint itself must be a permitted small architecture below 100 MB and must not contain a large model.

## R(2+1)D and YOLO Initialization

- Topic: `Clarification: IG-65M pretrained R(2+1)D — "small pretrained CNN" or "large pretrained backbone"?`
- URL: https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/738333
- Organizer message: `3521750`
- Decision: YOLO11n used for deterministic label-free person cropping is allowed when its weights are included in the package. R(2+1)D-34 initialized from IG-65M and Kinetics-400 is also an allowed pretrained CNN in this track.

## Test Pseudo-Labeling

- Topic: `Clarification on Whether Pseudo-Labeling Test Data Is Allowed`
- URL: https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/729989
- Organizer message: `3504932`
- Decision: model-generated pseudo-labels on unlabeled test samples may be used for self-training or semi-supervised learning. Ground-truth test labels and manual annotation remain forbidden. The organizer warns that test-specific adaptation may generalize poorly to the Stage 2 and Stage 3 unseen sets.

## Ensemble and Quantization Accounting

- Topic: `Clarification on ensembles under the 100 MB model size limit`
- URL: https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/729056
- Organizer message: `3504929`
- Decision: all inference weights, including every ensemble member, must be bundled into one checkpoint below 100 MB on disk. FP16, INT8 and lower quantization are permitted. Latency and memory should be reported, although storage size is the hard leaderboard-stage constraint.

## Operational Consequences

- The V24 CNN/RNN/Transformer pipeline may continue with AI coding assistance.
- The existing IG65M/Kinetics R(2+1)D-34 and YOLO11n components are permitted.
- The final release audit must count the sum of all loaded inference weights and enforce `<100,000,000` bytes.
- Test-time pseudo-label adaptation remains a separately validated candidate rather than the default, because final verification uses new unseen subjects.
