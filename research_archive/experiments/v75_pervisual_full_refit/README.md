# V75 Full-Data Per-Visual FiLM Refit

V75 is conditionally launched only if V65 improves V55's combined raw
cross-user score. Its epoch count is the rounded mean of the two independently
selected V65 best epochs. It refits the same joint-FiLM inference graph on all
3,036 labeled clips, with per-visual auxiliary supervision used only during
training. The compact INT8 checkpoint remains about 65 MB.
