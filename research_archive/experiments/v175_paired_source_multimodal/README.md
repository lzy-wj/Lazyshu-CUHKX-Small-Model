# V175 Paired-Source Multimodal

V175 directly tests the full-data deployment advantage on the historical
paired users. Each run holds out one target alias (`8`, `18`, `7`, or `17`)
while retaining its same-person opposite-environment counterpart in training.
The warm checkpoint excluded both aliases, so target labels never enter model
optimization.

This mirrors 329/405 test clips whose hidden user has a labeled counterpart
(`10/20`, `11/21`, or `26/16`). A low learning rate and low visual-backbone
factor protect the cross-user representation while adding counterpart data.
