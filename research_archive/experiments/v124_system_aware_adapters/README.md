# V124 System-Aware Paired Adapters

V124 resolves a teacher mismatch in V113. Hard pseudo labels come from the
sequence system, but V113's soft retention teacher is only the V23 base model.
V124 adds a fixed 0.5 KL loss toward V104 soft probabilities on target rows,
while keeping counterpart true labels, V23 retention, visual stream
distillation, architecture, epochs, and optimizer unchanged. The adapter is
therefore trained to add person-specific evidence without drifting away from
the deployed system distribution.
