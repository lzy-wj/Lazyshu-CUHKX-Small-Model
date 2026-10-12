# V94 Upper-Body View Consistency

V94 trains the V88 pseudo-adapted classification heads on two semantic views of
every selected target clip: the standard person crop and its inherited upper
72%. Pseudo-label and teacher-retention losses are averaged equally across the
two views. Inference uses the same fixed equal full/upper view family as V93.

No model parameters, pseudo labels, optimizer settings, epochs, or blend weights
are added or searched. This tests whether explicit crop invariance improves
fine-grained hand-object generalization.
