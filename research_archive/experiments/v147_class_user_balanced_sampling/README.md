# V147 Class-User Balanced Sampling

V146 samples classes uniformly but still lets frequent `(class,user)` cells
dominate within each class. V147 assigns every available user equal mass
inside a class, then every clip equal mass inside that user-class cell. This
targets the measured cross-subject gap while preserving a uniform class
distribution.

All model, initialization, optimizer, augmentation, and distillation settings
are identical to V146. Only the sampler weights differ.

