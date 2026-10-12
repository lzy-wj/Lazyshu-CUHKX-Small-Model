# V74 FiLM Per-Visual Parameter Soup

V74 averages V55 and V65 weights because both checkpoints have exactly the
same inference graph. V65's per-visual flag only changes auxiliary supervision
during training; it adds no parameters. Fold 0 screens V65 weights 0.25, 0.50,
and 0.75, after which one frozen soup may advance to Fold 1. Each INT8 soup is
one approximately 65 MB model, not an ensemble asset.
