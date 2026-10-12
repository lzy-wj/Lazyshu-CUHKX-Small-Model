# Results

The strongest direct paired-user retrieval uses the three visual stream
embeddings with horizontal-flip pooling and nearest-source matching. It reaches
220/348 on Fold 0. Fused and structured embeddings are much weaker at 99/348
and 125/348 respectively.

Despite meaningful direct retrieval, every nonzero evidence strength reduces
the final V127 system on Fold 0. The paired decoder and V114 adapter already
retain substantially cleaner decisions than nearest-source matching. Formal
selection therefore returns V127 unchanged at 1,205/1,357.

