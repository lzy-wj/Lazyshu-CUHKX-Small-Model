# V160 Radar Batch Decoder

V160 changes only the grouping key used by the fixed V127 decoder. The
timestamp embedded in each Radar filename identifies the acquisition batch;
training batches have essentially unique activity labels. Visual frame and
CSV timestamps are retained for date routing, while the Radar filename
timestamp drives repeat pooling and source-template decoding.

The model probabilities, source users, template strengths, and transition
strength are unchanged. The test candidate uses source-user coverage
protection and does not use test labels.
