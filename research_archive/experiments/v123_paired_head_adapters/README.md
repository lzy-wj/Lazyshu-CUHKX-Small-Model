# V123 Paired Head Adapters

V123 adapts the multimodal classification and fusion-output heads rather than
the visual encoder. It uses the same counterpart true labels and target pseudo
labels as V113. This is motivated by V24: compact head self-training produced
the strongest verified public gain, while V113 established that paired source
labels add person-specific information. The backbone remains fully frozen.
