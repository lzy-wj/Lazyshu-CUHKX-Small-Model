# V68 Full-Data Modality FiLM Refit

V55 established a two-fold gain when its calibrated probabilities are blended
with V52: 1195/1357 versus 1193/1357, with improvement on both folds and a
substantial NLL reduction. V68 refits the same architecture on all 3036 labeled
training clips for 13 epochs, the rounded mean of the V55 best epochs (14 and
11).

The model uses one shared R(2+1)D-34 CNN with zero-initialized channel-wise FiLM
adapters for Depth, Thermal, and IR, temporal CNN/LSTM sensor encoders, and a
Transformer fusion network. Its compact INT8 checkpoint remains approximately
65.4 MB.

