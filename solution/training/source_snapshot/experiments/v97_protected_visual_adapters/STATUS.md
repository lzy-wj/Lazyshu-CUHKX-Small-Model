# Status

Complete. Exact initialization and BF16 gradient flow passed. The intermediate
adapter reaches 1173/1357 pair-aware, but its gain is concentrated in the
test-absent domain; the fixed dual/V89 average improves target-weighted pair
accuracy on both folds while tying V89's total count. V98 tests the stronger
target-domain formulation.
