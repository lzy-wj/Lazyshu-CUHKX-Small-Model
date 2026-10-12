# V155 Source-Subsequence Decoder

The paired test recordings visibly repeat partial versions of labeled source
scripts. V127's flexible alignment still permits arbitrary inserted labels,
which weakens source supervision exactly where the video classifier is unsure.

V155 tests the stronger acquisition hypothesis: a paired target round is an
ordered subsequence of one source script. For each round/template pair it
enumerates all source-label subsequences of the observed length, scores them
with the unchanged unary and transition model, and performs global assignment
with three slots per source script. Longer target rounds retain free decoding.

The rule has no fitted parameter. Existing V127 strengths are reused, and the
same rule is evaluated on both local folds before any test candidate is built.
