# V143 Global Template Matching

The paired decoder currently chooses the best labeled source template for each
target round independently. V143 collapses detected repeated target rounds and
matches these units jointly to the paired source user's canonical scripts.
Each source script instance can be used at most once; unmatched target units
retain the free transition-aware prediction.

This is a parameter-free structural test. It keeps V127 unchanged for users
without paired labeled supervision and uses no validation labels during
decoding.

