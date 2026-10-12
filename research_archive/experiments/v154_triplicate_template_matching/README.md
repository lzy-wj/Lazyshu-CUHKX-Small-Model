# V154 Triplicate Template Matching

Each labeled CUHK-X source script is normally recorded three times. V143 used a
one-to-one assignment between target units and unique source templates, which
under-counted this known acquisition capacity. The unrestricted V127 decoder
has the opposite problem: it may reuse one attractive source script arbitrarily.

V154 assigns every target round globally to three slots per unique paired-source
template, with free-decoding dummy slots as a safe fallback. Flexible alignment,
transition scores, and all strengths remain exactly V127. The rule has no fitted
parameter and is applied identically to both held-out folds.
