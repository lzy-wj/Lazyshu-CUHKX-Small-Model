# Results

The runs were stopped after four epochs because neither leave-one-user target
improved over its first checkpoint and both remained far below V166:

- user 23 best raw: `99/141`; decoded: `102/141`; V166: `107/141`.
- user 22 best raw: `157/194`; decoded: `170/194`; V166: `178/194`.

Adding the three other historical held-out users does not solve the unmatched
target ceiling. Decision: reject and redirect compute to paired-source tests.
