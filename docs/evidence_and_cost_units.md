# Evidence units and what the audit budget measures

The generator computes public and exhaustive hidden scores once for offline
evaluation. During recursive replay, the controller receives hidden scores only
at its recorded paid audit events. Thus the budget measures **accounted oracle
revelations in replay**, not wall-clock time or newly executed test suites. The
offline evaluator separately retains all scores to measure outcomes and failure.
This separation supports controlled algorithm comparisons, not production-cost
claims. Leakage tests check that unqueried rewards cannot alter decisions.

The paid-draw protocol charges repeated candidate draws; the source-cached protocol
charges distinct source identities only. These are different cost models and must
not be pooled. Even source caching does not identify all semantically equivalent
programs or measure mutual information. A full score includes 29 disjoint hidden
inputs, and is not a single noisy human label. Verifier fitting and movement
computation are not included in the label budget.

Evidence hierarchy:

1. A configuration sweep on one fixed bank tests algorithmic behavior conditional
   on that bank. Thousands of rows are not thousands of independent model samples.
2. Different audit seeds test acquisition variation, not generator variation.
3. Distinct generation seeds yield separate sampled banks from the same frozen
   checkpoint; the task population and model family are still shared.
4. The transfer-v1 suite has unseen specifications, but reuses a restricted
   expression grammar and was authored after seeing development findings.
5. No current experiment trains model weights or validates a general coding agent.

Confidence intervals resample tasks because optimizer settings and audit seeds
are correlated within tasks. Eight transfer tasks and two generator seeds do not
justify distribution-free deployment-risk guarantees. A correct finite-bank
certificate can be simultaneous over proposals while still saying nothing about
support outside that bank. These distinctions must remain in the paper.
