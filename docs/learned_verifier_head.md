# Learned verifier head v1

Issue #66 names a learned verifier but previously had no reusable runtime
component: the neural-head implementation lived inside the one-off Qwen
learned-verifier bridge. This module extracts the same prospectively used design
into a deployable verifier.

The representation is the mean of the final causal-LM hidden states over the
exact response-token IDs already carried by token-exact replay. The head is:

`LayerNorm(hidden) -> Linear(hidden, 128) -> GELU -> Linear(128, 1)`

and training uses full-batch AdamW with class-balanced
`BCEWithLogitsLoss(pos_weight=n_negative/n_positive)`. The defaults match the
existing learned-verifier experiment; tests may reduce width/steps only for
contract speed.

Artifacts contain numeric head tensors plus JSON metadata in compressed NumPy
format and are loaded with `allow_pickle=False`. The causal-LM backbone is
supplied explicitly by the worker process and must match the artifact's
`model_identity`. No model code, Python object, or optimizer object is loaded
from the artifact.

The component is compatible with the normal verifier protocol and with the
two-phase verifier deployment control plane. A new fitted head therefore becomes
a new immutable verifier artifact/version; `refresh()` on a deployed head is
intentionally unsupported.

This is a reward-model component and deployment primitive, not evidence that the
head is aligned or useful. Trusted-label fit eligibility and held-out evaluation
remain separate experimental gates.
