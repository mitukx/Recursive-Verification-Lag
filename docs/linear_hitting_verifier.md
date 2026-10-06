# Linear Hitting Verifier

The first #116-inspired implementation used all 12 task identities as a fixed
probe basis. On the retained Qwen bank that design was safe but too
conservative: the fixed-probe controller refreshed every proposal even at 48
trusted labels.

This follow-up changes the abstraction. Instead of "probe every task", declare a
bounded verifier-error class

[
e=v-y=Phieta+r,qquad lVert rVert_inftyleho.
]

A deterministic, label-blind set of rows (H) is chosen so that
(Phi_H) spans the row space of (Phi). Only those trusted labels are needed
to form

[
widehat e=PhiPhi_H^+e_H.
]

For any future policy contrast (delta=q-p),

[
widehat g=delta^	op(v-widehat e)
]

and the implementation proves/uses the finite-dimensional bound

[
|g_{m true}-widehat g|
le
holeft(
lVertdeltaVert_1+
lVertdelta^	opPhiPhi_H^+Vert_1
ight).
]

Therefore the same fixed trusted probes can be reused for arbitrarily many
future policy shifts **if the declared tubular error class remains valid**.

This is an elementary linear-algebra result inspired by the structural idea of
OpenAI Math family #116. It is not a corollary of the noncommutative identity
testing result.

## Why this is a technical step beyond cadence refresh

A cadence controller asks when to buy new labels. A hitting verifier instead
asks whether the current verifier error lies in a class already distinguishable
by a fixed trusted basis. Inside that class, the trusted cost can be amortized
across recursive generations.

The Qwen benchmark compares two hand-specified classes:

- `global_features`: one global public-feature error model;
- `taskwise_features`: a richer block-diagonal model with separate feature
  coefficients per task.

For each class the benchmark records feature rank, required fixed probe count,
a #140-inspired `rank * log(1/epsilon)` information-budget proxy, and the
full-label least-squares residual as an evaluator-only misspecification
diagnostic.

The locked residual-radius grid is evaluated conditionally. In addition, an
`evaluator_sufficient` row uses the full-label least-squares residual radius
only to answer: *if a valid residual bound of this size were known, would the
fixed hitting set resolve the real Qwen proposals?* That row is diagnostic and
cannot be used as an online controller.

## Evidence boundary

The bank is previously inspected development data, so this is a post-hoc
mechanism study. The theorem contract is genuine under its explicit bounded
linear-error assumption; whether that class fits frontier verifier errors is an
empirical question, not assumed.

The CI workflow tests the exact and bounded-residual theorem on synthetic data,
runs the retained-Qwen benchmark, independently checks every interval for rows
whose residual-class assumption has an evaluator witness, and retains all raw
rows.
