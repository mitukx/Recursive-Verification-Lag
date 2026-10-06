# Progress 32 — Cheap-feature residual smoothness is falsified on observed banks

## Hypothesis and preregistration status

Could a Lipschitz verifier-error model over currently available observable
features support a useful policy-movement certificate? This audit was
designed **after inspecting both Qwen-1.5B expression banks**. It is a
retrospective assumption check, not an independent predictive evaluation.

Within each task collapse repeated *identical source text* to one candidate.
Compare all distinct-source pairs using (i) public score alone and (ii) all
currently available public/syntactic feature columns. Inspect the residual
\(r-\text{public score}\) from exhaustive evaluation. If two sources have
identical features and different residuals, their distance is zero while
error differs: **no finite Lipschitz constant exists for that exact
representation**. These full-bank outcomes enter this diagnostic only;
they were not treated as free trusted labels in any controller experiment.

| Bank | Public only: contradictory pairs / tasks affected | All available features: contradictory pairs / tasks affected | Largest residual gap at zero feature distance |
| --- | ---: | ---: | ---: |
| v1, 8 tasks | 120 / 6 | 1 / 1 | .759 |
| v2, 8 tasks | 175 / 7 | 2 / 2 | .966 public; .552 all |

For the full syntactic representation, even a Lipschitz-plus-bounded-error
model \(e=f+b\), \(|b|\le\xi\), needs \(\xi\ge .379\) on the v1 collision
and \(\xi\ge .276\) on the v2 collision, by the half-gap argument in
`notes/theory/metric_covering_minimax.md`. These are retrospective
within-bank floors, not an independently calibrated envelope.

The full rows, including exact pair counts per task and post-hoc minimum
Euclidean Lipschitz constants where finite, are in
`results/residual_metric_collisions_v1.csv` and
`results/residual_metric_collisions_v2.csv`. These are small, deliberately
selected expression tasks; their counts are descriptive, with many related
pairs per task. The scalar Euclidean constant depends on feature scaling,
while zero-distance contradictions do not. Identical feature vectors also
force identical output from **any deterministic verifier that uses only those
features**, so changing the verifier's fitted coefficients cannot resolve
these particular contradictions. A semantically richer representation or
an explicit irreducible-error envelope could.

## Theoretical and empirical consequence

The metric-covering label result in
`notes/theory/metric_covering_minimax.md` is mathematically conditional;
plugging these public or syntactic metrics into it would produce an invalid
safety claim on affected tasks. This is a *negative result*, not a reason to
choose a sufficiently large constant after seeing the same trusted labels.
For RVL, the sharper target is a representation with prospectively calibrated
error structure **plus** the movement-weighted uncertainty
\(\sum_g|q_g-p_g|d(g,A)\), tested with matched paid labels.

Next decision: hold the conditional theorem as a guide, and prioritize
isolated scoring of the already locked MBPP+ bank. If cheap-feature
collisions persist on independently selected standard code tasks, retain
the assumption-light finite-bank box result and seek different structures
(e.g. independently calibrated functional equivalence or explicit bounded
misspecification). Do not claim a general metric law from either toy bank.

Reproduce: `python -m src.diagnose_residual_metric data/support_qwen15b_seed25.jsonl --output results/residual_metric_collisions_v1.csv`
and the same command for `data/transfer_v2_qwen15b_seed26.jsonl` with the v2
output filename.
