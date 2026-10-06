# Progress 41 — active geometry-aware trusted-label auditing

This step turns the verifier-geometry mechanism into a more operational
controller question: if trusted labels are expensive, where should they be
queried?

A new active audit samples state (i) with probability proportional to

[
p_i |v_i-mathbb E_p[v]|.
]

For the importance-weighted covariance estimator, this proposal minimizes the
worst-case absolute coefficient over all proposal distributions without using
trusted outcomes. The passive control samples directly from the current policy.

The evaluation protocol is locked before outcomes in
`configs/active_alignment_audit_v1.json`. It uses 1,024 task seeds disjoint
from exploratory seeds, harmful/benign verifier-error geometry at fixed
(sigma=2), and label budgets from 16 to 512. The primary endpoint at 256
labels asks whether active auditing blocks the harmful arm substantially more
often than passive sampling while maintaining a low wrong-sign decision rate.

This is deliberately narrower than claiming a generic scalable-oversight
algorithm. It tests whether the project's proposed verifier-error geometry is
not only explanatory, but can also guide label acquisition efficiently in a
controlled setting.
