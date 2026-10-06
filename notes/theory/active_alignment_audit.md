# Active trusted-label auditing by verifier geometry

The existing covariance audit samples trusted labels from the current policy
(p) and estimates

[
C=operatorname{Cov}_p(y,v)
 = mathbb E_p[y(v-mathbb E_p[v])].
]

When trusted labels are costly but the current policy probabilities and verifier
scores are available, the label-query distribution itself can be optimized.

Let (a_i=v_i-mathbb E_p[v]), and sample an index (Isim r). The
importance-weighted single-sample estimator is

[
X=rac{p_I}{r_I}y_I a_I.
]

For (y_iin[0,1]), its worst-case absolute coefficient is
(max_i |p_i a_i/r_i|). Minimizing this over all proposal distributions gives

[
r_i^star=rac{p_i|a_i|}{sum_j p_j|a_j|},
]

on nonzero coefficients. The proof is the usual equalization argument: if
(M=max_i |p_i a_i/r_i|), then
(1=sum_i r_ige sum_i |p_i a_i|/M), hence
(Mgesum_i p_i|a_i|), and equality is attained by (r^star).

This is useful because it uses only proxy geometry—no trusted labels—to target
queries toward states with large leverage on the verifier/truth covariance.
The estimator remains unbiased, and the repository uses a distribution-free
Hoeffding interval computed from the exact importance-coefficient range.

The prospective experiment in `configs/active_alignment_audit_v1.json`
compares this active proposal against passive policy sampling on 1,024 disjoint
evaluation seeds. Thresholds were selected on separate development seeds and
locked before evaluation. The experiment is a finite-support mechanism/control
study, not evidence that the proposal is optimal for a learned LLM verifier or
real production traffic.
