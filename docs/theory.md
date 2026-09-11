# Theory Ledger

This document lists the main theoretical results and their current status. It intentionally distinguishes proved results, standard machinery, architecture-specific claims, and open targets.

## 1. Adaptive moving-tail minimax law — proved in the hard family

For a recursive binary-tree environment where each round exposes a fresh rare reward sign, the probability of classifying all updates correctly satisfies

\[
\Pr(\text{all correct})
\le
\prod_t\left[1-\frac12(1-1/M_t)^{m_t}\right].
\]

The resulting heterogeneous fixed-budget complexity is

\[
B^\star_{\rm current}
=\Theta\!\left(\sum_t h_t\log\frac{H}{\delta h_t}\right),
\quad
h_t=[-\log(1-1/M_t)]^{-1}.
\]

For equal `M`,

\[
B^\star_{\rm current}=\Theta(TM\log(T/\delta)).
\]

Candidate-aware auditing gives

\[
B^\star_{\rm candidate}=\Theta(T\log(T/\delta)).
\]

The logarithm is a fixed-budget/high-probability latency term; expected sequential discovery has different scaling.

## 2. Noisy trusted verification — proved in the hard family

If candidate-salient audit mass is `a` and the label-world Bhattacharyya coefficient is `rho`, the local information rate is

\[
I(a,\rho)=-\log[(1-a)+a\rho].
\]

For Bernoulli margin/noise `gamma`,

\[
I(a,\gamma)=\Theta(a\gamma^2),
\]

so rarity and label noise multiply.

## 3. Structured verification complexity — proved under stated regularity conditions

For a closed linear error/reward class `F`, define

\[
\mathcal V_{\mathcal F}(p,q;\mu)
=
\sup_{f\in\mathcal F,f\ne0}
\frac{(\mathbb E_qf-\mathbb E_pf)^2}{\mathbb E_\mu[f^2]}.
\]

The one-step minimax trusted-label complexity is

\[
m^\star
=\Theta\!\left(
\frac{\mathcal V_{\mathcal F}(p,q;\mu)}{\Gamma^2}
\log\frac1\delta
\right)
\]

under the local boundedness condition used by the lower bound.

For linear features,

\[
\mathcal V_{\mathcal F}=d^T\Sigma_\mu^\dagger d.
\]

This structured coefficient is standard-adjacent to restricted-chi-square results in off-policy evaluation; the intended novelty is its role inside a recursive optimizer-verifier loop.

## 4. External verification entropy — proved

If correct recursive operation identifies one of `K` possible reward worlds and each trusted observation supplies conditional mutual information at most `C_i`, then

\[
\sum_i C_i
\ge
\log K-h(\alpha)-\alpha\log(K-1).
\]

Secrecy can reduce gaming but cannot create missing reward information.

## 5. Same-verifier self-evaluation blindness — proved for canonical plug-in architecture

For

\[
\widehat\theta=\theta+e,
\qquad
d=\eta\Sigma\widehat\theta,
\]

the same learned verifier reports

\[
\widehat\Delta
=\eta\|\theta+e\|_\Sigma^2\ge0.
\]

This is an architecture-specific result, not a universal theorem about all self-evaluation.

## 6. Rank-one candidate-audit effect — proved in the stated linear model

For

\[
G_\lambda=\Sigma+\lambda dd^T,
\]

\[
\operatorname{tr}(\Sigma G_\lambda^{-1})
=(p-1)+\frac1{1+\lambda M},
\]

and

\[
\operatorname{tr}[(\Sigma G_\lambda^{-1})^2]
=(p-1)+\frac1{(1+\lambda M)^2}.
\]

Candidate mixing fixes only one rank-one optimism direction in this model.

## 7. Misspecification zero crossing — proved in the quadratic model

For

\[
r(y)=\theta^Ty+\frac12y^TQy,
\]

and adverse curvature,

\[
\eta_{\max}
=
\frac{2\theta^T\Sigma\theta}{|\theta^T\Sigma Q\Sigma\theta|}.
\]

This is the point at which the true gain of the fixed update family crosses zero. It is **not** a universal verification ceiling.

## 8. Composed stale-candidate theorem — conditional bridge theorem

If a verifier is frozen over several exponential updates, the endpoint depends on cumulative strength

\[
\Lambda_s=\sum_k\eta_{s,k}.
\]

Conditional on the endpoint candidate being fixed before the current fresh certification batch, the structured one-step theorem applies directly:

\[
m_s
=O\!\left(
\frac{\mathcal V_{\mathcal F_s}(p_s,q_{s:L};\mu_s)}{\Gamma_s^2}
\log\frac1{\delta_s}
\right).
\]

A matching conditional lower bound holds when the block probes a reward direction unresolved by the old transcript.

This is a bridge theorem rather than a standalone novelty claim.

## Open theoretical target

The strongest remaining question is an **observable adaptive refresh frontier**: can one choose when to refresh the verifier, without oracle access to the true gain, so as to maximize useful progress per trusted label while keeping familywise recursive validity?
