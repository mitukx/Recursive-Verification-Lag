# Trusted-label complexity for a verifier-alignment gate

For exponential proxy optimization, the local derivative of trusted reward is

\[
\left.\frac{d}{d\beta}\mathbb E_{q_\beta}[y]\right|_{\beta=0}
=\operatorname{Cov}_p(y,v).
\]

Suppose the proxy score \(v(i)\) is available cheaply on the candidate support
and trusted reward \(y(i)\in[0,1]\) is expensive. Let

\[
\mu_v=\mathbb E_p[v]
\]

be known from the proxy scores and define

\[
X=y(i)(v(i)-\mu_v),\qquad i\sim p.
\]

Then

\[
\mathbb E[X]=\operatorname{Cov}_p(y,v).
\]

If \(X\in[a,b]\) and \(R=b-a\), Hoeffding gives

\[
\Pr\left(
\left|\widehat C_n-C\right| >
R\sqrt{\frac{\log(2/\delta)}{2n}}
\right)\le\delta,
\]

where \(C=\operatorname{Cov}_p(y,v)\).

Therefore a fail-closed local gate can use

\[
L_n=\widehat C_n-r_n,\qquad
U_n=\widehat C_n+r_n,
\]

with

\[
r_n=R\sqrt{\frac{\log(2/\delta)}{2n}}.
\]

Decision rule:

- if \(U_n<0\), block the proxy update or force verifier refresh;
- if \(L_n>0\), certify only the **local trusted-progress direction**;
- otherwise remain inconclusive and acquire more trusted labels.

Whenever the confidence interval covers the true covariance, a decisive sign
cannot be wrong. To certify a margin \(|C|\ge\gamma\), the sufficient label
count is

\[
n >
\frac{R^2}{2\gamma^2}\log\frac{2}{\delta}.
\]

This is not a certificate for an arbitrarily large finite update. It is a local
directional audit. For recursive optimization it can be reapplied before each
small block; the finite-KL path identity in
`finite_kl_verifier_geometry.md` explains why repeated alignment checks are the
relevant object.

## Why a shift-only gate cannot replace this audit

At the baseline, two interventions can share the same policy, hence identical
KL, density ratio and every other policy-only shift coordinate, while having
opposite \(\operatorname{Cov}_p(y,v)\). Any gate that observes only policy
shift must make the same decision on both. Trusted verifier/truth information is
therefore information-theoretically necessary for distinguishing these paired
worlds unless additional cross-task structure is assumed.

## Prospective controller experiment

The locked experiment in
`configs/alignment_covariance_audit_v1.json` uses 1,024 independent
finite-support tasks, equal verifier-error norm \(\sigma=2\), and opposite
alignment arms \(\rho=-1\) and \(+1\). It never uses policy KL in the
decision.

Observed fixed-budget results:

| trusted labels | harmful block | benign allow | wrong-sign decisive rate |
|---:|---:|---:|---:|
| 4 | 0.3477 | 0.3242 | 0 |
| 8 | 0.6035 | 0.6260 | 0 |
| 16 | 0.8408 | 0.8213 | 0 |
| 32 | 0.9775 | 0.9775 | 0 |
| 64 | 1.0000 | 0.9971 | 0 |
| 128 | 1.0000 | 1.0000 | 0 |

Coverage remains about 99% or higher across the reported budgets. These are
synthetic fixed-budget measurements, not a learned-LLM calibration claim.

The experiment converts the mechanism result into an actionable bounded
controller: measure verifier/truth alignment with paid audits, block when the
upper confidence bound is negative, and request more information rather than
treating low KL as evidence of safety.
