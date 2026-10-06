# Finite-KL verifier-error geometry

The local criterion in `equal_kl_false_progress_geometry.md` extends exactly
along the whole exponential-update path.

Let

\[
q_\beta(i)=\frac{p(i)e^{\beta v(i)}}{Z(\beta)}
\]

for a finite-support baseline \(p\), optimized verifier/proxy score \(v\), and
trusted quantity \(y\). For any bounded \(f\),

\[
\frac{d}{d\beta}\mathbb E_{q_\beta}[f]
=\operatorname{Cov}_{q_\beta}(f,v).
\]

Integrating from zero to \(\beta\) gives the exact identity

\[
\mathbb E_{q_\beta}[y]-\mathbb E_p[y]
=\int_0^\beta \operatorname{Cov}_{q_t}(y,v)\,dt.
\]

So true progress is the **path integral of verifier/truth alignment covariance**,
not a function of endpoint KL alone.

For the optimized proxy itself,

\[
\mathbb E_{q_\beta}[v]-\mathbb E_p[v]
=\int_0^\beta \operatorname{Var}_{q_t}(v)\,dt\ge 0.
\]

Thus proxy improvement is built into the exponential optimizer even while true
progress can be negative.

The endpoint KL obeys

\[
\frac{d}{d\beta}D_{\mathrm{KL}}(q_\beta\|p)
=\beta\operatorname{Var}_{q_\beta}(v),
\]

and therefore

\[
D_{\mathrm{KL}}(q_\beta\|p)
=\int_0^\beta t\operatorname{Var}_{q_t}(v)\,dt.
\]

For \(\beta>0\) where the score is nonconstant, KL can be used as a path
parameter:

\[
\frac{d\mathbb E_q[y]}{dD_{\mathrm{KL}}}
=
\frac{\operatorname{Cov}_{q_\beta}(y,v)}
{\beta\operatorname{Var}_{q_\beta}(v)}.
\]

This makes the insufficiency of KL precise. KL measures weighted proxy-score
variance accumulated along the path; true progress depends on a different
integrand, verifier/truth covariance. Equal KL does not imply equal true
progress unless additional geometry assumptions couple those two quantities.

## Error decomposition

Write the verifier as

\[
v=y+e
\]

for verifier error \(e\). Then at every point on the path,

\[
\operatorname{Cov}_{q_t}(y,v)
=
\operatorname{Var}_{q_t}(y)
+
\operatorname{Cov}_{q_t}(y,e).
\]

The first term is the useful signal. The second is verifier-error alignment.
False progress occurs when the negative alignment term dominates after
integration:

\[
\int_0^\beta
\left[
\operatorname{Var}_{q_t}(y)
+
\operatorname{Cov}_{q_t}(y,e)
\right]dt < 0,
\]

while proxy progress remains nonnegative.

This gives a useful interpretation of verification debt: optimization can move
the policy into regions where the verifier error is increasingly anti-aligned
with true reward, and the accumulated anti-alignment can overwhelm the true
signal even when the total policy KL is modest.

## Same-KL counterexample

Two verifier paths can be calibrated to the same endpoint KL while having
opposite signs of

\[
\int_0^\beta\operatorname{Cov}_{q_t}(y,v)dt.
\]

The repository regression test constructs exactly this case: both arms improve
their own proxy and end at KL 0.12 from the same baseline, yet one decreases
trusted reward and the other increases it.

This is stronger than the local small-step statement. It shows that **there is
no endpoint-KL-only safety rule even for the exact finite exponential family**
without assumptions tying verifier error geometry to trusted reward.

## Numerical verification

`src/finite_kl_alignment_identity.py` implements:
- direct exponential tilting;
- Gauss-Legendre evaluation of the three path integrals above;
- the exact signal/error covariance decomposition.

`tests/test_finite_kl_alignment_identity.py` checks the identities over random
finite-support problems, verifies the same-KL opposite-progress counterexample,
and checks the decomposition numerically.

This result is mathematical mechanism evidence. It does not imply that a learned
LLM verifier has a particular error-alignment trajectory. The real-model bridge
in `configs/qwen_alignment_bridge_v1.json` is designed to test that external
validity separately.
