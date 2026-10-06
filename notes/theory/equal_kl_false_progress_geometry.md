# Equal-KL false progress from verifier-error alignment

This note isolates a local mechanism. It is an elementary finite-support
calculation, not a claim that real learned verifiers obey the construction.

Let a baseline policy be \(p\) on finite support and let \(v\) be the score
optimized by an exponential update

\[
q_\beta(i)=\frac{p(i)e^{\beta v(i)}}{Z(\beta)},\qquad \beta\ge 0.
\]

For any bounded quantity \(f\),

\[
\frac{d}{d\beta}\mathbb E_{q_\beta}[f]
=\operatorname{Cov}_{q_\beta}(f,v).
\]

Therefore at the baseline,

\[
\left.\frac{d}{d\beta}\mathbb E_{q_\beta}[y]\right|_{\beta=0}
=\operatorname{Cov}_p(y,v).
\]

By contrast, the optimized proxy necessarily improves locally whenever the
score is nonconstant:

\[
\left.\frac{d}{d\beta}\mathbb E_{q_\beta}[v]\right|_{\beta=0}
=\operatorname{Var}_p(v)>0.
\]

The KL displacement satisfies

\[
\frac{d}{d\beta}D_{\mathrm{KL}}(q_\beta\|p)
=\beta\operatorname{Var}_{q_\beta}(v)\ge 0,
\]

so each sufficiently small positive KL radius identifies a unique positive
\(\beta\) along a nonconstant score path.

## Controlled error geometry

Write the centered true reward as

\[
y-\mathbb E_p[y]=s_y\,\widetilde y,
\qquad
\mathbb E_p[\widetilde y]=0,
\qquad
\mathbb E_p[\widetilde y^2]=1.
\]

Choose a nuisance direction \(u\) with

\[
\mathbb E_p[u]=0,\quad
\mathbb E_p[u^2]=1,\quad
\mathbb E_p[u\widetilde y]=0.
\]

For error magnitude \(\sigma>0\) and alignment \(\rho\in[-1,1]\), define

\[
e=\sigma\left(\rho\widetilde y+\sqrt{1-\rho^2}\,u\right),
\qquad
v=\widetilde y+e.
\]

Then

\[
\operatorname{Cov}_p(y,v)
=s_y(1+\sigma\rho).
\]

Hence, for sufficiently small positive updates,

- \(1+\sigma\rho>0\): proxy and true reward initially improve together;
- \(1+\sigma\rho<0\): the proxy improves while true reward decreases;
- \(1+\sigma\rho=0\): first-order true progress vanishes and higher-order
  terms determine the finite-KL sign.

The error norm is held fixed at \(\|e\|_{L_2(p)}=\sigma\). Changing only
\(\rho\) rotates the same-magnitude verifier error relative to the reward
direction. Matching KL across those interventions therefore holds total policy
displacement fixed while changing verifier-error geometry.

This gives a sharp local counterexample to any rule that treats policy KL alone
as sufficient to determine safe verifier reuse: two updates can have the same
KL and the same verifier-error norm but opposite true-progress signs.

## Prospective stress test

The locked protocol in
`configs/equal_kl_alignment_stress_v1.json` evaluates this prediction across
512 independently generated finite-support tasks, four error magnitudes, nine
alignment values and four matched KL budgets. The primary endpoint is fixed at
KL 0.005 before execution. Exact boundary cells are excluded from the sign
prediction, and unattainable KL cells are retained as such rather than silently
substituted.

The experiment is intentionally narrower than an LLM result. Its value is to
identify a mechanism that the real-model RVL experiments can later test:
**false progress should track verifier-error alignment conditional on policy
shift, not policy shift alone.**
