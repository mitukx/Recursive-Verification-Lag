# A conditional metric-covering label frontier

**Scope and novelty.** This is a proved finite-metric conditional statement,
not a novel general minimax theorem for recursive model improvement. Uniform
Lipschitz function recovery by covers is classical. The metric and its
constant must be fixed independently of the trusted outcomes used to claim
the guarantee. The diagnostic in `notes/progress/32_metric_assumption_audit.md`
shows that the current cheap representations often violate that assumption.

Let \((X,d)\) be a frozen finite candidate-source metric space. Cheap scores
\(s_x\) are known, trusted deterministic rewards \(r_x\in[0,1]\), and the
unknown error \(e_x=r_x-s_x\) is \(L\)-Lipschitz. An audit at a source reveals
its exact error. After at least one audited source \(A\), write the covering
radius \(\rho(A)=\max_{x\in X}d(x,A)\). For a baseline/current policy \(p\)
and a possibly transcript-adaptive proposal \(q\), write \(w=q-p\).

**Upper bound.** Nearest-audit error imputation gives an observable
\(\widehat G_A(p,q)=\sum_x w_x(s_x+e_{a(x)})\) and, for every pair of
distributions simultaneously,

\[
|G(p,q)-\widehat G_A(p,q)|
 \le L\sum_x |w_x|d(x,A)
 \le 2L\rho(A). \tag{1}
\]

This follows by subtracting the imputed errors and applying Lipschitzness
term by term; \(\sum|w_x|\le2\). Because the entire function obeys the
assumption, proposals and audit locations may adapt to previously paid
labels without invalidating (1). A sufficient certificate for this pair is
\(\widehat G_A-L\sum |w_x|d(x,A)>0\). Strictness is needed for a claim of
strict gain. The right side is **proposal-specific** and measures movement
in unresolved verification directions; the uniform covering radius is a
worst-case simplification, not the central RVL predictor.

**Transcript lower bound.** Fix any nonempty audited set \(A\), choose
\(u\) at distance \(\rho(A)\), and choose any audited \(a\). Set every cheap
score to 1/2 and each observed reward to 1/2. The two
full reward functions

\[
 r_x^\pm=\tfrac12\pm\min\{Ld(x,A),\tfrac12\}
\]

are both in [0,1], are \(L\)-Lipschitz (clipped distance to a set is
\(L\)-Lipschitz),
and produce the *same* audited transcript. For the allowed pair
\(p=\delta_a,q=\delta_u\), the true gains are
\(\pm\min\{L\rho(A),1/2\}\). Consequently,
no transcript-only estimate is uniformly more accurate than
\(\min\{L\rho(A),1/2\}\) on these two worlds, and no transcript-only sign decision is
correct in both. This lower bound is for **uniform accuracy over all policy
pairs** and the flat-score class; it does not imply every actual optimizer
proposal needs another label.

For a deterministic adaptive auditing rule that promises uniform additive
gain accuracy \(0<\varepsilon<1/2\) for every compatible reward and every policy
pair after \(k\) labels, run it on the all-1/2 transcript. Its final set A
must have \(\rho(A)\le\varepsilon/L\) (within the reward-range regime).
Any outcome of that transcript violating this inequality has two compatible
worlds on which the rule fails. Conversely, choose an
\(\varepsilon/(2L)\)-net before observing rewards and (1) gives the promised
accuracy. If \(N(t)\) is the minimum size of a \(t\)-net, this gives the
conditional label bracket

\[
N(\varepsilon/L)\;\le\; k_{\rm uniform}(\varepsilon)
\;\le\;N(\varepsilon/(2L)), \tag{2}
\]

for \(0<\varepsilon<1/2\) and \(L>0\). The
lower argument allows deterministic adaptive query schedules but does **not**
assert a standard randomized minimax lower bound: choosing the two worlds
after seeing a randomized realized audit set would be insufficient for that.
Do not label (2) an RVL-specific optimal-rate result. The goal is to find a
validated residual class and a **proposal-specific** complexity frontier
where verifier refresh timing and endogenous policy movement matter.

Existing safe policy improvement work provides an important overlap
boundary: Laroche et al., ICML 2019,
https://proceedings.mlr.press/v97/laroche19a.html . The central missing
ingredient is calibrated residual geometry on independent real program
tasks, not additional manipulations of covering-number algebra.

## A misspecification floor forced by feature collisions

Suppose instead \(e_x=f(x)+b_x\), with \(f\) \(L\)-Lipschitz in the fixed
observable metric and \(|b_x|\le\xi\). Then nearest-audit imputation obeys

\[
|G-\widehat G_A|\le
\sum_{x\notin A}|w_x|\bigl(Ld(x,A)+2\xi\bigr). \tag{3}
\]

Audited terms are zero exactly; unaudited terms use the triangle inequality
on \(f(x)-f(a)\) and \(b_x-b_a\). If distinct sources \(x,y\) share identical
features, \(d(x,y)=0\), but their verified errors differ by \(\Delta\), then
necessarily \(2\xi\ge|\Delta|\). This is a deterministic *lower bound on the
required misspecification allowance* for that representation, regardless of
how its Lipschitz component is fitted. It does not imply the resulting
allowance is itself sufficient or calibrated on future tasks. Auditing both
colliding sources removes them from (3), while a generic semantic grouping
of their rewards would be invalid.
