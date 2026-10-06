# The full safe policy region, beyond a one-dimensional update

**Scope.** Exact deterministic rewards shared by identical-source classes in
one frozen finite candidate bank, in [0,1]. The audit transcript contains only
paid source labels. No confidence or generalization statement follows.

Let \(b\) be the initial policy, \(q\) any proposed policy, \(A\) the audited
source classes, and \(q_g,b_g\) their aggregate probabilities. The sharp
worst-case gain over all reward assignments compatible with the transcript is

\[
L_A(q,b)=\sum_{g\in A}(q_g-b_g)r_g
            -\sum_{g\notin A}(b_g-q_g)_+.
\]

**Proposition (exact robust safe projection).** Let \(S_A=\{q\in\Delta:
L_A(q,b)\ge0\}\). Every \(q\in S_A\) has reward at least that of \(b\)
in every compatible world. Every \(q\notin S_A\) is worse than \(b\) in at
least one compatible world. Given a proposed \(z\), the smallest amount of
total variation needed to make it universally safe is exactly
\(\frac12\min_{q\in S_A}\lVert q-z\rVert_1\). This minimum is attained and
can be computed by the following linear program:

\[
\begin{aligned}
\min_{q,t,u}\;&\sum_i t_i\\
\text{s.t. }&q\ge0,\quad \sum_i q_i=1,\quad t_i\ge q_i-z_i,\quad t_i\ge z_i-q_i,\\
&u_g\ge b_g-q_g,\quad u_g\ge0\quad(g\notin A),\\
&\sum_{g\notin A}u_g-\sum_{g\in A}(q_g-b_g)r_g\le0.
\end{aligned}
\]

**Proof.** For each unobserved source, minimize
\((q_g-b_g)r_g\) by selecting reward zero when the mass change is positive
and reward one when it is negative. These choices are independent and can be
attained jointly, establishing necessity and sufficiency of \(L_A\ge0\).
The displayed constraints force \(u_g\ge(b_g-q_g)_+\), and taking equality
shows that the LP projects onto exactly \(S_A\). Likewise its minimum uses
\(t_i=|q_i-z_i|\). The simplex is compact and \(b\in S_A\), so an optimum
exists. The result remains valid for proposals selected adaptively from the
same paid transcript. QED.

**Why the prior line-search obstruction is limited.** The certified set is a
convex polytope. A chosen ray from the current policy can point outside its
tangent cone even when another direction stays inside it and approaches the
proposal. A strictly negative lower-bound derivative on one ray therefore
rules out movement on that ray alone. The projection finds the closest safe
point in the entire simplex; it may concentrate on audited source classes and
may change the optimizer's intended direction. It is not a theorem of strict
improvement, a certified refresh cadence, or a result for unseen programs.

This convex formulation is elementary robust optimization. Safe policy
improvement with a baseline predates RVL; in particular see Laroche et al.,
ICML 2019, https://proceedings.mlr.press/v97/laroche19a.html . The empirical
question is whether recursive verifier updates hit a directional bottleneck
that full safe-set projection resolves at the *same* paid-label budget.
Source-aware error structure and independently scored standard code tasks are
needed before broader theoretical or empirical claims are justified.

**Falsification and next step.** Test the same predeclared rule on a new,
isolated scored program bank; report loss of full-policy support, propensity
concentration, actual label identities, and progress against equal-cost fixed
refresh. If projections merely collapse onto a few audited programs, treat
the gain as finite-bank exploitation, not general safe recursive learning.
