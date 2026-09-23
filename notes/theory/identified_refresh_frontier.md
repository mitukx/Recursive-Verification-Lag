# A sharp observable frontier for bounded finite-bank rewards

**Status:** proved elementary identification proposition; not a novel
statistical confidence bound, an optimal refresh-rate theorem, or a general
claim about real-model deployments. It clarifies exactly what trusted labels
can establish without verifier-error assumptions. The proof applies to frozen
finite support and deterministic trusted reward per identical-source class.

Let the finite candidate occurrences be grouped into source classes (g),
with common but unknown reward (r_g\in[a_g,b_g]). At a decision point, the
current and proposed distributions (q,p) and the paid audit transcript are
observable. Write (w_g=\sum_{i\in g}(p_i-q_i)). For audited classes (A),
the transcript reveals exact (r_g); for others there are only the specified
bounds. Define

\[
L_A(p,q)=\sum_{g\in A}w_gr_g+\sum_{g\notin A}\min\{w_ga_g,w_gb_g\},
\quad
U_A(p,q)=\sum_{g\in A}w_gr_g+\sum_{g\notin A}\max\{w_ga_g,w_gb_g\}.
\]

**Proposition (exact gain-identification interval).** Conditional on this
transcript and these assumptions, the set of feasible true gains
\(\mathbb E_p r-\mathbb E_q r\) is exactly \([L_A,U_A]\). In particular,
\(L_A\ge0\) certifies nonnegative gain for every compatible reward function.
If \(L_A<0<U_A\), there are two compatible worlds with opposite gain signs;
any refresh/stop rule using only the same transcript makes the same decision
in both. A new audit, a structural assumption on errors, or abstention is
necessary to resolve the sign. With group-independent reward intervals,
the interval width is exactly
\(\sum_{g\notin A}|w_g|(b_g-a_g)\).

**Proof.** The gain is \(\sum_g w_gr_g\). Audited summands are fixed.
For every unaudited class, the linear summand reaches its min and max at the
corresponding interval endpoints. Choices across unaudited classes are
independent, so these endpoint choices can be attained simultaneously; every
intermediate gain follows by convex interpolation. When zero lies strictly
inside, the two endpoint assignments generate identical observable policies,
cheap scores and audit transcript but have opposite true gains. Width follows
by subtracting the endpoint sums. QED.

For safety **relative to the initial policy**, set \(q=p_0\), not the last
refresh policy. The controller may compute \(p\) adaptively from previous
trusted labels: the proposition is deterministic and still holds conditional
on the realized transcript. To act before an update, evaluate the *proposed*
policy and require \(L_A\ge0\); if it fails, auditing extra high-impact
sources can shrink the interval. Auditing alone cannot guarantee that a useful
proposal becomes safe: the identified interval may still straddle zero even
after every allowed label is acquired, or the true gain may be negative.

This is a source-coverage frontier, not a single universal density-ratio law.
The terms \(|w_g|(b_g-a_g)\) directly show which unqueried reward directions
matter. With no structure, confident improvement requires covering those
directions or accumulating enough positive audited contribution. If the cheap
verifier or task specification constrains reward jointly, this interval can
be narrowed with a justified coupled feasible set; arbitrary per-class
intervals deliberately avoid such assumptions. The experiment below checks
how often this robust certificate is informative. It cannot turn the algebra
into a statistical warning guarantee.
