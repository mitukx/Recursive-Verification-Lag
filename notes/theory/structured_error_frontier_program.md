# Structured verification frontier: an explicit research target

**Objective.** Move from a vacuous worst-case box to a *validated* verifier
error class that can predict when policy movement needs new trusted labels.
This is a research program, not a completed generalization theorem.

## One conditional bridge, with a matching two-world obstruction

Let \(g\) denote distinct program source classes, \(s_g\) the public score,
and \(e_g=r_g-s_g\) the verifier error. Before seeing outcomes, fix a metric
\(d\) on observable program features and a constant \(L\). Assume
\(|e_g-e_h|\le Ld(g,h)\) simultaneously for all classes in a **frozen** bank.
For each class \(g\), let \(a(g)\) be a nearest audited source; its residual
is known from a paid trusted reward. For an adaptive policy shift
\(w_g=q_g-p_g\), the imputed gain

\[
\widehat G_A(p,q)=\sum_g w_g(s_g+e_{a(g)})
\]

satisfies, simultaneously for every policy pair built from the transcript,

\[
|G(p,q)-\widehat G_A(p,q)|
\le L\sum_g |w_g|d(g,A). \tag{1}
\]

**Proof:** subtract and use the Lipschitz condition separately for each
term, followed by the triangle inequality. Since the event concerns the
entire error function, selecting \(q\) adaptively creates no extra
per-proposal union bound. If (1) fails on the real bank, the assumption has
failed and so has the certificate; a fitted post-hoc Lipschitz constant is
not a valid guarantee.

The dependence on unresolved movement is necessary already in a two-source
example. With sources \(a,u\), \(d(a,u)=D\), audit only \(a\), take public
scores \(s_a=s_u=1/2\), reward \(r_a=1/2\), and let \(p\) put all mass on
\(a\) and \(q\) put all mass on \(u\). For \(LD\le1/2\), the worlds
\(r_u^+=1/2+LD\) and \(r_u^-=1/2-LD\) both satisfy the assumptions and yield
opposite true gains \(\pm LD\). The right side of (1) is \(LD\), attained
in both worlds. A safe/strictly-progressing decision must audit \(u\), have
additional information, or abstain. This is a *restricted two-world
obstruction*, not a general minimax rate.

## Hard questions that would make this a stronger theory paper

1. Derive an instance-dependent **upper and lower label complexity** under a
   stated residual class with a calibrated constant, allowing the policy and
   verifier to change after each label. Query order must be predictable from
   available information. Equal-cost comparison is essential.
2. Replace the nearest-audit upper bound by the sharp coupled Lipschitz
   residual LP and determine when the observable bound is informative. This
   must account for reward bounds and score errors across the whole bank.
3. Establish calibration on independent source groups or a simultaneous
   confidence event for adaptive histories. Without it, (1) is only a
   conditional deterministic identity. A representation learned using
   trusted labels must be covered by the event.
4. Pre-register a new program bank and test whether calibrated residual
   geometry beats KL, max log density ratio, and naive age under matched
   sources and label budgets. Keep counterexamples where metric geometry
   fails and report policy concentration after projection.

Any eventual minimax claim must cite the prior safe policy improvement and
active learning literature, and specify its distinct recursive coupling.
The nearest-neighbor inequality itself is elementary and should not be
marketed as an Oral-level theorem. If no representation satisfies a useful
predeclared error condition on independently scored programs, do not build
the research story around that condition.
