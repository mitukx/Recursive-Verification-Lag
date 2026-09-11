# Progress Update XXII — Composed stale-candidate verification theorem

**Date:** 2026-09-11  
**Status:** Core blockwise theorem proved as a conditional corollary of the existing structured one-step theorem. The result cleanly connects the early minimax theory to the later recursive-lag experiments, but **is not by itself a new central theorem**: the proof is one-step minimax verification + exact policy composition + conditional familywise accounting.

---

## 1. Why this update matters

The empirical sequence XVIII–XXI changed the project in an important way.

The naive recursive claim

> “verification becomes hard because many rounds have elapsed”

is false.

The better empirical claim was

> “verification becomes stale when the optimizer moves far under a verifier whose error geometry is no longer controlled.”

Update XXI then showed that even raw policy-shift magnitude is insufficient across verifier classes.

The natural mathematical question is therefore:

> **Can the recursive stale-verifier problem be reduced exactly to the already-proved one-step structured verification problem applied to the composed endpoint candidate?**

The answer is **yes**, under a clean temporal-independence condition.

This yields a precise result:

\[
\boxed{
\text{block verification cost}
\asymp
\frac{
\mathcal V_{\mathcal F}
(p_{\rm start},q_{\rm end};\mu)
}{
\Gamma^2
}
\log\frac1\delta,
}
\]

where \(q_{\rm end}\) is the policy reached after all stale updates in the block.

The number of stale steps \(L\) matters only through the **composed endpoint distribution** and the resulting true margin.

---

# 2. Base one-step theorem being reused

The v15 structured-verification theorem fixes a current policy \(p\), candidate \(q\), audit distribution \(\mu\), and a closed linear reward/error class

\[
\mathcal F\subseteq L_2(\mu).
\]

For

\[
L_{p,q}(f)
=
\mathbb E_q[f]-\mathbb E_p[f],
\]

let \(g_{\mathcal F}\) be the Riesz representer:

\[
L_{p,q}(f)
=
\langle g_{\mathcal F},f\rangle_{L_2(\mu)}.
\]

Define

\[
\boxed{
\mathcal V_{\mathcal F}(p,q;\mu)
=
\|g_{\mathcal F}\|_{L_2(\mu)}^2
=
\sup_{f\in\mathcal F,\ f\ne0}
\frac{
(\mathbb E_qf-\mathbb E_pf)^2
}{
\mathbb E_\mu[f^2]
}.
}
\]

For bounded trusted labels whose conditional mean is \(f\), the minimax one-step sign-classification cost is, under the local boundedness condition for the lower bound,

\[
\boxed{
m^\star
=
\Theta\!\left(
\frac{
\mathcal V_{\mathcal F}(p,q;\mu)
}{
\Gamma^2
}
\log\frac1\delta
\right).
}
\]

The upper bound uses the Riesz-weighted statistic and a median-of-means estimator.

The lower bound uses the least-favorable directions

\[
f_\pm
=
\pm
\frac{\Gamma}{V}g_{\mathcal F}
\]

and a Bretagnolle–Huber testing argument.

The present update does not re-prove that theorem from scratch; it lifts it to recursively generated stale blocks.

---

# 3. Exact composition of a stale exponential-update block

Let a verifier score \(v_s(y)\) be frozen at the beginning of a block.

Start from policy

\[
p_{s,0}=p_s.
\]

For \(k=0,\ldots,L-1\), perform

\[
p_{s,k+1}(dy)
=
\frac{
\exp(\eta_{s,k}v_s(y))
p_{s,k}(dy)
}{
\mathbb E_{p_{s,k}}
[\exp(\eta_{s,k}v_s)]
}.
\]

Define cumulative stale strength

\[
\Lambda_s
=
\sum_{k=0}^{L-1}\eta_{s,k}.
\]

## Lemma XXII.1 — Exact stale-block composition

\[
\boxed{
p_{s,L}(dy)
=
\frac{
\exp(\Lambda_s v_s(y))p_s(dy)
}{
\mathbb E_{p_s}
[\exp(\Lambda_s v_s)]
}.
}
\]

### Proof

Induction.

For one step the formula is the update definition.

Assume

\[
p_{s,k}(dy)
=
\frac{
e^{\Lambda_{s,k}v_s(y)}p_s(dy)
}{
Z(\Lambda_{s,k})
}
\]

with

\[
\Lambda_{s,k}
=
\sum_{\ell<k}\eta_{s,\ell}.
\]

Then

\[
p_{s,k+1}(dy)
\propto
p_{s,k}(dy)e^{\eta_{s,k}v_s(y)}
\propto
p_s(dy)
e^{(\Lambda_{s,k}+\eta_{s,k})v_s(y)}.
\]

Normalizing gives the claim. \(\square\)

### Immediate consequence

For a frozen verifier,

\[
\boxed{
\text{the endpoint depends on the stale update schedule only through }
\Lambda_s.
}
\]

In particular, for constant per-step strength \(\eta\),

\[
\Lambda_s=L\eta.
\]

This is why the empirical \(\eta L\) collapse in Update XX was so strong.

It is also why raw round count \(L\) cannot be fundamental.

---

# 4. Block-compression principle

Let

\[
q_s:=p_{s,L}
\]

be the endpoint of a stale block.

The true net block gain is

\[
\Delta_s
=
\mathbb E_{q_s}[r]
-
\mathbb E_{p_s}[r].
\]

For the purpose of deciding whether the block as a whole is beneficial, the entire internal path

\[
p_{s,1},\ldots,p_{s,L-1}
\]

can be discarded.

The statistical verification problem is exactly the one-step comparison

\[
p_s
\longrightarrow
q_s.
\]

## Proposition XXII.2 — Block compression

Assume:

1. true reward is static during the block;
2. the acceptance objective concerns the **net endpoint gain**
   \[
   \Delta_s
   =
   \mathbb E_{q_s}r-\mathbb E_{p_s}r;
   \]
3. audit observations depend on the reward and sampled outcome, not on the hidden path used to produce \(q_s\).

Then any two optimization paths with the same start \(p_s\) and the same endpoint \(q_s\) induce the same endpoint verification problem.

Therefore their minimax trusted-label complexities under a fixed \((\mathcal F_s,\mu_s,\Gamma_s,\delta_s)\) are identical.

### Important limitation

This proposition does **not** apply if the safety requirement is

\[
\Delta_{s,k}\ge0
\qquad
\text{for every intermediate update }k.
\]

If intermediate policies are deployed or irreversible, the path matters.

Thus there are two distinct operational goals:

- **block-end / rollback safety**: only the accepted endpoint must improve;
- **strict monotonicity**: every intermediate update must improve.

The composed theorem concerns the first.

---

# 5. Conditional recursive setup

Let

\[
\mathcal G_s
\]

be the complete transcript **after the block endpoint \(q_s\) has been generated but before the trusted certification batch for that block is observed**.

Conditional on \(\mathcal G_s\), assume the following are fixed / measurable:

- \(p_s\);
- \(q_s\);
- audit distribution \(\mu_s\);
- reward/error class \(\mathcal F_s\);
- required margin \(\Gamma_s\);
- error allocation \(\delta_s\).

Now acquire a fresh trusted batch

\[
S_s
=
\{(Y_{s,i},Z_{s,i})\}_{i=1}^{m_s},
\]

with

\[
Y_{s,i}\mid\mathcal G_s
\stackrel{\rm iid}{\sim}
\mu_s
\]

and bounded trusted labels satisfying

\[
\mathbb E[
Z_{s,i}
\mid
Y_{s,i}=y,\mathcal G_s
]
=
f(y),
\qquad
f\in\mathcal F_s.
\]

The crucial condition is:

\[
\boxed{
q_s\text{ is generated before seeing }S_s.
}
\]

The candidate may depend arbitrarily on **all previous** trusted batches.

It simply may not depend on the current certification batch before that batch is used to certify it.

---

# 6. Theorem XXII.3 — Conditional composed-block verification

Define

\[
V_s
=
\mathcal V_{\mathcal F_s}
(p_s,q_s;\mu_s).
\]

Suppose

\[
|L_{p_s,q_s}(f)|
\ge
\Gamma_s.
\]

Then there is a block-end sign test with conditional error probability at most \(\delta_s\) using

\[
\boxed{
m_s
=
O\!\left(
\frac{V_s}{\Gamma_s^2}
\log\frac1{\delta_s}
\right).
}
\]

### Proof

Condition on \(\mathcal G_s\).

After conditioning,

\[
p_s,q_s,\mu_s,\mathcal F_s
\]

are fixed and the new batch is iid from \(\mu_s\).

Therefore the setting is exactly the one-step structured verification theorem.

Let \(g_s\) be the conditional Riesz representer and define

\[
X_{s,i}
=
g_s(Y_{s,i})Z_{s,i}.
\]

Then

\[
\mathbb E[X_{s,i}\mid\mathcal G_s]
=
L_{p_s,q_s}(f),
\]

and

\[
\mathbb E[X_{s,i}^2\mid\mathcal G_s]
\le
V_s.
\]

Conditional median-of-means concentration gives

\[
|\widehat L_s-L_{p_s,q_s}(f)|
\le
C
\sqrt{
\frac{
V_s\log(1/\delta_s)
}{
m_s
}
}
\]

with conditional probability at least \(1-\delta_s\).

Choosing \(m_s\) so that the right-hand side is \(<\Gamma_s\) yields the correct sign. \(\square\)

---

# 7. Corollary XXII.4 — Familywise validity over adaptive refresh blocks

Let blocks \(s=1,2,\ldots\) be chosen adaptively.

Suppose each \(\delta_s\) is predictable before batch \(S_s\) is observed and

\[
\boxed{
\sum_s\delta_s
\le
\delta
\quad\text{almost surely}.
}
\]

Apply Theorem XXII.3 in every attempted block.

Then

\[
\boxed{
\Pr(
\text{any block-end verification error}
)
\le
\delta.
}
\]

### Proof

Let \(E_s\) be the event that block \(s\) is misclassified.

By the conditional theorem,

\[
\Pr(E_s\mid\mathcal G_s)\le\delta_s.
\]

Therefore

\[
\Pr\left(\bigcup_sE_s\right)
\le
\sum_s\Pr(E_s)
=
\sum_s
\mathbb E[
\Pr(E_s\mid\mathcal G_s)
]
\le
\mathbb E\sum_s\delta_s
\le
\delta.
\]

\(\square\)

### Known finite number of blocks

If at most \(J\) blocks are attempted, the simplest choice is

\[
\delta_s=\delta/J.
\]

Then

\[
\boxed{
m_s
=
O\!\left(
\frac{V_s}{\Gamma_s^2}
\log\frac{J}{\delta}
\right).
}
\]

Hence total fresh labels obey

\[
\boxed{
B
=
O\!\left(
\log\frac{J}{\delta}
\sum_{s=1}^J
\frac{V_s}{\Gamma_s^2}
\right).
}
\]

This is a valid online familywise statement.

### Weighted confidence allocation

If deterministic difficulty envelopes

\[
\bar h_s
\ge
V_s/\Gamma_s^2
\]

are known in advance, one may use

\[
\delta_s
=
\delta
\frac{\bar h_s}{\sum_j\bar h_j}
\]

to recover the weighted logarithmic form

\[
\sum_s
\bar h_s
\log
\frac{
\sum_j\bar h_j
}{
\delta\bar h_s
}.
\]

If future \(h_s\) are unknown, this oracle allocation should **not** be presented as directly implementable; equal or summable predictable \(\delta_s\) schedules remain valid.

---

# 8. Corollary XXII.5 — Forward reuse of one trusted stream

After block \(s\) has been certified using \(S_s\), the complete batch \(S_s\) may be fed into:

- verifier training;
- policy optimization;
- feature construction;
- candidate selection;
- future audit design.

That is, \(S_s\) may become part of the next transcript

\[
\mathcal G_{s+1}.
\]

This does not invalidate the block-\(s\) guarantee because \(q_s\) was fixed before \(S_s\) was observed.

Therefore:

\[
\boxed{
\text{one stream of trusted batches can be fresh for the current candidate
and reusable for all future candidates.}
}
\]

This formalizes the temporal idea behind the post-v15 single-stream interpretation.

It also explains why the earlier “sealed audit and fresh audit are two structurally necessary resources” conjecture was false.

The division is temporal, not necessarily physical:

\[
\boxed{
\text{fresh now}
\;\longrightarrow\;
\text{reusable later}.
}
\]

---

# 9. Conditional lower bound: when genuinely fresh information is necessary

A generic lower bound cannot simply sum one-step costs across blocks.

Why?

Because old trusted data may already determine the reward direction relevant to the new block.

A blockwise lower bound requires **residual statistical novelty**.

## Definition — conditionally fresh verification direction

At block \(s\), call a reward direction conditionally fresh if, given the old transcript \(\mathcal G_s\), there remain two reward worlds

\[
f_+,\ f_-\in\mathcal F_s
\]

such that:

1. the law of the old transcript is identical under the two worlds;
2. their correct block decisions are opposite:
   \[
   L_{p_s,q_s}(f_\pm)=\pm\Gamma_s.
   \]

This says the past contains no information that resolves the current sign.

## Theorem XXII.6 — Conditional fresh-information lower bound

Let \(g_s\) be the Riesz representer and

\[
V_s=\|g_s\|_{L_2(\mu_s)}^2.
\]

Assume \(g_s\) is essentially bounded and the local-margin condition

\[
\Gamma_s
\le
\frac{
V_s
}{
2\|g_s\|_\infty
}
\]

holds.

If the least-favorable pair

\[
f_\pm
=
\pm
\frac{\Gamma_s}{V_s}g_s
\]

is a conditionally fresh direction in the above sense, then any block-end test with worst-case conditional error at most \(\delta_s<1/4\) requires

\[
\boxed{
m_s
=
\Omega\!\left(
\frac{
V_s
}{
\Gamma_s^2
}
\log\frac1{\delta_s}
\right).
}
\]

### Proof

Condition on the common old transcript.

By assumption, all pre-existing information has the same law in the two reward worlds.

Only the current trusted batch can distinguish them.

The conditional one-query KL is the same as in the one-step lower bound:

\[
D_{\rm KL}(P_+\|P_-)
\le
4\frac{\Gamma_s^2}{V_s}.
\]

After \(m_s\) fresh observations,

\[
D_{\rm KL}
\le
4m_s\frac{\Gamma_s^2}{V_s}.
\]

Bretagnolle–Huber then gives the same lower bound. \(\square\)

### Interpretation

This is the correct place where “freshness” enters.

\[
\boxed{
\text{fresh labels are necessary when the composed endpoint probes a reward
direction that remains unresolved after conditioning on all old trusted information.}
}
\]

If old data already identify that direction, this lower bound does not apply.

That is exactly consistent with the shared-structure experiments where current-policy refresh self-corrects.

---

# 10. No generic additive lower bound without innovation

The upper bound can always spend a new batch in every block.

But a lower bound of the form

\[
\sum_s
\frac{V_s}{\Gamma_s^2}
\]

is **not universal** if reward structure is shared.

A single old observation may resolve several future blocks.

Additivity requires an independent/orthogonal innovation construction, such as the existing adaptive moving-tail family.

Thus:

\[
\boxed{
\text{blockwise hardness is generic;
blockwise additive hardness requires blockwise statistical innovation.}
}
\]

This distinction prevents the new composed theorem from reintroducing the old error “fresh verification every round.”

---

# 11. Full unstructured class: exact divergence formulas

Take the unrestricted class

\[
\mathcal F=L_2(\mu).
\]

When \(p,q\ll\mu\),

\[
g(y)
=
\frac{dq}{d\mu}(y)
-
\frac{dp}{d\mu}(y).
\]

Hence

\[
\boxed{
\mathcal V_{\rm full}(p,q;\mu)
=
\int
\frac{(dq-dp)^2}{d\mu}.
}
\]

Three audit choices are especially informative.

## Current-policy audit: \(\mu=p\)

\[
\boxed{
V_{\rm current}
=
\chi^2(q\|p).
}
\]

This is the original distribution-shift verification tax.

## Candidate audit: \(\mu=q\)

\[
\boxed{
V_{\rm candidate}
=
\chi^2(p\|q).
}
\]

This can be extremely large if the candidate suppresses regions important under \(p\).

## Balanced audit: \(\mu=(p+q)/2\)

\[
V_{\rm bal}
=
2
\int
\frac{(q-p)^2}{p+q}.
\]

Since

\[
(q-p)^2
\le
(p+q)^2,
\]

\[
\boxed{
V_{\rm bal}
\le4.
}
\]

Therefore, for bounded rewards and a fixed selected block endpoint,

\[
\boxed{
m_{\rm bal}
=
O\!\left(
\frac1{\Gamma^2}
\log\frac1\delta
\right)
}
\]

independently of how large the raw density ratio \(q/p\) becomes.

This is the clean mathematical reason a balanced \(p/q\) audit defeated the earlier two-audit necessity conjecture.

It is not a novel statistical fact; it is essentially the bounded-variance two-sample mean-difference principle expressed in the structured-verification notation.

---

# 12. Exponential stale block under current-policy auditing

For a frozen verifier \(v\), define

\[
q_\Lambda(dy)
=
\frac{
e^{\Lambda v(y)}p(dy)
}{
Z(\Lambda)
},
\qquad
Z(\Lambda)
=
\mathbb E_p[e^{\Lambda v}].
\]

Let

\[
\psi(\Lambda)
=
\log Z(\Lambda).
\]

For current-policy audit \(\mu=p\),

\[
V(\Lambda)
=
\chi^2(q_\Lambda\|p).
\]

Exactly,

\[
\boxed{
1+V(\Lambda)
=
\frac{
Z(2\Lambda)
}{
Z(\Lambda)^2
}
=
\exp[
\psi(2\Lambda)-2\psi(\Lambda)
].
}
\]

## Monotonicity

For \(\Lambda\ge0\),

\[
\frac{d}{d\Lambda}
\log(1+V(\Lambda))
=
2[
\psi'(2\Lambda)-\psi'(\Lambda)
]
\ge0,
\]

because \(\psi\) is convex.

Therefore

\[
\boxed{
\chi^2(q_\Lambda\|p)
\text{ is nondecreasing in cumulative stale strength }\Lambda.
}
\]

This formalizes the intuition that a stale current-policy audit sees progressively worse coverage as the optimizer repeatedly tilts away from it.

---

# 13. But verification hardness is \(V/\Gamma^2\), not \(V\)

Let the actual true block gain be

\[
\Delta_r(\Lambda)
=
\mathbb E_{q_\Lambda}[r]
-
\mathbb E_p[r].
\]

For sign verification at the realized margin, the relevant normalized hardness is

\[
\boxed{
h(\Lambda)
=
\frac{
V(\Lambda)
}{
\Delta_r(\Lambda)^2
}.
}
\]

This corrects another tempting but false simplification:

> larger policy shift does not necessarily mean larger sample complexity.

The margin changes too.

---

# 14. Local asymptotic law near the refresh point

Assume the relevant moments exist and

\[
\operatorname{Cov}_p(r,v)\ne0.
\]

For small \(\Lambda\),

\[
\psi(\Lambda)
=
\Lambda\mathbb E_pv
+
\frac{\Lambda^2}{2}
\operatorname{Var}_p(v)
+
O(\Lambda^3).
\]

Therefore

\[
\boxed{
V(\Lambda)
=
\Lambda^2
\operatorname{Var}_p(v)
+
O(\Lambda^3).
}
\]

Also,

\[
\frac{d}{d\Lambda}
\mathbb E_{q_\Lambda}[r]
\Big|_{\Lambda=0}
=
\operatorname{Cov}_p(r,v),
\]

so

\[
\boxed{
\Delta_r(\Lambda)
=
\Lambda
\operatorname{Cov}_p(r,v)
+
O(\Lambda^2).
}
\]

Hence

\[
\boxed{
\lim_{\Lambda\downarrow0}
h(\Lambda)
=
\frac{
\operatorname{Var}_p(v)
}{
\operatorname{Cov}_p(r,v)^2
}.
}
\]

### Interpretation

Near a freshly fitted policy, shift and useful signal both grow linearly in \(\Lambda\).

The shift variance \(V\) grows quadratically.

The squared true gain also grows quadratically.

Their ratio therefore approaches a finite constant.

So:

\[
\boxed{
\text{small stale movement is not automatically harder to verify merely because
more optimization steps have accumulated.}
}
\]

---

# 15. Quadratic blow-up at a true-gain zero crossing

Suppose there is a finite

\[
\Lambda_\star>0
\]

such that

\[
\Delta_r(\Lambda_\star)=0,
\]

with

\[
V(\Lambda_\star)>0
\]

and a transversal crossing

\[
\Delta_r'(\Lambda_\star)\ne0.
\]

Then

\[
\Delta_r(\Lambda)
=
\Delta_r'(\Lambda_\star)
(\Lambda-\Lambda_\star)
+
o(|\Lambda-\Lambda_\star|).
\]

Therefore

\[
\boxed{
h(\Lambda)
\sim
\frac{
V(\Lambda_\star)
}{
[\Delta_r'(\Lambda_\star)]^2
(\Lambda-\Lambda_\star)^2
}.
}
\]

Thus direct sign verification becomes hardest **near the point where the true gain is nearly zero**.

This is independent of the self-certificate pathology.

It is ordinary statistical indistinguishability of a near-zero effect.

Past the crossing, if \(|\Delta_r|\) grows again, the fresh-audit cost can fall.

This gives a theorem-level explanation for the earlier empirical observation:

> fresh candidate auditing is extremely expensive near the harmful/beneficial boundary but becomes easier again farther into the harmful regime.

---

# 16. Numerical sanity check on the learned code-generator environment

The learned autoregressive DSL generator from Updates XVII–XXI gives:

\[
\Lambda_\star
\approx
22.5409.
\]

The local current-audit hardness limit predicted by

\[
\frac{
\operatorname{Var}_p(v)
}{
\operatorname{Cov}_p(r,v)^2
}
\]

is

\[
\boxed{
55.661.
}
\]

At \(\Lambda=1\), the measured

\[
V_{\rm current}/\Delta^2
\approx59.6,
\]

already close to the asymptotic value.

Near the true-gain crossing, hardness blows up.

| lambda_stale | true_gain | V_current | V_balanced | V_candidate | h_current | h_balanced | h_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.0172 | 0.0176 | 0.0177 | 0.0181 | 59.6257 | 59.7832 | 61.3177 |
| 5.0000 | 0.0669 | 0.4437 | 0.3772 | 0.5551 | 99.1337 | 84.2803 | 124.0341 |
| 10.0000 | 0.0824 | 2.0586 | 1.1129 | 4.3797 | 303.0134 | 163.8050 | 644.6611 |
| 15.0000 | 0.0593 | 5.3790 | 1.8129 | 33.0290 | 1528.0322 | 514.9907 | 9382.6247 |
| 20.0000 | 0.0203 | 10.1001 | 2.3655 | 321.3774 | 24628.3400 | 5768.0470 | 783658.3141 |
| 22.0000 | 0.0042 | 12.1353 | 2.5431 | 856.1309 | 682604.2766 | 143046.8242 | 48157059.0123 |
| 22.5000 | 0.0003 | 12.6421 | 2.5839 | 1099.5786 | 126275694.1334 | 25809127.9181 | 10983190427.1120 |
| 23.0000 | -0.0035 | 13.1459 | 2.6233 | 1415.0285 | 1057763.2557 | 211078.7629 | 113857571.9675 |
| 25.0000 | -0.0182 | 15.1150 | 2.7679 | 3951.7533 | 45600.9549 | 8350.7144 | 11922182.7708 |
| 30.0000 | -0.0492 | 19.5008 | 3.0507 | 57126.5184 | 8056.3539 | 1260.3173 | 23600628.0514 |
| 40.0000 | -0.0882 | 25.4400 | 3.3850 | 15889978.6211 | 3270.2237 | 435.1350 | 2042604976.3897 |

The balanced audit keeps its raw coverage term bounded; in this entire numerical sweep,

\[
V_{\rm balanced}<3.49<4.
\]

Nevertheless

\[
V_{\rm balanced}/\Delta^2
\]

still diverges at \(\Lambda_\star\), because the **margin** vanishes.

![True gain over stale strength](composed_block_true_gain.png)

![Verification hardness](composed_block_verification_hardness.png)

This is an important conceptual separation:

\[
\boxed{
\text{current audit suffers both shift and small-margin cost;}
}
\]

\[
\boxed{
\text{balanced fresh audit removes the shift blow-up but cannot remove
the fundamental }1/\Delta^2\text{ sign-testing cost.}
}
\]

---

# 17. Block batching can reduce verification frequency

The block-compression theorem suggests a practical architecture:

1. freeze a verifier;
2. perform several internal optimization steps;
3. do **not deploy them irreversibly**;
4. audit the block endpoint against the block start;
5. accept the entire block or roll back;
6. feed the fresh trusted batch into the next verifier refresh.

This allows one trusted audit to certify several internal optimization steps.

In the learned-generator population environment with per-step strength

\[
\eta=3,
\]

the first-block tradeoff is:

| stale_steps_L | cumulative_strength_etaL | block_true_gain | block_end_reward | minimum_reward_inside_block | Hoeffding_m_per_distribution_for_95pct_positive_certificate | total_fresh_labels_for_block |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 3.0000 | 0.0462 | 0.6173 | 0.5712 | 4112.0000 | 8224.0000 |
| 2.0000 | 6.0000 | 0.0740 | 0.6452 | 0.5712 | 1600.0000 | 3200.0000 |
| 3.0000 | 9.0000 | 0.0831 | 0.6543 | 0.5712 | 1270.0000 | 2540.0000 |
| 4.0000 | 12.0000 | 0.0766 | 0.6477 | 0.5712 | 1496.0000 | 2992.0000 |
| 5.0000 | 15.0000 | 0.0593 | 0.6305 | 0.5712 | 2490.0000 | 4980.0000 |
| 6.0000 | 18.0000 | 0.0365 | 0.6077 | 0.5712 | 6577.0000 | 13154.0000 |
| 8.0000 | 24.0000 | -0.0110 | 0.5602 | 0.5602 | — | — |

The largest first-block gain occurs at

\[
L=3,
\]

where

\[
\Delta_{\rm block}
\approx0.0831.
\]

A simple Hoeffding two-sample certificate at 95% confidence would require about

\[
2540
\]

fresh semantic labels total for that block, versus about

\[
8224
\]

for the smaller \(L=1\) gain under the same conservative bound.

But \(L=8\) already has negative net block gain.

Thus there is an operational tradeoff:

\[
\boxed{
\text{too-frequent verification gives small hard-to-certify gains;}
}
\]

\[
\boxed{
\text{too-infrequent verification allows stale overoptimization;}
}
\]

with an intermediate block length potentially giving the best progress per trusted label.

### Caution

This is an illustrative oracle/population calculation, not yet an optimized algorithm.

The verifier fit cost, unknown margin, adaptive block-length choice, and familywise confidence accounting all matter in a deployable procedure.

---

# 18. What this theorem does and does not establish

## Established

1. **Exact stale-block composition** for frozen exponential verifiers.
2. **Exact reduction of block-end verification to one-step endpoint verification.**
3. A conditional blockwise minimax upper bound
   \[
   O(V_s\Gamma_s^{-2}\log(1/\delta_s)).
   \]
4. A matching conditional lower bound when the block probes a reward direction unresolved by the old transcript.
5. **Familywise correctness over adaptively generated blocks** under predictable summable error allocation.
6. **Forward reuse:** the same trusted batch can certify the current candidate and train all future verifiers.
7. In the full class:
   - current audit \(V=\chi^2(q\|p)\);
   - candidate audit \(V=\chi^2(p\|q)\);
   - balanced audit \(V\le4\).
8. Under exponential stale movement:
   \[
   1+\chi^2(q_\Lambda\|p)
   =
   e^{\psi(2\Lambda)-2\psi(\Lambda)}.
   \]
9. Local hardness is finite near \(\Lambda=0\) when covariance is nonzero.
10. Hardness diverges quadratically near a transversal true-gain zero crossing.

## Not established

1. A universal additive lower bound across arbitrary recursive blocks.
2. A universal critical \(\eta L\).
3. A universal KL or density-ratio threshold for harm.
4. A theorem saying every stale verifier must fail.
5. A theorem saying fresh candidate labels are necessary when old data already identify the relevant reward direction.
6. Strict safety of every intermediate policy inside a rollback block.
7. Optimal adaptive choice of block length.

---

# 19. Relationship to the adaptive moving-tail theorem

The new theorem and the old moving-tail lower bound are complementary.

## Shared-structure world

The conditional residual reward class shrinks as trusted data accumulate.

Eventually the current block direction may be determined by old data.

Then the lower-bound freshness assumption fails, correctly allowing reuse.

## Persistent moving-tail world

Every block reaches a new reward/error coordinate whose sign remains conditionally unresolved.

Then the conditional lower bound activates repeatedly.

In the orthogonal adaptive-tree family, these costs accumulate, recovering the earlier linear-in-innovation lower bounds.

Therefore the common principle is:

\[
\boxed{
\text{fresh verification is required exactly when the composed stale candidate
creates a verification functional that remains statistically unresolved by the old transcript.}
}
\]

This is more precise than either:

- “fresh labels are needed every round,” or
- “fresh labels are never needed if the audit is sealed.”

---

# 20. Relationship to Theorem U

The composed-block theorem concerns **external trusted verification of the endpoint**.

Theorem U concerns **same-model self-certification**.

They are different.

For the canonical linear plug-in architecture,

\[
\widehat\Delta
=
\eta\|\widehat\theta\|_\Sigma^2
\ge0,
\]

so the self-certificate cannot reject.

The blockwise external audit instead estimates

\[
\Delta
=
\mathbb E_qr-\mathbb E_pr
\]

from trusted labels.

Thus the two results fit together:

\[
\boxed{
\text{optimization can make internal evidence structurally optimistic,
while fresh endpoint evidence remains statistically valid.}
}
\]

The fresh evidence may be expensive near \(\Delta=0\), but that is an ordinary small-margin problem, not self-evaluation blindness.

---

# 21. Novelty assessment

The result is useful but should be described accurately.

The core proof consists of:

- an elementary exponential-tilt composition identity;
- the previously proved one-step restricted-\(\chi^2\) / Riesz verification theorem;
- conditional probability / union-bound accounting.

Therefore:

\[
\boxed{
\text{XXII.3 by itself is not a Spotlight-level new theorem.}
}
\]

Its value is as a **bridge theorem**.

It makes the paper story coherent:

\[
\text{one-step structured coverage}
\longrightarrow
\text{composed stale candidate}
\longrightarrow
\text{recursive refresh blocks}.
\]

The genuinely new/high-upside theorem would still need to characterize the **optimal adaptive refresh frontier** in an endogenous family, rather than merely apply a one-step theorem after a block has been chosen.

---

# 22. Next theoretical target after this result

The right next target is now very narrow.

Define the residual uncertainty after old transcript \(\mathcal H_s\), and a block hardness

\[
h_s(L)
=
\frac{
\mathcal V_{\mathcal F_s^{\rm residual}}
(p_s,q_{s:L};\mu_s)
}{
\Gamma_s(L)^2
}.
\]

Then ask:

> **Given a total trusted-label budget, when should the verifier be refreshed so that useful progress per label is maximized while familywise block safety is maintained?**

A satisfactory result would derive an adaptive stopping / refresh rule from \(h_s(L)\) or a valid upper confidence envelope for it.

This would be more than the current bridge theorem because it would optimize the endogenous choice of \(L\).

However, do not pursue this immediately unless the rule can be made observable without knowing the true margin \(\Gamma_s(L)\).

If it requires oracle access to the true reward gain, the direction should be stopped.

---

# 23. Research decision

This update successfully closes the specific task posed after Update XXI.

The early theory and latest experiments can now be joined without inventing a new metric:

\[
\boxed{
\text{recursive lag}
\;\to\;
\text{composed endpoint }q_{s:L}
\;\to\;
\mathcal V_{\mathcal F}(p_s,q_{s:L};\mu_s)
\;\to\;
\text{trusted-label complexity}.
}
\]

The key conceptual advance is not a new divergence.

It is the identification of the **correct unit of recursive verification**:

\[
\boxed{
\text{a statistically novel composed policy comparison between verifier refreshes.}
}
\]

This should replace “one verification payment per round” in the final paper framing.
