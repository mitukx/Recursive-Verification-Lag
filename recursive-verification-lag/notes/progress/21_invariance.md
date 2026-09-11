# Progress Update XXI — From stale-step count to coverage-relative shift

**Date:** 2026-09-11  
**Status:** Completed. The previous \(\eta L\) phase law survives as an optimizer-specific coordinate, but a stronger invariance test shows that no policy-shift scalar alone is a universal safety law. The best current synthesis is **optimizer-induced shift relative to verifier misspecification geometry**.

---

## 1. Motivation

Progress Update XX found an approximate recursive collapse boundary

\[
\eta L \approx 15\text{--}20
\]

for exponential/KL updates under a stale verifier.

That raised two questions:

1. Is \(\eta L\) a meaningful cross-system quantity, or only a consequence of the arbitrary numerical scale of the verifier score?
2. Can a policy-shift measure such as KL, \(\chi^2\), or maximum density-ratio amplification predict failure across:
   - score rescaling,
   - different optimizer families,
   - different verifier representations?

This update tests those invariances.

---

## 2. Result A — raw \(\eta L\) is not score-scale invariant

Take the same fitted verifier score \(v\), but replace it by

\[
v_c = c\,v.
\]

Under exponential update,

\[
q(y)\propto p(y)\exp(\eta v_c(y))
=
p(y)\exp(c\eta v(y)).
\]

Therefore changing the arbitrary score scale by \(c\) must change the raw \(\eta\)-threshold by \(1/c\).

This was tested in the recursive learned-generator environment.

| score_scale | min_failing_raw_etaL | min_failing_scale_adjusted_etaL |
| --- | --- | --- |
| 0.500 | 32.000 | 16.000 |
| 1.000 | 16.000 | 16.000 |
| 2.000 | 8.000 | 16.000 |
| 4.000 | 4.000 | 16.000 |

The first failing raw \(\eta L\) changes exactly as expected:

\[
32,\ 16,\ 8,\ 4
\]

for score scales

\[
c=0.5,\ 1,\ 2,\ 4.
\]

But the scale-adjusted product

\[
\boxed{c\eta L}
\]

is exactly

\[
\boxed{16}
\]

for all four settings in this grid.

### Consequence

\[
\boxed{
\eta L \text{ is not a universal physical quantity.}
}
\]

It is useful only after fixing verifier score calibration.

The invariant object must be a quantity induced by the **actual policy change**.

![Score-rescaling invariance](recursive_score_rescaling_invariance.png)

---

## 3. Result B — within a fixed verifier, max density-ratio amplification transfers best across optimizer families

The next test keeps the **same proxy-only verifier** but changes the optimizer:

- exponential tilt;
- exact Best-of-\(N\) selection.

For Best-of-\(N\), the selected distribution is computed exactly from the learned generator distribution, including score ties.

For each Best-of-\(N\) candidate distribution, an exponential-tilt candidate with the closest value of a chosen shift metric is found. The resulting true semantic gains are compared.

| matching_metric | mean_abs_true_gain_difference | median_abs_true_gain_difference | true_gain_correlation | sign_agreement |
| --- | --- | --- | --- | --- |
| max_log_ratio | 0.0043 | 0.0032 | 0.9985 | 0.9950 |
| mem_mass_shift | 0.0043 | 0.0032 | 0.9985 | 0.9950 |
| chi2 | 0.0053 | 0.0032 | 0.9990 | 0.9875 |
| proxy_gain | 0.0075 | 0.0033 | 0.9980 | 0.9749 |
| KL | 0.0085 | 0.0033 | 0.9955 | 0.9674 |
| TV | 0.0129 | 0.0034 | 0.9720 | 0.9524 |

The most transportable tested metric is

\[
\boxed{
\max_y \log\frac{q(y)}{p(y)}
=
\log \left\|\frac{dq}{dp}\right\|_\infty.
}
\]

It gives:

\[
\text{true-gain correlation}\approx 0.9985,
\]

\[
\text{mean absolute gain mismatch}\approx0.0043,
\]

\[
\text{sign agreement}\approx99.5\%.
\]

\(\chi^2\) is close. KL is useful but weaker.

### Interpretation

This result connects directly back to the original candidate-amplification parameter

\[
M=\left\|\frac{dq}{dp}\right\|_\infty.
\]

The original moving-tail theory used \(M\) because current-policy verification becomes expensive when the candidate puts much more mass than \(p\) on verifier-salient regions.

The new experiment suggests that this \(L_\infty\)-style amplification is not merely a hard-family artifact: **within a fixed verifier, it is also highly portable across two very different selection mechanisms.**

![Optimizer-family gain versus KL](recursive_optimizer_family_gain_vs_kl.png)

The plot above uses KL for readability; the quantitative matching table shows that max log-density ratio performs better.

---

## 4. Result C — even max density-ratio amplification is not enough across verifier classes

A stronger verifier changes not only the scale of the score but the **direction in program space that is being optimized**.

To isolate this effect, candidate distributions were matched to the same maximum log density ratio under three verifier representations:

- proxy only;
- medium exploit signature;
- strong exploit signature.

| target_max_log_ratio | representation | eta_required | actual_max_log_ratio | true_gain | proxy_gain | KL | chi2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | proxy_only | 4.0853 | 1.0000 | 0.0585 | 0.0686 | 0.1364 | 0.2922 |
| 1.0000 | medium_signature | 3.7382 | 1.0000 | 0.0556 | 0.0632 | 0.1152 | 0.2421 |
| 1.0000 | strong_signature | 3.5780 | 1.0000 | 0.0581 | 0.0616 | 0.1065 | 0.2177 |
| 2.0000 | proxy_only | 9.9490 | 2.0000 | 0.0825 | 0.1460 | 0.6675 | 2.0340 |
| 2.0000 | medium_signature | 8.7033 | 2.0000 | 0.0869 | 0.1305 | 0.5257 | 1.4313 |
| 2.0000 | strong_signature | 8.1627 | 2.0000 | 0.1041 | 0.1213 | 0.4484 | 1.0947 |
| 3.0000 | proxy_only | 22.1217 | 3.0000 | 0.0033 | 0.2356 | 2.0184 | 12.2589 |
| 3.0000 | medium_signature | 16.2642 | 3.0000 | 0.0642 | 0.1986 | 1.3516 | 5.9198 |
| 3.0000 | strong_signature | 14.3445 | 3.0000 | 0.1317 | 0.1736 | 1.0239 | 3.6035 |
| 4.0000 | medium_signature | 31.0520 | 4.0000 | -0.0258 | 0.2542 | 2.5803 | 19.4228 |
| 4.0000 | strong_signature | 23.1146 | 4.0000 | 0.1482 | 0.2170 | 1.8185 | 11.7376 |
| 5.0000 | medium_signature | 62.7767 | 5.0000 | -0.0879 | 0.2752 | 3.4504 | 39.3009 |
| 5.0000 | strong_signature | 36.6750 | 5.0000 | 0.1779 | 0.2546 | 2.9203 | 47.9186 |

At matched shift magnitude, the semantic outcome can differ substantially.

For example, around max log density ratio \(=4\):

- medium representation: true gain \(\approx -0.026\);
- strong representation: true gain \(\approx +0.148\).

Thus essentially the same worst-case policy amplification can be harmful under one verifier and strongly beneficial under another.

A matched-KL control shows the same point even more starkly.

| target_KL | representation | eta_required | true_gain | proxy_gain | cauchy_risk_ratio | actual_KL | chi2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | proxy_only | 12.8473 | 0.0725 | 0.1753 | 1.1873 | 1.0000 | 3.7323 |
| 1.0000 | medium_signature | 13.0893 | 0.0806 | 0.1745 | 1.1703 | 1.0000 | 3.6226 |
| 1.0000 | strong_signature | 14.0907 | 0.1310 | 0.1719 | 1.1196 | 1.0000 | 3.4616 |
| 2.0000 | proxy_only | 21.9257 | 0.0048 | 0.2347 | 1.5938 | 2.0000 | 12.0597 |
| 2.0000 | medium_signature | 22.8468 | 0.0207 | 0.2322 | 1.5919 | 2.0000 | 11.8736 |
| 2.0000 | strong_signature | 25.2399 | 0.1519 | 0.2245 | 1.7867 | 2.0000 | 15.0315 |
| 2.5000 | proxy_only | 28.1615 | -0.0388 | 0.2550 | 1.7925 | 2.5000 | 17.9940 |
| 2.5000 | medium_signature | 29.6758 | -0.0192 | 0.2515 | 1.8211 | 2.5000 | 18.2292 |
| 2.5000 | strong_signature | 31.3247 | 0.1644 | 0.2423 | 2.2939 | 2.5000 | 28.8534 |
| 3.0000 | proxy_only | 38.7456 | -0.0847 | 0.2703 | 1.9881 | 3.0000 | 24.8839 |
| 3.0000 | medium_signature | 40.7680 | -0.0597 | 0.2661 | 2.0858 | 3.0000 | 26.7588 |
| 3.0000 | strong_signature | 37.7138 | 0.1808 | 0.2568 | 2.9189 | 3.0000 | 52.4918 |

At KL \(\approx2.5\):

- proxy-only: true gain \(\approx-0.039\);
- medium representation: true gain \(\approx-0.019\);
- strong representation: true gain \(\approx+0.164\).

### Consequence

\[
\boxed{
\text{No policy-shift magnitude alone determines recursive safety.}
}
\]

A shift metric must be interpreted **relative to the verifier's error / misspecification geometry**.

![Matched max-log-ratio representation control](recursive_matched_maxlog_representation.png)

![Matched KL representation control](recursive_matched_kl_representation.png)

---

## 5. Result D — the generic Cauchy coverage certificate is valid but too conservative to explain the empirical phase

The exact decomposition is

\[
\Delta
=
G
-
\left(
\mathbb E_q e-\mathbb E_p e
\right),
\qquad
e=v-r.
\]

If the fitted verifier contains an intercept, then for its population least-squares projection under \(p\),

\[
\mathbb E_p[e]=0.
\]

Hence

\[
\mathbb E_q e
=
\mathbb E_p
\left[
\left(\frac{q}{p}-1\right)e
\right].
\]

By Cauchy-Schwarz,

\[
\left|
\mathbb E_qe-\mathbb E_pe
\right|
\le
\|e\|_{L_2(p)}
\sqrt{\chi^2(q\|p)}.
\]

Therefore every stale block satisfies the valid lower bound

\[
\boxed{
\Delta
\ge
G-
\|e\|_{L_2(p)}
\sqrt{\chi^2(q\|p)}.
}
\]

So a sufficient safety condition is

\[
\boxed{
G>
\|e\|_{L_2(p)}
\sqrt{\chi^2(q\|p)}.
}
\]

This is not a new theorem; it is the earlier coverage bound applied to the composed stale-block candidate.

The empirical test confirms validity:

| representation | optimizer | n_blocks | certified_safe_fraction | endpoint_harmful_fraction | false_safe_fraction | coverage_of_truly_safe_blocks |
| --- | --- | --- | --- | --- | --- | --- |
| medium_signature | bon | 180 | 0.1667 | 0.1167 | 0.0000 | 0.1887 |
| medium_signature | exp | 240 | 0.0833 | 0.0500 | 0.0000 | 0.0877 |
| proxy_only | bon | 180 | 0.0278 | 0.1611 | 0.0000 | 0.0331 |
| proxy_only | exp | 240 | 0.0792 | 0.0958 | 0.0000 | 0.0876 |
| strong_signature | bon | 180 | 0.2167 | 0.0111 | 0.0000 | 0.2191 |
| strong_signature | exp | 240 | 0.1000 | 0.0000 | 0.0000 | 0.1000 |

There were **zero false-safe blocks** in the tested population trajectories.

However, the certificate is highly conservative. Depending on representation and optimizer, it certifies only about 3–22% of truly safe blocks.

### Consequence

The generic \(L_2\times\chi^2\) bound is appropriate as a worst-case safety certificate, but it is **not tight enough to explain the observed phase boundary**.

This is useful: it tells us not to force the empirical \(\eta L\) transition into a loose global Cauchy bound.

---

## 6. The exact latent quantity

For any realized verifier \(v\), define

\[
G(p,q;v)
=
\mathbb E_qv-\mathbb E_pv,
\]

and the verifier-error shift

\[
E_{\rm shift}(p,q;v,r)
=
\mathbb E_q(v-r)-\mathbb E_p(v-r).
\]

Then exactly

\[
\boxed{
\Delta
=
G-E_{\rm shift}.
}
\]

When \(G>0\), define the normalized exploitation ratio

\[
\boxed{
\rho_{\rm exploit}
=
\frac{E_{\rm shift}}{G}.
}
\]

Then

\[
\boxed{
\Delta>0
\iff
\rho_{\rm exploit}<1,
}
\]

\[
\boxed{
\Delta<0
\iff
\rho_{\rm exploit}>1.
}
\]

This identity is tautological, not a contribution by itself.

Its importance is statistical:

> **the verification problem is precisely the problem of estimating the policy-induced shift of verifier error after the optimizer has selected \(q\).**

That is the quantity that old current-policy data may fail to identify when \(q/p\) amplifies poorly covered regions, and the quantity that fresh candidate-aware labels estimate directly.

---

## 7. New synthesis with the earlier structured-verification theorem

The earlier one-step structured theory defined the policy-improvement functional

\[
L_{p,q}(f)
=
\mathbb E_qf-\mathbb E_pf
\]

and its function-class verification difficulty

\[
\mathcal V_{\mathcal F}(p,q;\mu).
\]

Progress Updates XIX–XXI now suggest a recursive interpretation:

1. At a verifier refresh time \(s\), the audit data constrain the residual/error class around \(p_s\).
2. The optimizer then moves under a stale verifier for several updates.
3. These updates compose into a new candidate \(q_{s:L}\).
4. What must be certified is not the number of elapsed rounds, but the functional
   \[
   L_{p_s,q_{s:L}}(e).
   \]
5. Its difficulty is governed by how the **composed stale shift** intersects the error class.

Thus the natural recursive quantity is not merely

\[
L,\qquad \eta L,\qquad \mathrm{KL}(q\|p),\qquad M.
\]

It is the pair

\[
\boxed{
\text{stale policy shift}
\quad+\quad
\text{verifier-error geometry}.
}
\]

In the unstructured worst case this reduces to density-ratio amplification such as \(M\).

In a structured class it reduces to restricted feature geometry such as

\[
d^\top\Sigma_\mu^\dagger d.
\]

This reconnects the newest recursive experiments to the earliest minimax theory.

---

## 8. Updated interpretation of the \(\eta L\) phase

The empirical law from Update XX should now be stated more carefully.

Old wording:

> recursive failure is controlled by \(\eta L\).

Better wording:

> **For exponential updates with a fixed score calibration and fixed misspecified verifier class, \(\eta L\) is an exact parameterization of accumulated stale optimization pressure. Across optimizer families, density-ratio amplification is more portable. Across verifier classes, no shift-only scalar is sufficient.**

So the hierarchy is:

### Level 1 — optimizer-specific coordinate

\[
\eta L.
\]

Exact for repeated exponential tilts with a frozen verifier.

### Level 2 — policy-shift coordinate

\[
\log M
=
\log\left\|\frac{dq}{dp}\right\|_\infty,
\]

or related KL / \(\chi^2\).

More portable across optimizer families.

### Level 3 — verification-relevant coordinate

\[
\mathcal V_{\mathcal F}(p,q;\mu)
\]

or an equivalent residual-sensitive coverage quantity.

Needed when verifier representations differ.

This three-level hierarchy is currently the cleanest synthesis.

---

## 9. Research implication: the central recursive theorem should be a composed-candidate theorem, not an \(\eta L\) theorem

A future theorem should not attempt to prove a universal numerical threshold in \(\eta L\).

A better target is:

> Given a verifier refreshed at time \(s\), characterize the trusted information required to certify every update until the next refresh in terms of the **composed candidate distribution** \(q_{s:L}\) and the audit-visible error class.

For frozen exponential updates,

\[
q_{s:L}(y)
\propto
p_s(y)\exp(L\eta v_s(y)).
\]

Then the existing structured one-step theorem can be applied directly to the composed stale candidate.

A schematic target is

\[
\boxed{
m_s
\asymp
\frac{
\mathcal V_{\mathcal F_s}
(p_s,q_{s:L};\mu_s)
}{
\Gamma_s^2
}
\log\frac1\delta.
}
\]

The actual research challenge is to make this useful recursively:

- \(L\) is chosen adaptively;
- \(p_s\) changes after refresh;
- \(\mathcal F_s\) may itself change after new trusted data;
- familywise guarantees must hold over refresh blocks;
- candidate-aware fresh labels may be fed into later verifier fits.

This target does **not** invent a new metric. It asks whether the existing verification functional can be lifted from one-step candidates to recursively composed stale blocks.

---

## 10. Current empirical verdict

The invariance test yields four clean conclusions.

### Supported

1. Raw \(\eta L\) is score-scale dependent.
2. Max density-ratio amplification is highly transportable between exponential tilt and Best-of-\(N\) under the same verifier.
3. No shift-only metric is sufficient across verifier representations.
4. Generic \(L_2\times\chi^2\) safety bounds are valid but too conservative to explain the observed phase sharply.

### Rejected / downgraded

1. A universal \(\eta L\) law.
2. A universal KL-only recursive safety threshold.
3. The idea that policy movement magnitude by itself determines failure.

### Strengthened

The original broader thesis:

\[
\boxed{
\text{verification difficulty is about optimizer-induced movement into directions
that the current verification state does not control.}
}
\]

This survives all current corrections.

---

## 11. Best current paper-level message

The most defensible conceptual statement is now:

> **Recursive verification lag is not elapsed time. It is the accumulation of optimizer-induced policy shift in directions where the current verifier remains uncertain or misspecified.**

A shorter version:

\[
\boxed{
\text{verification lag}
=
\text{policy shift relative to verifier-error geometry}.
}
\]

This is more general than the empirical \(\eta L\) law and more faithful to the minimax theory.

---

## 12. Next research step

The next theoretical step is now justified, but it should be narrow.

Do **not** create another broad theorem family.

Attempt exactly one result:

### Target — Composed stale-candidate verification law

Take the existing structured one-step verification theorem and prove a blockwise recursive corollary for a verifier frozen over multiple policy updates.

Required output:

1. exact composition for exponential update;
2. blockwise verification complexity in terms of
   \[
   \mathcal V_{\mathcal F}(p_s,q_{s:L};\mu_s);
   \]
3. familywise accounting over adaptively chosen refresh blocks;
4. a lower-bound construction showing that if the composed candidate reaches a new poorly covered error direction, no amount of mere round-count bookkeeping can replace fresh trusted information;
5. explicit counterexample showing why shift magnitude alone is insufficient when verifier representation changes.

If this closes cleanly, it is the right bridge between the early lower-bound theory and the new recursive experiments.

If it does not close without strong artificial assumptions, stop theory expansion and present the \(\eta L\) / max-density-ratio results as controlled empirical mechanism evidence.

---

## 13. Artifact inventory for this update

Data:

- `recursive_score_rescaling_phase.csv`
- `recursive_score_rescaling_boundary.csv`
- `recursive_cross_setting_stale_blocks.csv`
- `recursive_cross_setting_runs.csv`
- `recursive_cross_setting_metric_accuracy.csv`
- `recursive_matched_kl_representation_control.csv`
- `recursive_matched_maxlog_representation_control.csv`
- `recursive_matched_kl_optimizer_family.csv`
- `recursive_optimizer_metric_matching_quality.csv`
- `recursive_cauchy_certificate_tightness.csv`

Figures:

- `recursive_score_rescaling_invariance.png`
- `recursive_optimizer_family_kl_alignment.png`
- `recursive_optimizer_family_gain_vs_kl.png`
- `recursive_matched_kl_representation.png`
- `recursive_matched_maxlog_representation.png`
- `recursive_cauchy_risk_alignment.png`
