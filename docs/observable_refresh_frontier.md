# Observable refresh frontier: conditional results and an obstruction

Status: elementary propositions proved below; **not** a new minimax-optimal
refresh theorem, and not a safety guarantee for the empirical heuristic.
The focus is: when must verification catch up with a proposed policy movement?

## 1. What is observable

At a refresh, the transcript contains paid rewards, a fitted score, a bank, and
candidate probabilities. Between refreshes the optimizer can construct proposals
and their feature movement without seeing any new trusted outcomes. Offline
outcome tables are inaccessible to the decision rule except through paid queries.

The implementation's `restricted_geometry` is
`d^T (E_p[phi phi^T])^dagger d`, where `d=E_q phi-E_p phi` and `p` is the last
refresh policy. This uses the **known finite-bank audit distribution**, not an
estimated empirical covariance. It is observable but does not certify that the
true residual lies in the feature class. It is a heuristic covariate. The `all`
representation uses only syntax/public features, never hidden semantic signatures.

## 2. Conditional ellipsoid plus misspecification frontier

Let all policies be distributions on a finite bank, reward be stationary, and
`r(i)=phi(i)^T theta_* + b(i)`. Assume an event E on which, simultaneously at every
refresh s, `||theta_*-theta_hat_s||_{A_s} <= beta_s` and `||b||_infinity <= epsilon_s`,
with `A_s` positive definite. An assumption `P(E)>=1-delta` must come from a valid
confidence construction; fitting ridge and using its training residual does not
establish it. Features, fitted model, and envelopes must all satisfy that event,
including any adaptive choice of representation.

For any proposals p,q (even adaptively chosen using the same transcript), define

`G_s=(E_q phi-E_p phi)^T theta_hat_s`

`R_s=beta_s sqrt(d^T A_s^{-1} d) + epsilon_s ||q-p||_1`.

**Proposition.** On E, simultaneously for every proposal,

`E_q r-E_p r >= G_s-R_s`.

Proof: the parameter contribution is bounded by Cauchy-Schwarz in the
`A_s` inner product; the residual contribution is bounded by L1/L-infinity duality.
The event controls all directions, so adaptive proposal selection needs no
per-proposal union bound. If only per-refresh confidence is available, use a
predictable summable allocation `sum_s delta_s <= delta` to establish E first.

A rule may commit only if `G_s-R_s >= g_min`. Otherwise it must acquire information,
shrink/change the proposal, or abstain. For comparisons against the immediately
current deployed policy and `g_min>=0`, all committed rewards are nondecreasing on E.
For comparisons against a block anchor, each committed endpoint is above that
anchor, but adjacent endpoints need not be monotone. An endpoint certificate alone
does not cover untested intermediate deployments.

**Sharpness for the stated envelope.** For a fixed d, choose parameter error
`-beta A^{-1}d / sqrt(d^T A^{-1}d)` and residual
`b(i)=-epsilon sign(q_i-p_i)`. They attain the lower bound in the unconstrained
ellipsoid-times-box uncertainty class (zero directions interpreted by continuity).
If reward is additionally constrained to [0,1], the intersection can tighten the
bound; do not claim the same sharpness there.

This is a robust certification frontier, not a theorem that a refresh restores
safety, that labels are globally necessary, or that adaptive refresh beats a fixed
schedule. Progress/completeness is not proved: the rule can abstain indefinitely.
Unknown arbitrary misspecification is exactly the missing assumption, rather than
something eliminated by an observable covariance statistic.

## 3. Exact finite-bank frontier without realizability

For deterministic exhaustive scores, observing a candidate supplies its exact
reward. Let O be observed indices and U the rest. Rewards lie in [0,1]. For
`d=q-p`, the set of possible true gains consistent with the transcript is exactly

`[sum_O d_i r_i + sum_U min(d_i,0),
  sum_O d_i r_i + sum_U max(d_i,0)]`.

Proof: each unobserved reward is independently in [0,1]; a linear objective is
minimized/maximized at the corresponding box corner. Those corners are admissible,
so both bounds are attained. Repeated indices provide no further information in
this deterministic setting. The unresolved interval width is `sum_U |d_i|`.

**Observable relevance of a query.** Resolving index i shrinks this interval width
by exactly `|q_i-p_i|`, independent of its label. For a *fixed* comparison, equal-cost
queries sorted by this weight minimize remaining width at every query count.
This is a width objective, not expected stopping cost, a global recursive optimum,
or a claim for shared semantic structure or stochastic labels. See
`src/observable_certificate.py` and exhaustive corner-enumeration tests.

## 4. Indistinguishability counterexample

Take two unobserved candidates, p=(1/2,1/2), q=(0,1), and identical observable
features/scores in two worlds. Reward worlds r+=(0,1) and r-=(1,0) are consistent
with the same empty trusted transcript, while gains are +1/2 and -1/2. KL, max log
ratio, elapsed rounds, and any transcript-only geometry are identical. No such
observable can determine the sign in both worlds. Information or a valid
structural restriction is required. Abstention is possible, so this is not a
safety-only label lower bound.

## 5. Relation to prior work and next decision

The ellipsoid argument and robust box optimization are standard. Restricted
function-class divergence is already present in Duan, Jia and Wang (ICML 2020,
https://proceedings.mlr.press/v119/duan20b.html). Reward overoptimization across
optimization procedures is already established by Gao, Schulman and Hilton
(https://arxiv.org/abs/2210.10760). RVL's prospective contribution must be the
empirically transferable relation between endogenous policy movement, unresolved
error, and the cost/progress frontier of refreshing; these propositions alone do
not establish that contribution.

Next theoretical step, conditional on empirical support: study a structured
endogenous family with a learnable residual envelope and a progress/completeness
objective, comparing adaptive refresh to the best fixed cadence under an equal
label budget. Preserve counterexamples where a shift-only trigger fails.

## 6. Frozen Best-of-N is also composable (implementation identity)

For fixed score v and IID draws, let F_p(z)=P_p(v<=z). The random-tie Best-of-N
winner law has score CDF F_p(z)^N. Repeating with N_1,...,N_L gives
F_p(z)^(product_k N_k), hence the endpoint law equals Best-of-(product_k N_k),
including preserved conditional proportions within tied score groups. This is an
order-statistics identity, not a novelty claim. It explains why a frozen-bank
recursive selection experiment is not itself evidence of learned capability
creation, and why refresh-induced changes in the score ordering are essential.
