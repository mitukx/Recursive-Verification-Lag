# Propensity-weighted verifier alignment auditing

This note extends the i.i.d. covariance gate to randomized nonuniform auditing.
The target is a finite policy support or a fixed rollout batch. It does not by
itself remove uncertainty from how that support was sampled from a larger
population.

Let the target distribution have known masses (p_i>0), cheap proxy scores
(v_i), and expensive trusted rewards (y_iin[0,1]). Define

[
mu_v=sum_i p_i v_i,qquad
C=operatorname{Cov}_p(y,v)
=sum_i p_i y_i(v_i-mu_v).
]

For each support item, choose an audit independently with known inclusion
probability (pi_iin(0,1]):

[
I_isim mathrm{Bernoulli}(pi_i).
]

The Horvitz--Thompson estimator

[
widehat C
=
sum_i rac{I_i p_i y_i(v_i-mu_v)}{pi_i}
]

is conditionally unbiased:

[
mathbb E[widehat Cmid p,v,y]=C.
]

The inclusion probabilities may depend arbitrarily on **free information** such
as (p_i), proxy score, uncertainty, or task metadata, as long as they are fixed
before the corresponding trusted reward is observed and remain strictly
positive.

## Distribution-free confidence radius

Let

[
a_i=|v_i-mu_v|,qquad
w_i=p_i a_i.
]

Because (y_iin[0,1]), the random contribution from item (i) has range width
at most (w_i/pi_i). Hoeffding's inequality for independent, non-identically
bounded terms gives

[
Pr!left(
|widehat C-C|>r(pi,delta)
ight)
le delta,
]

with

[
r(pi,delta)
=
sqrt{
rac{log(2/delta)}{2}
sum_i
left(rac{w_i}{pi_i}ight)^2
}.
]

The same fail-closed decision rule applies:

- upper confidence bound below zero: block the local proxy update or refresh;
- lower confidence bound above zero: allow only the local direction;
- otherwise request more trusted information.

This is a certificate for the covariance of the declared finite target. If the
target itself is an empirical rollout batch, population generalization requires a
separate argument.

## Proxy-only optimal allocation for this bound

Given expected audit budget

[
sum_i pi_i=B,
]

the confidence radius is minimized by minimizing

[
sum_i rac{w_i^2}{pi_i^2}.
]

Ignoring box constraints, the Lagrangian first-order condition is

[
-2w_i^2pi_i^{-3}+lambda=0,
]

hence

[
pi_i propto w_i^{2/3}
=
left(p_i|v_i-mu_v|ight)^{2/3}.
]

With practical bounds (pi_{min}lepi_ile1), the optimum is the clipped
water-filling solution

[
pi_i
=
operatorname{clip}!left(
c,w_i^{2/3},
pi_{min},
1
ight)
]

for the unique scale (c) that satisfies the expected budget.

Crucially, this allocation uses no trusted reward. It spends labels where an
unknown reward can have the largest leverage on the covariance estimate.

## Why deterministic priority is different

Selecting the top-(k) proxy-risk items deterministically generally sets the
inclusion probability of all other items to zero. Then the full-support
covariance is not identified without additional structural assumptions.
Treating such a selected set as i.i.d. produces a false confidence statement.

Known positive propensities solve that specific problem: every target item
remains representable, and inverse-propensity weighting corrects the intentional
selection bias.

This does **not** make every adaptive audit valid. Required conditions include:

1. inclusion randomization with recorded propensities;
2. propensities fixed before the corresponding trusted outcome is observed;
3. positive inclusion probability for every target item relevant to the claim;
4. correct policy/support provenance;
5. an additional correction if (mu_v) itself is estimated rather than known.

## Experiment contract

`configs/propensity_alignment_audit_v1.json` compares uniform Bernoulli auditing
against the proxy-leverage allocation above at equal expected trusted-label
budgets. The follow-up is explicitly informed by the previous covariance-gate
experiment and is therefore an efficiency study, not independent confirmation
of the original mechanism.

The systems track deliberately does not yet enable this estimator inside the
existing deterministic adaptive controller. A runtime integration should first
randomize audit selection and persist the realized propensity of every candidate
that enters the target batch.
