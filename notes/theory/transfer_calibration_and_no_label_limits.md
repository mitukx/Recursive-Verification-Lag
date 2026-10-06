# Transfer calibration and the no-evaluation-label safety limit

These are elementary identification results, not novel generalization theorems.
They constrain interpretation of the completed development transfer experiment.

## Positive affine scores describe the same exponential path

On finite support with baseline p, let

q_beta(i) = p(i) exp(beta s(i)) / Z(beta), beta >= 0.

For a>0 and any b, replacing s with a*s+b gives exactly q_(a*beta).
Consequently, at a fixed update strength, a change in fitted score slope can
change trusted reward without changing the set of achievable policies.
An intercept cancels. For s(i)=theta_0+theta_1*x(i), theta_1>0, every fit has
the same nonnegative exponential path as the raw feature x.

Let A(beta)=log Z(beta). Then

KL(q_beta || p) = beta A'(beta)-A(beta),
d/d beta KL(q_beta || p) = beta Var_(q_beta)[s] >= 0.

For nonconstant supported scores this derivative is strictly positive for
beta>0. Therefore a target k in (0, -log p(argmax s)) selects one distribution
on the path. KL matching removes positive affine score scaling exactly.
At k=0, q=p. For constant supported scores only k=0 is attainable. With ties,
the limiting policy is p conditioned on the entire top-score set; there is
no arbitrary tie-breaking or injected support.

Equal KL does **not** imply equal policy, safety, computational cost, or reward
for non-affine scores. Negative slopes reverse the path. Nonlinear monotone
transforms preserve ranking but generally change the exponential path.

## Ranking controls identify a different object

Best-of-N depends only on the score ordering and tie partition. Any strictly
increasing transform preserves its winner distribution under the same baseline
and tie-preserving selection rule. Thus an unchanged best-of-N policy does not
prove affine equivalence, nor exclude useful within-path spacing changes.

In the completed 16-task diagnostic, both the public and all-feature fits
preserve every exact pairwise order/tie relation across refresh: 0 changes
among 1,920 within-task occurrence pairs per representation. Duplicate sample
occurrences remain included. The 1e-10 absolute tie sensitivity check also has
zero changes. These are finite-bank facts, not statements about unseen texts.

## Training labels alone do not certify an arbitrary evaluation task

Condition on any paid training transcript, cheap evaluation features and chosen
p,q. Suppose there is no valid relation between training and evaluation rewards,
and no paid evaluation label. Every evaluation reward assignment y in [0,1]^m
remains compatible with the observations. Then

min_y (q-p)^T y = sum_(i:q_i<p_i) (q_i-p_i) = -TV(p,q).

Proof: assign y_i=1 wherever q_i-p_i<0, and y_i=0 wherever it is positive.
Because both distributions sum to one, negative and positive total mass are
equal in magnitude. The displayed assignment also uses only binary rewards.
It can coexist with an identical training transcript in a second world with
y_i=1 on the positive differences. Any nonidentity update is harmful in the
first world and beneficial in the second. Its trusted sign is unidentifiable.

If semantic duplicates impose equal rewards within a source group g, first
aggregate d_g=sum_(i in g)(q_i-p_i). The sharp bound is
sum_g min(d_g,0)=-TV(P_source,Q_source). An update that only redistributes
within an exact-source group can be neutral. Neither raw occurrence bound nor
source-group bound should be presented as empirical generalization accuracy.

Paid evaluation labels, valid structural assumptions linking rewards to
features, or an independent exchangeable-task evaluation design can change the
information available. Empirical average transfer and per-task distribution-free
certification are distinct claims. A learned verifier with an observed good
average is not, by itself, such an assumption.

`src/transfer_calibration.py` reports the unrestricted occurrence-box bound as
a deliberately conservative diagnostic. No evaluation label affects decisions,
so its negative value cannot become a positive safety certificate. This is the
existing source-box geometry specialized to an unpaid new task, rather than a
new lower-bound family.
