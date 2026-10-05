# From uncertainty width to the cost of a policy decision

Status: exact finite-support propositions and an executable decision-cost
benchmark. These are elementary reductions to active evaluation, not a new
general optimal-refresh theorem. All prior empirical banks have been observed;
the accompanying eight-task analysis is retrospective.

## Fixed comparison and source labels

Freeze a proposed policy q and reference policy b while acquiring labels.
Group exact duplicate source texts within one task. Let w_g=q_g-b_g, with
deterministic reward r_g in [0,1], observed exactly at one paid source call.
There is no reward smoothness or cheap-score calibration assumption.

For the observed source set A, let
L_A=sum_(g in A) w_g r_g + sum_(g not in A) min(0,w_g), and
U_A=sum_(g in A) w_g r_g + sum_(g not in A) max(0,w_g).
The exact interval is already proved in `identified_refresh_frontier.md`.
We now ask whether gain reaches a declared floor gamma: accept when
L_A >= gamma, reject when U_A < gamma, and otherwise acquire or abstain.
A rejection need not mean negative gain: it means gain below the chosen floor.
The experiment fixes gamma=.01 before computing the new comparative results.

## Proposition 1: exact realized certificate-size frontier

For g outside A define two nonnegative corrections

\[
c_g = w_g r_g-\min(0,w_g),\qquad
d_g = \max(0,w_g)-w_g r_g.
\]

For any additional source set S, exactly

\[
L_{A\cup S}=L_A+\sum_{g\in S}c_g,\qquad
U_{A\cup S}=U_A-\sum_{g\in S}d_g.
\]

If realized gain is at least gamma, the smallest *additional* unit-cost
certificate consists of the smallest k such that the sum of the k largest
c_g is at least gamma-L_A. If gain is below gamma, the analogous minimum
uses the largest d_g until their sum is strictly greater than U_A-gamma.
Zero additional labels suffice if the current interval already decides.

Proof: revealing one label replaces precisely its interval endpoint term;
the displayed identities follow by subtraction. Among every set of k sources,
the sum of its nonnegative corrections is no larger than the top-k sum.
The top-k set therefore attains the optimum whenever it crosses the stated
threshold, and no smaller set can cross. Full observation must decide because
the interval collapses to the realized gain. QED.

This is a **clairvoyant lower bound**, computed by the offline evaluator.
The corrections depend on unpaid rewards and must not guide a controller.
It distinguishes two failure modes: too few labels even with hindsight,
versus enough labels in principle but poor acquisition. It does not lower-bound
all possible methods that use justified additional reward structure.

Width reduction from one audit is |w_g|, irrespective of its outcome. In
contrast, the lower-bound increase is c_g, which can be zero. Thus maximal
width reduction need not minimize the cost of accepting a useful proposal.
Even a perfect verifier refit cannot remove this distinction without supplying
new valid information about currently unobserved sources.

## Proposition 2: budgeted decision-aware acquisition

Suppose rewards are binary and their planning prior is independent Bernoulli
with known probabilities pi_g strictly between zero and one. For an observed
assignment z and remaining budget k define (P(z,k),C(z,k)) as maximal prior
probability of resolving gain >= gamma versus gain < gamma before exhaustion,
and, among tied maxima, minimal expected additional paid labels. A resolved
state has (1,0); an unresolved zero-budget state has (0,0). For each available
source g, compute

\[
P_g=(1-\pi_g)P(z\cup\{g=0\},k-1)+\pi_gP(z\cup\{g=1\},k-1),
\]
\[
C_g=1+(1-\pi_g)C(z\cup\{g=0\},k-1)+\pi_gC(z\cup\{g=1\},k-1).
\]

Include abstention (P,C)=(0,0) as an available action in an unresolved state.
Choose maximal P_g, then minimal C_g, then fixed lexical source order. If no
source can give positive resolution probability within the cap, abstain
without spending an additional label.
Backward induction proves exact optimality for this **fixed comparison,
binary independent prior and unit-label-cost objective**. Randomizing cannot
improve a linear expectation over available first actions. A zero-weight
source can be omitted. At most (k+1)3^G cached states are possible; this is an
exponential reference algorithm, capped at ten source groups in the code.

The certificate itself never uses pi_g. A wrong prior can waste labels or
prefer rejections, but cannot create a mathematically invalid accepted gain.
The benchmark uses a fixed pi_g=.5 without claiming empirical calibration.
No expected-query optimality follows under the actual benchmark reward law.
This is stochastic Boolean/linear-threshold evaluation, whose literature
predates this project; see Deshpande, Hellerstein and Kletenik (2014),
https://arxiv.org/abs/1303.0726 . A dynamic program alone is not new theory.

## Recursive validity and its limit

Before a round, q can be any adaptive function of prior paid labels. During
that round freeze q, acquire labels, then gate it using the exact box interval
relative to the initial b. Accept only if L_A >= gamma >= 0; otherwise retain
the previously accepted policy. Refit the verifier between rounds. Every
accepted policy is at least gamma above the initial baseline on the finite
support, for every reward vector compatible with its current transcript.
This follows directly from the exact interval, regardless of adaptive history.
It does **not** imply improvement relative to the previous accepted policy,
optimal audit allocation across rounds, or any generalization to new programs.

If q is recomputed after each label, the displayed fixed-contrast Bellman
optimum no longer applies. The deterministic box gate would remain valid,
but it would be a different decision problem with endogenous targets. This
is why `decision_refresh.py` explicitly freezes candidates during auditing.

The meaningful next theoretical problem is budget allocation among future
verifier refits, current certification and fresh proposal generation under an
endogenous trajectory. Solving one threshold decision cannot claim to solve it.

## Distinction from adjacent work

Gao et al. (2023), https://proceedings.mlr.press/v202/gao23h.html, measure
reward overoptimization; they do not make this finite-support audit reduction
novel. Laroche et al. (2019),
https://proceedings.mlr.press/v97/laroche19a.html, precede this project in safe
policy improvement with a baseline. Dughmi et al. (2026),
https://arxiv.org/html/2605.17609v2, optimize generated-candidate search until
a verified positive under monotone score/success structure. Here the target
is the signed expected reward difference of a frozen policy comparison and no
monotone cheap-score assumption is used. Moya et al. (2026),
https://arxiv.org/abs/2609.35677, study imperfect-verifier gradient-flow reward
hacking and partial-audit selective correction. The remaining potential RVL
contribution is **endogenous comparison/audit interaction**, if its advantages
survive prospectively matched-cost, new-support experiments.
