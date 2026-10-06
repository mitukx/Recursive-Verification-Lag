# Safety, information, and progress are different objectives

Status: elementary propositions, an exact small control reference, and a
retrospective falsification study. These are not new general dual-control
theorems. The finite bank and binary deterministic trusted outcomes are assumed
throughout. A separately locked fresh-task development experiment tests a
limited transfer claim rather than transporting these certificates.

## 1. A current-decision certificate can create an absorbing learning trap

Let a verifier be fitted by positive-ridge least squares to paid labels, let
soft exponential tilting or tie-preserving best-of-N generate a proposal, and
let additional labels be acquired only while that proposal's sharp gain
interval is unresolved. Fix a strictly positive gain floor gamma. Suppose all
initial paid labels equal zero.

**Proposition 1.** In exact arithmetic this architecture makes no further
queries or updates, regardless of the positive outcomes among unpaid sources.

Proof. Ridge regression solves theta=(X'X+rho I)^(-1)X'y=0. The fitted score is
constant zero, and each specified optimizer preserves the current policy p.
Initially p=b, so the contrast q-b is zero and its exact interval is [0,0].
Since gamma>0, the comparison is already rejected and no query is permitted.
The paid transcript and policy remain unchanged. Induction gives the result
for every future round. QED. With gamma=0 the identity update can be accepted,
but still supplies no information or policy progress.

This is a restricted exploration failure, not an impossibility theorem for
refreshing verifiers. Other estimators, optimistic priors, proactive queries,
or policies that explore despite a resolved comparison can escape it.
Numerical implementations can perturb an identity tilt at roundoff scale;
the .01 experiment floor is much larger than these perturbations.

**Strict separation example.** Three sources have uniform baseline mass,
public features (0,0,1), and rewards (0,0,R). The first two labels are paid.
Current-decision acquisition has zero gain and zero additional cost. Query the
third source once. If R=1, the ridge slope is

\[
s=\frac{2+\rho}{(3+\rho)(1+\rho)-1}>0.
\]

One soft update of strength eta gives the third source probability
exp(eta s)/(2+exp(eta s)); all rewards are now known, so its gain
Delta=exp(eta s)/(2+exp(eta s))-1/3 is exactly certified. For any
0<gamma<Delta, an independent prior Pr(R=1)=pi>0 gives expected terminal gain
pi Delta>0 with at most one additional label. Without a new label the
specified zero-label ridge architecture cannot change its proposal. Thus the
one-label versus zero-progress separation is sharp for this example. It does
not guarantee positive realized progress: the all-zero reward world remains.

## 2. The objective must include how a paid label changes future proposals

A state is (z,p,k,h): paid binary assignment z, deployed policy p, remaining
source-label cap k, and remaining accepted-update slots h. The baseline b,
cheap features X, ridge estimator, optimizer and gamma are fixed. Fitting on z
gives q(z,p). A deployed update is legal only if its source-box lower bound
against b is at least gamma. Allowed actions are stop, query an unpaid source
and refit, or apply that currently fitted certified proposal. No unpaid reward
enters an action. Rejected proposals need not consume an accepted-update slot.

Under a declared independent binary prior, let the terminal value be

\[
G(z,p)=\sum_g (p_g-b_g)\,\mathbb E[r_g\mid z].
\]

At h=0, stop. Otherwise the Bellman candidates are:

\[
G(z,p),\quad V(z,q(z,p),k,h-1)\ \text{if certified},
\]
\[
(1-\pi_g)V(z\cup\{g=0\},p,k-1,h)
+\pi_gV(z\cup\{g=1\},p,k-1,h),\quad g\notin z,\ k>0.
\]

Maximize expected terminal gain and, among utility ties, minimize expected
additional paid labels. Query costs obey the analogous recursion with an
added unit cost. Every non-stop action decreases k+h. Backward induction
therefore proves prior-optimality in this finite action family. Policies are
part of the cache key, so a bound counting only 3^G transcripts would be
incorrect. `src/endogenous_audit.py` caps groups at eight and expanded states
at 200,000; exceeding the cap raises rather than silently changing algorithms.
Utility tie comparisons tolerate 1e-12 float error; acceptance never relaxes
the lower-bound safety threshold.

This reference refits after every query. It differs from freezing a proposal
through an audit episode in `decision_refresh.py`; neither optimality claim
automatically covers the other's action space. The proposed policy remains
adaptive, but each applied lower-bound certificate contains only paid labels.
Consequently every deployed policy has gain >= gamma against b on this bank
or remains b. This is baseline safety, not monotone stepwise improvement,
expected-cost superiority under a wrong prior, or safety on new support.

This formulation is a standard Bayes-adaptive control problem. Dual control
already treats actions and future observations jointly: Klenske and Hennig
(2016), https://www.jmlr.org/papers/v17/15-162.html . Active reward learning
alongside policy optimization also predates this project: Su et al. (2016),
https://arxiv.org/abs/1605.07669 . Our narrow added diagnostic is the interaction
between exact baseline certificates, proposal generation and a zero-label
absorbing state; an executable Bellman recursion alone is not novelty.

## 3. The strongest finite-bank baseline can bypass verifier learning

Let A_1 and A_0 be paid positive and negative sources, and let B_1 and B_0 be
their baseline probability masses. If B_1>0, define q as the baseline
conditioned on A_1. Its true reward is exactly one. For the sharp independent
[0,1] source box,

\[
L_A(q,b)=(1-B_1)-b(A^c)=B_0.
\]

**Proposition 2.** If B_0>=gamma, directly selecting a paid known positive is
certified and attains the maximum possible realized reward one. No fitted
verifier can exceed its terminal reward on the same deterministic bank.

Proof. Positive contrast weights occur only on known positives; all other
weights are negative. Substitute these into the box lower bound. The realized
reward is one because q has support only on known positives, and rewards are
bounded above by one. QED. If no paid positive exists, or B_0<gamma, this
baseline can query a predeclared stream or abstain. Binary labels matter for
the reward-one statement. This method has access to paid source labels, not
to unpaid evaluator outcomes.

The `direct` control implements this baseline with the same initial sources
and additional-label cap. It intentionally leaves the ridge-proposal action
family, so the exact planner is not an upper bound on its performance.
If direct selection wins, fixed-bank progress cannot demonstrate learned
capability creation. Transfer to unpaid new tasks, new generated support or
an updated generator requires different evidence.

## 4. What the completed full-pilot comparison supports

All eight tasks, two representations, soft eta=1 and best-of-N=4, and five
audit seeds give 160 correlated settings per controller. Each starts with the
same two paid source labels, permits two additional labels, and permits two
accepted updates. Independent planning priors .1/.5/.9 are sensitivity
conditions; .5 is the primary reference. No calibration or best-prior selection
is claimed. Full records are in `results/endogenous_audit_mbppplus_v1/`.

| Controller | Mean actual source labels | Mean terminal baseline gain |
| --- | ---: | ---: |
| Myopic, query only for current unresolved comparison | 2.7875 | .105999 |
| One forced stream query from an all-zero paid transcript | 3.23125 | .122718 |
| Exact two-update lookahead, prior .5 | 3.35625 | .122825 |
| Replan for one update at a time, prior .5 | 3.34375 | .125735 |
| Spend all stream queries before updates | 4.0000 | .048124 |
| Direct selection of a paid known positive | 3.3250 | .165625 |

The lookahead-minus-myopic descriptive task-bootstrap gain interval is
[-.008622,.047174]. The one-query escape accounts for 99.36% of that mean
difference; this ratio is a post-hoc mechanism diagnostic, not a population
effect. Eight settings on task4/seed3 and task606/seed4 have initial-zero labels
despite positive candidates. Myopic never escapes these states; lookahead and
the one-query escape do so in all eight. Three all-zero tasks also consume
proactive labels with no possible realized gain. The primary exact planner
is empirically inferior in both mean gain and mean label cost to direct
selection, and loses to a shorter planning horizon on realized mean gain.
These are aggregate retrospective comparisons, not per-instance dominance
or confirmatory significance claims. Independent-prior optimality survives
this negative empirical result; real-prior superiority does not follow.
