# Progress XXX — Partial certified updates barely recover progress

## Hypothesis, experiment, outcome

**Hypothesis.** A robust controller that abstains on uncertifiable proposals
might leave safe progress unused. After exhausting the same six-source budget,
mix the current certified policy with the optimizer's proposed policy, take
the largest fraction whose sharp lower bound versus the initial policy is
nonnegative, and retain abstention if no positive fraction exists. Keep the
same two observed 1.5B expression banks, eight tasks per bank, 160
configurations per bank, 12 rounds, optimizers, feature representations,
seeds, and acquisition rule. This is exploratory *after* the earlier bank
outcomes were inspected, not prospective validation.

**Outcome.** Zero failures remain by construction on both frozen banks.
At 12 rounds, v1 mean gain rises from .188034 to .188778 (+.000743), and v2
from .101377 to .101810 (+.000433). Both changes are small relative to the
fixed six-source uniform gains of .365623 (v1 subset) and .278445 (v2 subset).
Task-cluster descriptive 95% bootstrap intervals for paired gain differences
are [−.000014, +.002244] (v1) and [+.000101, +.000865] (v2); eight tasks and
many correlated settings do not support a population-scale claim.
Paid distinct-source means remain 4.7875 (v1) and 4.75 (v2); the initial
two-source stream and acquisition strategy are identical across paired
controllers. A nonzero partial line step occurs in only 3/1920 trajectory
states on v1 and 10/1920 on v2. At some settings it lowers realized gain
relative to retaining the earlier safe policy, since the guarantee compares
to the initial policy rather than requiring monotonic reward at every step.

The explanation is observable. At budget exhaustion, there are 816 (v1) and
978 (v2) rejected full proposals. In the 813 and 968 cases where the mixed
controller ultimately abstains, respectively, the current certificate lower
bound is zero and the right derivative of the bound in the proposed direction
is **strictly negative**. The boundary lemma in
`notes/theory/identified_refresh_frontier.md` proves that no strictly positive
interpolation step along that particular direction can pass the certificate
without new information or a changed proposal. The remaining 3 and 10 cases
have a nonzero certifiable fraction. These are correlated states across
trajectories, not independent empirical replications.
The least-negative abstention slopes are −.000670 (v1) and −.0000605 (v2),
well away from the 1e-10 numerical classification tolerance.

## Falsifiability and next decision

The hypothesis that simple line interpolation substantially rescues useful
progress is rejected on both observed suites. This does not prove that all
safe adaptive controllers must lose this much gain: alternate source queries,
coupled error assumptions, richer proposals, or an explicitly estimated
residual envelope can change the feasible set. The bounded-reward box
certificate is assumption-light and often reaches a sharp boundary; a new
Oral-level contribution must explain and predict the cost/progress frontier
under justified structure. Do not report these expression-bank results as
general coding-agent performance. The independent MBPP+ raw bank remains
**unscored** until isolated execution passes the reference-validation gate.
