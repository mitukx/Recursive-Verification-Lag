# Timing identification: what this follow-up can establish

Status: design and elementary algebra; no new optimality theorem.

## Causal controls

A fixed audit stream draws distinct source-text identities without replacement,
independently of policy and hidden reward, before the loop begins. Every schedule
reveals its first two labels before update 1 and the same next four labels in two
batches before update 12. The verifier is always fitted from the revealed prefix.
Thus comparisons within this stream share initial policy, initial verifier,
label identity, label order, total calls, final verifier and optimizer strength.
Only reveal times (and the induced policy trajectory) differ. Adaptive deadlines
are explicit interventions, not safety guarantees.

The companion acquisition condition chooses unaudited source groups using current
policy mass with 5% uniform exploration, after the same initial two labels. This
condition holds total unique-source calls fixed but allows label identity to vary.
Its contrast with the stream measures the combined acquisition-policy effect;
it is not a pure timing contrast. Two different source strings can still implement
the same function. Source identity does not establish semantic independence.

## Exact exposure identity for soft updates

For a fixed stream let v_k be the score after the kth label batch, and let L_k be
the number of updates using that score. The score vectors depend only on the
revealed label prefix. For constant eta and a strictly positive base distribution,

p_T(i) = p_0(i) exp(eta sum_k L_k v_k(i)) / Z.

This follows by multiplying normalized exponential updates; normalization factors
cancel into Z. It is an instance of the existing composed-block identity, not a
novel theorem. Timing can change final reward even when the final verifier and
all information acquired are identical, because the policy accumulates different
exposure to intermediate scores. Intermediate failures are not determined by the
endpoint identity alone. Numeric support underflow can limit exact computation.

Consequently, a timing effect alone does not establish a new recursive phenomenon
beyond cumulative score exposure. The research must test which *unresolved error
contrasts* make that exposure harmful and whether an observable, assumption-valid
certificate can identify them before outcome measurement. Existing conditional
ellipsoid/box results must not be relabeled as a new minimax theorem.

## Prior-art boundary (primary sources inspected 2026-09-22)

- Gao, Schulman, Hilton, ICML 2023, Scaling Laws for Reward Model Overoptimization:
  https://proceedings.mlr.press/v202/gao23h.html . Establishes optimization against
  imperfect reward models and differences across optimization procedures.
- Yang et al., 2026-09-04, HackProbe, arXiv:2609.04665v1:
  https://arxiv.org/html/2609.04665v1 . A secret comparison core, rotating probe
  layer and intervention address reward hacking in self-evolving systems. Its
  evaluated host uses controlled injected channels. Detection and intervention
  alone cannot be RVL's distinguishing claim.
- Beigi et al., 2026-02-02, Adversarial Reward Auditing, arXiv:2602.01750v1:
  https://arxiv.org/html/2602.01750v1 . Learns an auditor through adversarial hacking
  and uses auditor-guided reward gating. Active auditing alone is not new.

This is a focused overlap check, not a complete novelty review or endorsement of
all claims in these preprints. RVL's prospective distinction is a falsifiable
information-cost / policy-movement relation with controlled refresh timing and
assumption-explicit guarantees. That distinction remains to be demonstrated.
