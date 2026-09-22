# Execution recovery and evidence status

The first CPU pilot completed 192 pretrained candidates and 14,400 recursive
runs. Implementation commit `02de4962e6c5052cd904d023674660f40826518a` was
saved to GitHub before outcome analysis. The workspace was reset before its bank,
trajectories, final report and post-commit edits were pushed. Raw outputs from that
execution could not be recovered. Its console aggregates remain in the conversation
but are **not** treated as an auditable published result.

Historical console-only totals: 1,197 baseline-crossing runs / 14,400; 620 first-round
crossings; 299 first crossings with stale age > 1; 936 declines exceeding .01; 668
exceeding .05; worst baseline change -0.2440134099616859. These numbers must not be
presented as a substitute for the missing raw bank or reconstructed into fake rows.

The recovery execution uses the same immutable model revision and declared sampling
settings, but a distinct `recovered_qwen05b` artifact identifier. The old bank's hash
and complete environment manifest are unavailable, so bitwise identity to it cannot
be verified even if aggregates match. Raw candidates and metadata must be committed
before the recovery sweep begins, and all recovery outcomes must then be archived.

The post-commit parser hardening (reject multi-statement function wrappers instead
of taking their first return) is restored. Before the reset, all 192 original rows
were rescored under it and matched their original stored scores/features exactly.
That check survives as console output only. The recovery run records the current
parser hash, rather than implying that the older generator source is unchanged.

The shared-initially-safe cohort analysis is a **post-hoc diagnostic**, motivated
by the first pilot's high fraction of initial-step failures. It is not part of the
original confirmatory protocol. It verifies that first-round states coincide across
fixed cadences, then uses a common offline outcome-based inclusion criterion. It
reports actual label counts and does not claim matched-cost superiority.
