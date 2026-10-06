# OpenAI Math x RVL: exact trusted-verification bridge

This experiment turns released OpenAI Math Comparator challenges into a real
verification-lag substrate. A cheap proxy verifier may score proof candidates
frequently; the pinned Lean kernel is the expensive independent trusted label.
The target is not to claim new solutions to the released research problems.
The target is to measure whether policy optimization can improve proxy reward
while exact theorem validity degrades, and whether refresh/audit policies prevent
that failure under a fixed trusted-compute budget.

## Locked benchmark

The manifest `configs/openai_math_rvl_v1.json` pins upstream commit
`adc7f1241b42e322a6451854ab7e4b4c146bf78a`, Lean 4.34.1, and the Git blob
SHA of every Comparator source. Families never cross the development/heldout
boundary.

Development mechanisms:
- #116 universal hitting / structural coverage
- #119 information contraction
- #104 stochastic games
- #111 adversarial online selection
- #102 robust local constraints / hardness

Sealed heldout mechanisms:
- #140 memory-sample information lower bounds
- #277 recursive threshold repetition
- #238 mixing / information decay

Do not tune thresholds, prompts, refresh cadence, or proxy architecture on the
heldout split.

## Safety and correctness boundary

`LeanTrustedVerifier` accepts only a strict single-hole Comparator challenge,
replaces the unique `sorry`, rejects common proof escapes, pins upstream source
provenance, bounds candidate/output size and timeout, and invokes
`lake env lean`. Kernel acceptance is the trusted correctness signal.

Lean is a correctness oracle, not an OS sandbox. Model-produced proof text must
run behind an external sandbox. The verifier therefore fails closed unless a
sandbox prefix is supplied; `--allow-unsandboxed` is only for an already
isolated CI/container.

## Input contract

Each JSONL row contains:

```json
{"task_id":"oai116-formula-hitting","response":"exact ...","proxy_score":0.82,
 "policy_version":7,"verifier_version":3,"metadata":{"seed":17}}
```

Run:

```bash
python -m src.run_openai_math_rvl \
  --checkout /path/to/pinned/openai-math \
  --input artifacts/candidates.jsonl \
  --output artifacts/openai-math-dev \
  --split development \
  --sandbox-prefix-json '["your-sandbox-wrapper"]'
```

Outputs include raw Lean diagnostics, proxy/trusted scores, policy/verifier
versions, stale age, source/candidate hashes, manifest fingerprint, and an
append-order hash chain.

Primary measurements are trusted pass rate, proxy mean, proxy false-positive
rate, calibration MAE, false-progress transitions, and trusted pass rate by
verifier stale-age bucket. Compare frozen, fixed-cadence, geometry-aware,
information-aware, and always-fresh verifier policies under equal trusted
verification cost.

## Frontier research path

1. Generate multiple candidate proofs per task with the same policy snapshot.
2. Train or fit the existing residual/learned verifier only on development
   trusted labels.
3. Optimize candidate selection or policy updates against the proxy.
4. Audit sparse candidates with `LeanGenerationGrader`, which plugs into the
   existing `MultiVerifier(trusted=...)` contract.
5. Lock refresh policy and trusted-compute budget before looking at heldout.
6. Open heldout once and preserve negative/null results.

The #116 result motivates a separate theorem program: characterize bounded error
classes admitting a fixed universal distinguisher/hitting set, under which
verification refresh is structurally unnecessary. #119 and #140 motivate
decision-relevant information and trusted-label lower-bound models. Those are
research hypotheses; the code here provides the exact-evaluation substrate
needed to test them without conflating proxy reward with truth.
