# Verification-aware GPU phase execution contract

Issue #66 already locks the scientific phase diagram. The execution layer now
turns that lock into a machine-enforced experiment plan rather than a prose
checklist.

## Plan compilation

`src.verification_aware_phase_plan` expands every locked phase, arm, override,
and seed into a deterministic cell ID. Each cell contains the exact effective
system configuration plus pinned model/dataset/generation metadata. The plan
records both the protocol SHA-256 and source commit SHA.

The compiler does not inspect outcomes and does not tune the sweep.

## Driver contract

`src.run_verification_aware_gpu_phase` executes one or more cells. The child
driver receives the following environment variables:

- `RVL_PHASE_CELL_ID`
- `RVL_PHASE_NAME`
- `RVL_PHASE_ARM`
- `RVL_PHASE_SEED`
- `RVL_PHASE_PROTOCOL_SHA256`
- `RVL_PHASE_SOURCE_SHA`
- `RVL_PHASE_EFFECTIVE_SYSTEM_JSON`
- `RVL_EVIDENCE_DIR`

A successful driver must write `timeseries.jsonl` and `summary.json` to
`RVL_EVIDENCE_DIR`. The phase runner independently records hardware, stdout,
stderr, command, wall time, source/protocol identity, GPU telemetry, and a
SHA-256 manifest.

A failed cell is retained as `failure.json`; it is not silently dropped or
replaced by another configuration.

## Evidence validation

`src.validate_verification_aware_gpu_evidence` checks that every completed cell
contains every timeseries and terminal metric prospectively required by the
locked protocol, and that every hard invariant was explicitly evaluated and
passed. Failed cells are valid evidence only if their failure record and hashes
are retained. Full-bundle validation fails if an expected cell is missing.

This infrastructure is evidence plumbing, not a GPU result. It makes the future
real-GPU run auditable and prevents success-only result selection.
