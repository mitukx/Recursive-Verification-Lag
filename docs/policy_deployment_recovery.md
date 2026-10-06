# Crash-consistent policy deployment

The rollout control plane must preserve policy-version safety across trainer or coordinator process failure. In-memory two-phase commit is insufficient because a restart can otherwise reuse a version, forget worker acknowledgements, or let an obsolete coordinator activate a policy after a replacement has taken over.

## Durable invariants

1. **Manifested versions are never reused.** `WeightPublisher` recovers the highest immutable weight manifest and allocates only larger versions.
2. **Acknowledgements are monotonic and durable.** Worker ACKs survive restart and may never move backwards or acknowledge an unpublished version.
3. **Pending deployment survives restart.** The active version, pending version, required worker set, and coordinator epoch are atomically persisted.
4. **Recovery verifies bytes before control resumes.** A pending artifact whose byte count or SHA-256 differs from its manifest prevents coordinator startup.
5. **Coordinator takeover fences stale writers.** Every coordinator startup advances a durable epoch. Publish, ACK, activation, and status checks compare their local epoch to durable state and fail closed after a newer coordinator takes over.
6. **Activation remains monotonic.** A recovered deployment may skip an unreferenced version left by a crash between manifest publication and deployment-state commit, but an active version can never move backwards.

The filesystem reference implementation uses same-directory atomic rename for small JSON control records and immutable weight artifacts/manifests. This proves the state-machine contract on one durable filesystem; it is not a claim of a multi-node consensus service.

## Evidence

Run:

    python -m src.benchmark_policy_deployment_recovery --output artifacts/policy-deployment-recovery.json

The benchmark exercises partial-ACK recovery, active-state recovery, monotonic version allocation after restart, stale-coordinator fencing, and fail-closed recovery from a corrupted pending artifact. Systems CI retains the JSON output inside the SHA-linked artifact bundle.
