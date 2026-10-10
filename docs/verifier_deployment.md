# Two-phase verifier deployment

Remote verifier workers now support a durable deployment protocol separate from
ordinary verification RPC.

A deployment publishes an immutable artifact with a monotonically increasing
version and SHA-256 identity. The coordinator then:

1. sends `prepare_verifier` to every worker;
2. each worker verifies the artifact checksum, loads it through an explicit
   application-provided loader, persists its prepared state, and ACKs;
3. only after every worker ACKs does the coordinator send `activate_verifier`;
4. activation swaps the verifier version and persists active artifact identity.

Coordinator state carries a monotonically increasing epoch. Constructing a new
coordinator instance advances the durable epoch; once workers observe it,
mutations from an older coordinator are rejected. Stale coordinators are also
prevented from overwriting the newer durable coordinator state.

Worker restart reloads the active artifact and any prepared artifact from its
durable state. Artifact corruption fails closed before the RPC server starts.

Activation additionally fences requests that began under the previous verifier:
if the active version changes while verification is in flight, the old result is
rejected instead of being returned as a valid reward.

The artifact loader is explicit and supplied by the embedding application. This
control plane does not execute arbitrary artifact content and does not claim the
loaded verifier is semantically correct; trusted evaluation remains separate.

## Prepare commit fencing

An application-provided artifact loader may be asynchronous. A prepare must
revalidate its coordinator epoch and active manifest after loading returns,
before publishing any prepared state. A coordinator takeover during loading
rejects the old prepare; an active-version advance rejects an older candidate.
If the same version activated during loading, only the identical manifest is
accepted idempotently. These checks and the prepared-state write have no
intervening await in the worker's event loop.

`python -m unittest tests.test_verifier_prepare_fencing -v` exercises
event-controlled localhost TCP schedules, conflicting artifacts with the same
version, an active-version advance, idempotence, and restart. It uses no model
downloads. This contract is scoped to one worker/event loop and its existing
persistence model; it is not fleet-wide atomic activation or a multi-host
consensus guarantee.

## Same-epoch identity and concurrency

Within one coordinator instance, public `prepare`, `activate`, and the entire
`deploy` operation are serialized with an asyncio lock. Separate `prepare` calls
replace the pending candidate; `activate()` targets the latest pending manifest,
not a transaction lease belonging to an earlier caller. Use `deploy(manifest)`
when callers require an indivisible coordinator operation. This lock is local to
one instance/event loop and does not serialize independent processes.

Worker management RPCs can still overlap in the same epoch. Before activation,
artifacts from different publication roots can carry the same version and different
digests; a version alone is not an artifact identity. The activation RPC therefore
requires `artifact_sha256` as well as version and epoch. Missing digests fail closed
(old management clients must upgrade). Both normal and idempotent activation must
match that digest. Prepare ACKs, activation ACKs, and final health checks must all
match the intended artifact; activation ACKs also match the coordinator epoch.

A concurrent raw prepare may replace a prepared candidate after its ACK. The
contract guarantees rejection rather than successful activation of the wrong
artifact; it does not guarantee progress for arbitrary concurrent raw writers.
A rejected activation retains coordinator pending state and can be retried by
explicitly redeploying the intended manifest. It does not roll back workers that
already activated. Once a version is active, a different artifact under that
version remains invalid. Fleet-wide atomicity, process-safe coordinator epoch
allocation, and semantic verifier quality are outside this contract.

`python -m unittest tests.test_verifier_same_epoch -v` covers the deterministic
same-epoch overlapping-loader schedule, identical-artifact control, coordinator
serialization, and deliberately mismatched ACK/health identities.
