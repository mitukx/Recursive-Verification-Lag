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
