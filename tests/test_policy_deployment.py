import asyncio
import tempfile
import unittest
from pathlib import Path

from src.rvl_systems.backends import ToyTabularBackend
from src.rvl_systems.policy_deployment import PolicyDeploymentCoordinator
from src.rvl_systems.worker_rpc import RolloutWorkerServer, TCPWorkerBackend


class PolicyDeploymentTest(unittest.TestCase):
    def test_two_phase_activation_requires_every_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = PolicyDeploymentCoordinator(tmp, ["w0", "w1"])
            manifest = coordinator.publish(b"weights-v1")
            self.assertEqual(manifest.version, 1)
            self.assertFalse(coordinator.can_activate())
            coordinator.acknowledge("w0", 1)
            self.assertFalse(coordinator.can_activate())
            with self.assertRaisesRegex(RuntimeError, "have not acknowledged"):
                coordinator.activate()
            coordinator.acknowledge("w1", 1)
            self.assertTrue(coordinator.can_activate())
            self.assertEqual(coordinator.activate(), 1)
            coordinator.assert_rollout_version(1)
            with self.assertRaises(RuntimeError):
                coordinator.assert_rollout_version(0)

    def test_cannot_overlap_deployments_or_accept_wrong_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = PolicyDeploymentCoordinator(tmp, ["w0"])
            coordinator.publish(b"v1")
            with self.assertRaises(RuntimeError):
                coordinator.publish(b"v2")
            with self.assertRaises(ValueError):
                coordinator.acknowledge("w0", 2)
            with self.assertRaises(KeyError):
                coordinator.acknowledge("unknown", 1)

    def test_restart_recovers_partial_ack_then_active_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = PolicyDeploymentCoordinator(tmp, ["w0", "w1"])
            pending = first.publish(b"v1")
            first.acknowledge("w0", pending.version)

            recovered = PolicyDeploymentCoordinator(tmp, ["w0", "w1"])
            status = recovered.status()
            self.assertEqual(status.active_version, 0)
            self.assertEqual(status.pending_version, 1)
            self.assertEqual(status.workers_behind, {"w1": 1})
            recovered.acknowledge("w1", 1)
            self.assertEqual(recovered.activate(), 1)

            restarted = PolicyDeploymentCoordinator(tmp, ["w0", "w1"])
            self.assertEqual(restarted.status().active_version, 1)
            self.assertIsNone(restarted.status().pending_version)
            self.assertEqual(restarted.publish(b"v2").version, 2)

    def test_new_epoch_fences_stale_coordinator(self):
        with tempfile.TemporaryDirectory() as tmp:
            stale = PolicyDeploymentCoordinator(tmp, ["w0"])
            fresh = PolicyDeploymentCoordinator(tmp, ["w0"])
            with self.assertRaisesRegex(RuntimeError, "stale policy deployment coordinator"):
                stale.publish(b"must-not-publish")
            manifest = fresh.publish(b"fresh")
            with self.assertRaisesRegex(RuntimeError, "stale policy deployment coordinator"):
                stale.acknowledge("w0", manifest.version)

    def test_corrupt_pending_artifact_fails_closed_on_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = PolicyDeploymentCoordinator(tmp, ["w0"])
            pending = coordinator.publish(b"good")
            (Path(tmp) / pending.artifact).write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                PolicyDeploymentCoordinator(tmp, ["w0"])

    def test_worker_set_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            PolicyDeploymentCoordinator(tmp, ["w0", "w1"])
            with self.assertRaisesRegex(RuntimeError, "worker set changed"):
                PolicyDeploymentCoordinator(tmp, ["w0"])

    def test_real_rpc_workers_follow_active_version(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                coordinator = PolicyDeploymentCoordinator(tmp, ["w0", "w1"])
                servers = [
                    RolloutWorkerServer(ToyTabularBackend(actions=("0", "1"))),
                    RolloutWorkerServer(ToyTabularBackend(actions=("0", "1"))),
                ]
                addresses = [await server.start() for server in servers]
                try:
                    manifest = coordinator.publish(b"weights-v1")
                    for worker_id, server in zip(coordinator.workers, servers):
                        server.publish_policy_version(manifest.version)
                        coordinator.acknowledge(worker_id, manifest.version)
                    active = coordinator.activate()
                    clients = [
                        TCPWorkerBackend(host, port, expected_policy_version=active)
                        for host, port in addresses
                    ]
                    outputs = await asyncio.gather(*[
                        client.generate(
                            f"p{i}",
                            "prompt",
                            n=1,
                            temperature=1.0,
                            seed=i,
                        )
                        for i, client in enumerate(clients)
                    ])
                    self.assertEqual(
                        [batch[0].metadata["policy_version"] for batch in outputs],
                        [1, 1],
                    )

                    second = coordinator.publish(b"weights-v2")
                    servers[0].publish_policy_version(second.version)
                    coordinator.acknowledge("w0", second.version)
                    self.assertFalse(coordinator.can_activate())
                    self.assertEqual(coordinator.active_version, 1)

                    servers[1].publish_policy_version(second.version)
                    coordinator.acknowledge("w1", second.version)
                    self.assertEqual(coordinator.activate(), 2)
                finally:
                    await asyncio.gather(*(server.close() for server in servers))

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
