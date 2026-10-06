import asyncio
import tempfile
import unittest

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
