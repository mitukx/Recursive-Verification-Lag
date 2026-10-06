import asyncio
import unittest

from src.rvl_systems.backends import ToyTabularBackend
from src.rvl_systems.worker_rpc import RolloutWorkerServer, TCPWorkerBackend


class WorkerRPCTest(unittest.TestCase):
    def test_roundtrip_and_policy_version(self):
        async def run():
            server = RolloutWorkerServer(
                ToyTabularBackend(actions=("0", "1")),
                policy_version=3,
            )
            host, port = await server.start()
            try:
                client = TCPWorkerBackend(
                    host,
                    port,
                    expected_policy_version=3,
                )
                self.assertEqual(await client.ping(), 3)
                rows = await client.generate(
                    "p",
                    "prompt",
                    n=3,
                    temperature=1.0,
                    seed=7,
                )
                self.assertEqual(len(rows), 3)
                self.assertTrue(all(r.metadata["policy_version"] == 3 for r in rows))
                self.assertTrue(all("rpc_worker" in r.metadata for r in rows))
            finally:
                await server.close()
        asyncio.run(run())

    def test_stale_client_rejects_new_worker_version(self):
        async def run():
            server = RolloutWorkerServer(ToyTabularBackend(), policy_version=1)
            host, port = await server.start()
            try:
                client = TCPWorkerBackend(host, port, expected_policy_version=1)
                server.publish_policy_version(2)
                with self.assertRaisesRegex(RuntimeError, "does not match expected"):
                    await client.generate(
                        "p", "prompt", n=1, temperature=1.0, seed=1
                    )
            finally:
                await server.close()
        asyncio.run(run())

    def test_server_rejects_backward_version(self):
        server = RolloutWorkerServer(ToyTabularBackend(), policy_version=2)
        with self.assertRaises(ValueError):
            server.publish_policy_version(1)


if __name__ == "__main__":
    unittest.main()
