import asyncio
import unittest

from src.rvl_systems.backends import ToyTabularBackend
from src.rvl_systems.remote_backend import VersionedRemoteBackend
from src.rvl_systems.telemetry import Telemetry
from src.rvl_systems.worker_pool import RetryPolicy, VersionedWorkerPool, with_retries


class RemoteWorkerTest(unittest.TestCase):
    def test_retry_succeeds_after_transient_failures(self):
        async def run():
            calls = {"n": 0}
            telemetry = Telemetry()

            async def flaky():
                calls["n"] += 1
                if calls["n"] < 3:
                    raise RuntimeError("transient")
                return 7

            result = await with_retries(
                flaky,
                policy=RetryPolicy(attempts=3, backoff_s=0.0),
                telemetry=telemetry,
            )
            self.assertEqual(result, 7)
            self.assertEqual(calls["n"], 3)
            snap = telemetry.snapshot()
            self.assertEqual(snap["worker.failures"], 2)
            self.assertEqual(snap["worker.success"], 1)
        asyncio.run(run())

    def test_stale_result_is_rejected(self):
        async def run():
            pool = VersionedWorkerPool()
            backend = VersionedRemoteBackend(
                ToyTabularBackend(actions=("0", "1")),
                pool,
                retry=RetryPolicy(attempts=1),
            )
            generations = await backend.generate(
                "p", "prompt", n=2, temperature=1.0, seed=1
            )
            backend.validate(generations)
            pool.publish_new_version()
            with self.assertRaises(RuntimeError):
                backend.validate(generations)
        asyncio.run(run())

    def test_fresh_version_after_publish_is_accepted(self):
        async def run():
            pool = VersionedWorkerPool()
            pool.publish_new_version()
            backend = VersionedRemoteBackend(
                ToyTabularBackend(actions=("0", "1")),
                pool,
            )
            generations = await backend.generate(
                "p", "prompt", n=1, temperature=1.0, seed=3
            )
            backend.validate(generations)
            self.assertEqual(generations[0].metadata["policy_version"], 1)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
