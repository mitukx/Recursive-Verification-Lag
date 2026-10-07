import asyncio
import unittest

from src.rvl_systems.types import Generation
from src.rvl_systems.verifier import ExactMatchVerifier
from src.rvl_systems.verifier_rpc import (
    DistributedVerifierFleet,
    TCPVerifierClient,
    VerifierWorkerServer,
)


def generation(response: str = "42") -> Generation:
    return Generation("p", "6*7?", response, -0.1, 1, 0.01, {"policy_version": 3})


class VerifierRPCTest(unittest.TestCase):
    def test_remote_roundtrip_preserves_generation_and_version(self):
        async def run():
            server = VerifierWorkerServer(
                ExactMatchVerifier({"p": "42"}), worker_id="v0", capacity=2
            )
            host, port = await server.start()
            try:
                client = TCPVerifierClient(host, port)
                info = await client.ping()
                self.assertTrue(info["healthy"])
                self.assertEqual(info["worker_id"], "v0")
                self.assertEqual(info["verifier_version"], 0)
                out = await client.verify(generation(), expected_verifier_version=0)
                self.assertEqual(out.generation, generation())
                self.assertEqual(out.reward, 1.0)
                self.assertEqual(out.verifier_version, 0)
                self.assertEqual(out.metadata["rpc_worker_id"], "v0")
            finally:
                await server.close()
        asyncio.run(run())

    def test_version_mismatch_fails_closed(self):
        async def run():
            verifier = ExactMatchVerifier({"p": "42"})
            server = VerifierWorkerServer(verifier, worker_id="stale")
            host, port = await server.start()
            try:
                verifier.refresh()
                client = TCPVerifierClient(host, port)
                with self.assertRaisesRegex(RuntimeError, "does not match expected"):
                    await client.verify(generation(), expected_verifier_version=0)
            finally:
                await server.close()
        asyncio.run(run())

    def test_fleet_quarantines_stale_worker_and_fails_over(self):
        async def run():
            stale_verifier = ExactMatchVerifier({"p": "42"})
            stale_verifier.refresh()
            stale = VerifierWorkerServer(stale_verifier, worker_id="stale")
            fresh = VerifierWorkerServer(ExactMatchVerifier({"p": "42"}), worker_id="fresh")
            h1, p1 = await stale.start()
            h2, p2 = await fresh.start()
            try:
                fleet = DistributedVerifierFleet(
                    [TCPVerifierClient(h1, p1), TCPVerifierClient(h2, p2)],
                    expected_verifier_version=0,
                )
                health = await fleet.refresh_health()
                self.assertFalse(health[0]["compatible"])
                self.assertTrue(health[1]["compatible"])
                out = await fleet.verify(generation())
                self.assertEqual(out.metadata["rpc_worker_id"], "fresh")
                self.assertEqual(fleet.quarantined, (True, False))
            finally:
                await stale.close()
                await fresh.close()
        asyncio.run(run())

    def test_unhealthy_worker_is_not_used(self):
        async def run():
            bad = VerifierWorkerServer(ExactMatchVerifier({"p": "42"}), worker_id="bad")
            good = VerifierWorkerServer(ExactMatchVerifier({"p": "42"}), worker_id="good")
            bad.set_health(False)
            h1, p1 = await bad.start()
            h2, p2 = await good.start()
            try:
                fleet = DistributedVerifierFleet(
                    [TCPVerifierClient(h1, p1), TCPVerifierClient(h2, p2)],
                    expected_verifier_version=0,
                )
                await fleet.refresh_health()
                out = await fleet.verify(generation())
                self.assertEqual(out.metadata["rpc_worker_id"], "good")
            finally:
                await bad.close()
                await good.close()
        asyncio.run(run())

    def test_expected_version_publish_fences_old_workers(self):
        async def run():
            verifier = ExactMatchVerifier({"p": "42"})
            server = VerifierWorkerServer(verifier, worker_id="v")
            host, port = await server.start()
            try:
                fleet = DistributedVerifierFleet(
                    [TCPVerifierClient(host, port)], expected_verifier_version=0
                )
                verifier.refresh()
                fleet.publish_expected_version(1)
                health = await fleet.refresh_health()
                self.assertTrue(health[0]["compatible"])
                self.assertEqual((await fleet.verify(generation())).verifier_version, 1)
                with self.assertRaises(ValueError):
                    fleet.publish_expected_version(0)
            finally:
                await server.close()
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
