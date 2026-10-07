import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from src.rvl_systems.types import Generation, VerifiedGeneration
from src.rvl_systems.verifier import FunctionalVerifier
from src.rvl_systems.verifier_deployment import (
    VerifierDeploymentCoordinator,
    publish_verifier_artifact,
)
from src.rvl_systems.verifier_rpc import TCPVerifierClient, VerifierWorkerServer


def generation():
    return Generation("p", "prompt", "response", 0.0, 1, 0.0, {})


def json_loader(manifest):
    payload = json.loads(Path(manifest.path).read_text())
    reward = float(payload["reward"])
    return FunctionalVerifier(lambda _: reward)


class SlowVerifier:
    def __init__(self, reward: float, delay_s: float):
        self._version = 0
        self.reward = reward
        self.delay_s = delay_s

    @property
    def version(self):
        return self._version

    def refresh(self):
        self._version += 1

    async def verify(self, g):
        await asyncio.sleep(self.delay_s)
        return VerifiedGeneration(g, self.reward, self.delay_s, self._version)


def slow_loader(manifest):
    payload = json.loads(Path(manifest.path).read_text())
    return SlowVerifier(float(payload["reward"]), float(payload.get("delay_s", 0)))


class FailOnceActivationClient:
    def __init__(self, inner):
        self.inner = inner
        self.failed = False

    async def ping(self):
        return await self.inner.ping()

    async def prepare_verifier(self, manifest, *, coordinator_epoch):
        return await self.inner.prepare_verifier(
            manifest, coordinator_epoch=coordinator_epoch
        )

    async def activate_verifier(self, version, *, coordinator_epoch):
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected activation failure")
        return await self.inner.activate_verifier(
            version, coordinator_epoch=coordinator_epoch
        )


class VerifierDeploymentTests(unittest.TestCase):
    def test_two_phase_deploy_converges_two_workers(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                servers = [
                    VerifierWorkerServer(
                        FunctionalVerifier(lambda _: 0.0),
                        worker_id=f"v{i}",
                        deployment_loader=json_loader,
                        deployment_state_path=root / f"worker-{i}.json",
                    )
                    for i in range(2)
                ]
                addresses = [await server.start() for server in servers]
                clients = [TCPVerifierClient(*address) for address in addresses]
                try:
                    manifest = publish_verifier_artifact(
                        root / "artifacts",
                        version=1,
                        content=b'{"reward":0.75}',
                        suffix=".json",
                    )
                    coordinator = VerifierDeploymentCoordinator(
                        clients,
                        state_path=root / "coordinator.json",
                    )
                    self.assertEqual(await coordinator.deploy(manifest), 1)
                    health = await asyncio.gather(*(client.ping() for client in clients))
                    self.assertEqual({row["verifier_version"] for row in health}, {1})
                    self.assertEqual(
                        {row["active_artifact_sha256"] for row in health},
                        {manifest.sha256},
                    )
                    rows = await asyncio.gather(
                        *(client.verify(generation(), expected_verifier_version=1) for client in clients)
                    )
                    self.assertTrue(all(row.reward == 0.75 for row in rows))
                finally:
                    await asyncio.gather(*(server.close() for server in servers))
        asyncio.run(run())

    def test_duplicate_worker_identity_fails_closed(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                server = VerifierWorkerServer(
                    FunctionalVerifier(lambda _: 0.0),
                    worker_id="same",
                    deployment_loader=json_loader,
                    deployment_state_path=root / "worker.json",
                )
                address = await server.start()
                client = TCPVerifierClient(*address)
                coordinator = VerifierDeploymentCoordinator(
                    [client, client], state_path=root / "coordinator.json"
                )
                manifest = publish_verifier_artifact(
                    root / "artifacts",
                    version=1,
                    content=b'{"reward":0.4}',
                    suffix=".json",
                )
                try:
                    with self.assertRaisesRegex(RuntimeError, "duplicate worker identity"):
                        await coordinator.prepare(manifest)
                    self.assertEqual((await client.ping())["verifier_version"], 0)
                finally:
                    await server.close()
        asyncio.run(run())

    def test_partial_activation_retry_converges_idempotently(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                servers = [
                    VerifierWorkerServer(
                        FunctionalVerifier(lambda _: 0.0),
                        worker_id=f"v{i}",
                        deployment_loader=json_loader,
                        deployment_state_path=root / f"worker-{i}.json",
                    )
                    for i in range(2)
                ]
                addresses = [await server.start() for server in servers]
                raw_clients = [TCPVerifierClient(*address) for address in addresses]
                clients = [raw_clients[0], FailOnceActivationClient(raw_clients[1])]
                coordinator = VerifierDeploymentCoordinator(
                    clients, state_path=root / "coordinator.json"
                )
                manifest = publish_verifier_artifact(
                    root / "artifacts",
                    version=1,
                    content=b'{"reward":0.4}',
                    suffix=".json",
                )
                try:
                    with self.assertRaisesRegex(RuntimeError, "activation incomplete"):
                        await coordinator.deploy(manifest)
                    versions = {
                        int((await client.ping())["verifier_version"])
                        for client in raw_clients
                    }
                    self.assertEqual(versions, {0, 1})
                    self.assertEqual(await coordinator.deploy(manifest), 1)
                    health = await asyncio.gather(
                        *(client.ping() for client in raw_clients)
                    )
                    self.assertEqual({row["verifier_version"] for row in health}, {1})
                    self.assertTrue(
                        all(row["prepared_verifier_version"] is None for row in health)
                    )
                finally:
                    await asyncio.gather(*(server.close() for server in servers))
        asyncio.run(run())

    def test_new_coordinator_epoch_fences_old_before_workers_observe_takeover(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                server = VerifierWorkerServer(
                    FunctionalVerifier(lambda _: 0.0),
                    worker_id="v",
                    deployment_loader=json_loader,
                    deployment_state_path=root / "worker.json",
                )
                address = await server.start()
                client = TCPVerifierClient(*address)
                try:
                    m1 = publish_verifier_artifact(
                        root / "artifacts", version=1, content=b'{"reward":0.1}', suffix=".json"
                    )
                    old = VerifierDeploymentCoordinator(
                        [client], state_path=root / "coordinator.json"
                    )
                    await old.deploy(m1)

                    # Takeover advances durable coordinator epoch but has not yet
                    # contacted the worker. The worker therefore still reports
                    # the old epoch, which is the critical race window.
                    new = VerifierDeploymentCoordinator(
                        [client], state_path=root / "coordinator.json"
                    )
                    before = await client.ping()
                    self.assertEqual(before["verifier_version"], 1)
                    self.assertEqual(before["coordinator_epoch"], old.epoch)

                    m3 = publish_verifier_artifact(
                        root / "artifacts", version=3, content=b'{"reward":0.3}', suffix=".json"
                    )
                    with self.assertRaisesRegex(RuntimeError, "stale verifier coordinator fenced"):
                        await old.prepare(m3)

                    # The rejected stale coordinator must not mutate the worker.
                    still_old = await client.ping()
                    self.assertEqual(still_old["verifier_version"], 1)
                    self.assertEqual(still_old["coordinator_epoch"], old.epoch)
                    self.assertIsNone(still_old["prepared_verifier_version"])

                    m2 = publish_verifier_artifact(
                        root / "artifacts", version=2, content=b'{"reward":0.2}', suffix=".json"
                    )
                    await new.deploy(m2)
                    health = await client.ping()
                    self.assertEqual(health["verifier_version"], 2)
                    self.assertEqual(health["coordinator_epoch"], new.epoch)
                    durable = json.loads((root / "coordinator.json").read_text())
                    self.assertEqual(durable["coordinator_epoch"], new.epoch)
                finally:
                    await server.close()
        asyncio.run(run())

    def test_worker_restart_recovers_active_artifact(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                state = root / "worker.json"
                server = VerifierWorkerServer(
                    FunctionalVerifier(lambda _: 0.0),
                    worker_id="v",
                    deployment_loader=json_loader,
                    deployment_state_path=state,
                )
                address = await server.start()
                client = TCPVerifierClient(*address)
                manifest = publish_verifier_artifact(
                    root / "artifacts", version=1, content=b'{"reward":0.6}', suffix=".json"
                )
                coordinator = VerifierDeploymentCoordinator(
                    [client], state_path=root / "coordinator.json"
                )
                await coordinator.deploy(manifest)
                epoch = coordinator.epoch
                await server.close()

                recovered = VerifierWorkerServer(
                    FunctionalVerifier(lambda _: 0.0),
                    worker_id="v",
                    deployment_loader=json_loader,
                    deployment_state_path=state,
                )
                address2 = await recovered.start()
                try:
                    info = await TCPVerifierClient(*address2).ping()
                    self.assertEqual(info["verifier_version"], 1)
                    self.assertEqual(info["coordinator_epoch"], epoch)
                    out = await TCPVerifierClient(*address2).verify(
                        generation(), expected_verifier_version=1
                    )
                    self.assertEqual(out.reward, 0.6)
                finally:
                    await recovered.close()
        asyncio.run(run())

    def test_corrupt_artifact_fails_closed_on_restart(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                state = root / "worker.json"
                server = VerifierWorkerServer(
                    FunctionalVerifier(lambda _: 0.0),
                    worker_id="v",
                    deployment_loader=json_loader,
                    deployment_state_path=state,
                )
                address = await server.start()
                client = TCPVerifierClient(*address)
                manifest = publish_verifier_artifact(
                    root / "artifacts", version=1, content=b'{"reward":0.6}', suffix=".json"
                )
                coordinator = VerifierDeploymentCoordinator(
                    [client], state_path=root / "coordinator.json"
                )
                await coordinator.deploy(manifest)
                await server.close()
                Path(manifest.path).write_text('{"reward":0.9}')
                recovered = VerifierWorkerServer(
                    FunctionalVerifier(lambda _: 0.0),
                    worker_id="v",
                    deployment_loader=json_loader,
                    deployment_state_path=state,
                )
                with self.assertRaisesRegex(RuntimeError, "checksum"):
                    await recovered.start()
        asyncio.run(run())

    def test_activation_fences_old_inflight_reward(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                server = VerifierWorkerServer(
                    FunctionalVerifier(lambda _: 0.0),
                    worker_id="v",
                    deployment_loader=slow_loader,
                    deployment_state_path=root / "worker.json",
                )
                address = await server.start()
                client = TCPVerifierClient(*address)
                coordinator = VerifierDeploymentCoordinator(
                    [client], state_path=root / "coordinator.json"
                )
                m1 = publish_verifier_artifact(
                    root / "artifacts",
                    version=1,
                    content=b'{"reward":0.1,"delay_s":0.05}',
                    suffix=".json",
                )
                m2 = publish_verifier_artifact(
                    root / "artifacts",
                    version=2,
                    content=b'{"reward":0.9,"delay_s":0}',
                    suffix=".json",
                )
                try:
                    await coordinator.deploy(m1)
                    old_request = asyncio.create_task(
                        client.verify(generation(), expected_verifier_version=1)
                    )
                    await asyncio.sleep(0.01)
                    await coordinator.deploy(m2)
                    with self.assertRaisesRegex(RuntimeError, "changed version"):
                        await old_request
                    fresh = await client.verify(generation(), expected_verifier_version=2)
                    self.assertEqual(fresh.reward, 0.9)
                finally:
                    await server.close()
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
