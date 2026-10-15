import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from src.rvl_systems.verifier import FunctionalVerifier
from src.rvl_systems.verifier_deployment import VerifierDeploymentCoordinator, publish_verifier_artifact
from src.rvl_systems.verifier_rpc import TCPVerifierClient, VerifierWorkerServer
from src.rvl_systems.types import Generation

async def run(identical=False):
    entered, release = asyncio.Event(), asyncio.Event()
    events=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        a=publish_verifier_artifact(root/'a',version=1,content=b'{"reward":0.1}')
        b=a if identical else publish_verifier_artifact(root/'b',version=1,content=b'{"reward":0.9}')
        calls=0
        async def loader(m):
            nonlocal calls
            calls+=1
            if calls==1:
                events.append('A loader blocked')
                entered.set()
                await release.wait()
            r=json.loads(Path(m.path).read_text())['reward']
            return FunctionalVerifier(lambda _:r)
        state=root/'worker.json'
        server=VerifierWorkerServer(FunctionalVerifier(lambda _:0),worker_id='v',deployment_loader=loader,deployment_state_path=state)
        client=TCPVerifierClient(*(await server.start()),timeout_s=5)
        coordinator=VerifierDeploymentCoordinator([client],state_path=root/'coordinator.json')
        task=asyncio.create_task(client.prepare_verifier(asdict(a),coordinator_epoch=coordinator.epoch))
        try:
            await asyncio.wait_for(entered.wait(),5)
            await coordinator.prepare(b)
            events.append({'B ACK':b.sha256})
            release.set()
            events.append({'A ACK':await task})
            try:
                await coordinator.activate()
                success=True
                events.append('B coordinator declared activation success')
            except RuntimeError as e:
                success=False
                events.append({'activation_rejected':str(e)})
            health=await client.ping()
            # On rejected activation the bootstrap verifier is still active.
            row=await client.verify(Generation('p','prompt','response',0.,1,0.,{}),expected_verifier_version=health['verifier_version'])
            durable=json.loads(state.read_text())
            wrong_success=success and health['active_artifact_sha256']!=b.sha256
            try:
                await coordinator.deploy(b)
                recovery_error=None
            except RuntimeError as e:
                recovery_error=str(e)
            recovered=await client.ping()
            return {'identical':identical,'events':events,'expected_sha':b.sha256,'activation_success':success,'health':health,'reward':row.reward,'durable':durable,'wrong_artifact_success':wrong_success,'recovery_error':recovery_error,'recovery_sha':recovered['active_artifact_sha256'],'fixed_contract_pass':not wrong_success and recovered['active_artifact_sha256']==b.sha256 and (not identical or success)}
        finally:
            release.set()
            await asyncio.gather(task,return_exceptions=True)
            await server.close()

class SameEpochArtifactTests(unittest.TestCase):
    def test_conflicting_prepare_cannot_activate_wrong_artifact(self):
        row=asyncio.run(asyncio.wait_for(run(False),15))
        self.assertTrue(row["fixed_contract_pass"],row)
        self.assertFalse(row["activation_success"],row)

    def test_identical_overlapping_prepare_is_idempotent(self):
        row=asyncio.run(asyncio.wait_for(run(True),15))
        self.assertTrue(row["fixed_contract_pass"],row)
        self.assertTrue(row["activation_success"],row)

    def test_coordinator_rejects_wrong_ack_and_health_identity(self):
        class Client:
            def __init__(self, stage): self.stage=stage
            async def prepare_verifier(self,m,*,coordinator_epoch):
                self.m=m; self.epoch=coordinator_epoch
                return {"worker_id":"v","verifier_version":m["version"],"coordinator_epoch":coordinator_epoch,"artifact_sha256":"0"*64 if self.stage=="prepare" else m["sha256"]}
            async def activate_verifier(self,version,*,coordinator_epoch,artifact_sha256):
                return {"worker_id":"v","verifier_version":version,"coordinator_epoch":-1 if self.stage=="epoch" else coordinator_epoch,"artifact_sha256":"0"*64 if self.stage=="activate" else artifact_sha256}
            async def ping(self):
                return {"worker_id":"v","verifier_version":self.m["version"],"coordinator_epoch":self.epoch,"active_artifact_sha256":"0"*64 if self.stage=="health" else self.m["sha256"]}
        async def check(stage):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                m=publish_verifier_artifact(root/"artifact",version=1,content=b"artifact")
                c=VerifierDeploymentCoordinator([Client(stage)],state_path=root/"coordinator.json")
                with self.assertRaises(RuntimeError): await c.deploy(m)
                self.assertEqual(c.active_version,-1)
                self.assertIsNotNone(c.pending_manifest)
        for stage in ["prepare","activate","epoch","health"]:
            with self.subTest(stage=stage): asyncio.run(check(stage))

    def test_coordinator_serializes_complete_deployments(self):
        async def check():
            entered, release=asyncio.Event(),asyncio.Event()
            events=[]
            class Client:
                async def prepare_verifier(self,m,*,coordinator_epoch):
                    events.append(("prepare",m["version"]))
                    if m["version"]==1:
                        entered.set(); await release.wait()
                    self.m=m; self.epoch=coordinator_epoch
                    return {"worker_id":"v","verifier_version":m["version"],"coordinator_epoch":coordinator_epoch,"artifact_sha256":m["sha256"]}
                async def activate_verifier(self,version,*,coordinator_epoch,artifact_sha256):
                    events.append(("activate",version))
                    return {"worker_id":"v","verifier_version":version,"coordinator_epoch":coordinator_epoch,"artifact_sha256":artifact_sha256}
                async def ping(self):
                    return {"worker_id":"v","verifier_version":self.m["version"],"coordinator_epoch":self.epoch,"active_artifact_sha256":self.m["sha256"]}
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                c=VerifierDeploymentCoordinator([Client()],state_path=root/"coordinator.json")
                a=publish_verifier_artifact(root/"a",version=1,content=b"a")
                b=publish_verifier_artifact(root/"b",version=2,content=b"b")
                first=asyncio.create_task(c.deploy(a))
                await entered.wait()
                # Scheduling the second call before releasing the first creates overlap.
                second=asyncio.create_task(c.deploy(b))
                await asyncio.sleep(0)
                self.assertEqual(events,[("prepare",1)])
                release.set()
                self.assertEqual(await first,1)
                self.assertEqual(await second,2)
                self.assertEqual(events,[("prepare",1),("activate",1),("prepare",2),("activate",2)])
        asyncio.run(asyncio.wait_for(check(),15))

    def test_activation_requires_digest_including_idempotent_path(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                artifact = publish_verifier_artifact(root / "a", version=1, content=b"a")
                server = VerifierWorkerServer(
                    FunctionalVerifier(lambda _: 0), worker_id="v",
                    deployment_loader=lambda _: FunctionalVerifier(lambda _: 1),
                    deployment_state_path=root / "worker.json",
                )
                client = TCPVerifierClient(*(await server.start()))
                coordinator = VerifierDeploymentCoordinator([client], state_path=root / "coordinator.json")
                try:
                    await coordinator.deploy(artifact)
                    with self.assertRaisesRegex(RuntimeError, "digest"):
                        await client.activate_verifier(1, coordinator_epoch=coordinator.epoch, artifact_sha256="0" * 64)
                    with self.assertRaises(RuntimeError):
                        await client._management_rpc({"op": "activate_verifier", "verifier_version": 1, "coordinator_epoch": coordinator.epoch})
                    ack = await client.activate_verifier(1, coordinator_epoch=coordinator.epoch, artifact_sha256=artifact.sha256)
                    self.assertEqual(ack["artifact_sha256"], artifact.sha256)
                finally:
                    await server.close()
        asyncio.run(asyncio.wait_for(check(), 15))
