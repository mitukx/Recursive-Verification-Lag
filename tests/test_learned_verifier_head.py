import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.rvl_systems.learned_verifier_head import (
    ARTIFACT_FORMAT,
    LearnedVerifierHeadConfig,
    fit_learned_verifier,
    load_learned_verifier_artifact,
    save_learned_verifier_artifact,
)
from src.rvl_systems.types import Generation, VerifiedGeneration
from src.rvl_systems.verifier_deployment import (
    VerifierDeploymentCoordinator,
    publish_verifier_artifact,
)
from src.rvl_systems.verifier_rpc import TCPVerifierClient, VerifierWorkerServer
from src.rvl_systems.verifier import FunctionalVerifier


class TinyBackbone(torch.nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, hidden)
        with torch.no_grad():
            values = torch.arange(64 * hidden, dtype=torch.float32).reshape(64, hidden)
            self.embedding.weight.copy_((values % 17) / 17.0)

    def forward(self, input_ids, output_hidden_states=False):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(hidden_states=(hidden,) if output_hidden_states else None)


def generation(i):
    response = [3 + (i % 8), 20 + (i % 8)]
    return Generation(
        f"p-{i // 4}",
        "prompt",
        "response",
        -0.5,
        len(response),
        0.0,
        {
            "prompt_token_ids": [1, 2],
            "response_token_ids": response,
            "response_token_logprobs": [-0.2, -0.3],
        },
    )


def trusted_samples():
    rows = []
    for i in range(16):
        rows.append(
            VerifiedGeneration(
                generation(i),
                float(i % 2),
                0.0,
                0,
                {"trusted": True},
            )
        )
    return rows


class LearnedVerifierHeadTests(unittest.TestCase):
    def test_fit_verify_save_reload_without_pickle(self):
        async def run():
            backbone = TinyBackbone()
            config = LearnedVerifierHeadConfig(
                hidden_width=8,
                learning_rate=1e-2,
                full_batch_steps=20,
                minimum_positive_labels=2,
                minimum_negative_labels=2,
            )
            verifier, report = fit_learned_verifier(
                backbone,
                trusted_samples(),
                model_identity="tiny@deadbeef",
                seed=17,
                config=config,
            )
            before = await verifier.verify(generation(3))
            self.assertGreaterEqual(before.reward, 0.0)
            self.assertLessEqual(before.reward, 1.0)
            self.assertEqual(before.metadata["artifact_format"], ARTIFACT_FORMAT)

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "head.npz"
                save_learned_verifier_artifact(
                    path, verifier, training_report=report
                )
                raw = path.read_bytes()
                self.assertFalse(raw.startswith(b"\x80"), "artifact must not be pickle")
                restored = load_learned_verifier_artifact(
                    path,
                    backbone=TinyBackbone(),
                    expected_model_identity="tiny@deadbeef",
                )
                after = await restored.verify(generation(3))
                self.assertAlmostEqual(before.reward, after.reward, places=6)

                with self.assertRaisesRegex(ValueError, "backbone identity"):
                    load_learned_verifier_artifact(
                        path,
                        backbone=TinyBackbone(),
                        expected_model_identity="tiny@other",
                    )
        asyncio.run(run())

    def test_training_requires_both_classes(self):
        backbone = TinyBackbone()
        rows = [
            VerifiedGeneration(generation(i), 1.0, 0.0, 0)
            for i in range(8)
        ]
        with self.assertRaisesRegex(ValueError, "negative"):
            fit_learned_verifier(
                backbone,
                rows,
                model_identity="tiny@deadbeef",
                seed=17,
                config=LearnedVerifierHeadConfig(
                    hidden_width=8,
                    full_batch_steps=2,
                    minimum_positive_labels=2,
                    minimum_negative_labels=2,
                ),
            )

    def test_artifact_deploys_through_two_phase_verifier_control_plane(self):
        async def run():
            backbone = TinyBackbone()
            config = LearnedVerifierHeadConfig(
                hidden_width=8,
                learning_rate=1e-2,
                full_batch_steps=10,
                minimum_positive_labels=2,
                minimum_negative_labels=2,
            )
            verifier, report = fit_learned_verifier(
                backbone,
                trusted_samples(),
                model_identity="tiny@deadbeef",
                seed=29,
                config=config,
            )
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "learned.npz"
                save_learned_verifier_artifact(
                    source, verifier, training_report=report
                )
                manifest = publish_verifier_artifact(
                    root / "artifacts",
                    version=1,
                    content=source.read_bytes(),
                    suffix=".npz",
                )

                def loader(deployment_manifest):
                    return load_learned_verifier_artifact(
                        deployment_manifest.path,
                        backbone=TinyBackbone(),
                        expected_model_identity="tiny@deadbeef",
                    )

                server = VerifierWorkerServer(
                    FunctionalVerifier(lambda _: 0.0),
                    worker_id="learned-v0",
                    deployment_loader=loader,
                    deployment_state_path=root / "worker.json",
                )
                address = await server.start()
                client = TCPVerifierClient(*address)
                coordinator = VerifierDeploymentCoordinator(
                    [client], state_path=root / "coordinator.json"
                )
                try:
                    self.assertEqual(await coordinator.deploy(manifest), 1)
                    result = await client.verify(
                        generation(5), expected_verifier_version=1
                    )
                    self.assertEqual(result.verifier_version, 1)
                    self.assertEqual(
                        result.metadata["verifier"], "learned_verifier_head_v1"
                    )
                    self.assertGreaterEqual(result.reward, 0.0)
                    self.assertLessEqual(result.reward, 1.0)
                finally:
                    await server.close()
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
