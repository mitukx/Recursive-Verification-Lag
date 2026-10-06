from __future__ import annotations

from dataclasses import dataclass

from .backends import InferenceBackend
from .rollout import RolloutRequest
from .telemetry import Telemetry
from .types import Generation
from .worker_pool import RetryPolicy, VersionedWorkerPool, with_retries


@dataclass
class VersionedRemoteBackend:
    """Wrap a remote inference backend with retries and policy-version checks."""

    backend: InferenceBackend
    pool: VersionedWorkerPool
    retry: RetryPolicy = RetryPolicy()
    telemetry: Telemetry | None = None

    def __post_init__(self) -> None:
        self.telemetry = self.telemetry or Telemetry()

    async def generate(
        self,
        prompt_id: str,
        prompt: str,
        *,
        n: int,
        temperature: float,
        seed: int,
    ) -> list[Generation]:
        expected_version = self.pool.policy_version

        async def call() -> list[Generation]:
            generations = await self.backend.generate(
                prompt_id,
                prompt,
                n=n,
                temperature=temperature,
                seed=seed,
            )
            tagged: list[Generation] = []
            for generation in generations:
                metadata = dict(generation.metadata)
                metadata["policy_version"] = expected_version
                tagged.append(
                    Generation(
                        prompt_id=generation.prompt_id,
                        prompt=generation.prompt,
                        response=generation.response,
                        logprob=generation.logprob,
                        token_count=generation.token_count,
                        latency_s=generation.latency_s,
                        metadata=metadata,
                    )
                )
            return tagged

        return await with_retries(
            call,
            policy=self.retry,
            telemetry=self.telemetry,
            metric_prefix="remote_rollout",
        )

    def validate(self, generations: list[Generation]) -> None:
        for generation in generations:
            version = generation.metadata.get("policy_version")
            if version is None:
                raise RuntimeError("rollout result is missing policy_version")
            self.pool.validate_result_version(int(version))
