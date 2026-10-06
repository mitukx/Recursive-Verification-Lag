from __future__ import annotations

import asyncio
from dataclasses import dataclass
from statistics import fmean

from .backends import ToyTabularBackend
from .grpo import GRPOConfig, train_toy_policy
from .refresh import AdaptiveRefreshController
from .rollout import AsyncRolloutEngine, RolloutRequest
from .telemetry import Telemetry
from .types import VerifiedGeneration
from .verifier import Verifier


@dataclass(frozen=True)
class PipelineConfig:
    samples_per_prompt: int = 8
    temperature: float = 1.0
    rounds: int = 8
    base_seed: int = 1234


class RLVRPipeline:
    """End-to-end rollout -> verify -> GRPO -> refresh loop."""

    def __init__(
        self,
        *,
        backend: ToyTabularBackend,
        verifier: Verifier,
        refresh: AdaptiveRefreshController,
        grpo: GRPOConfig | None = None,
        telemetry: Telemetry | None = None,
        max_concurrency: int = 8,
    ) -> None:
        self.backend = backend
        self.verifier = verifier
        self.refresh = refresh
        self.grpo = grpo or GRPOConfig()
        self.telemetry = telemetry or Telemetry()
        self.rollouts = AsyncRolloutEngine(
            backend, max_concurrency=max_concurrency, telemetry=self.telemetry
        )

    async def _verify_all(self, generations) -> list[VerifiedGeneration]:
        return list(await asyncio.gather(*(self.verifier.verify(g) for g in generations)))

    @staticmethod
    def _movement(before: dict[str, list[float]], after: dict[str, list[float]]) -> float:
        """Mean L1 change in softmax distributions across prompt groups."""
        if not before:
            return 0.0
        distances = []
        for key in before:
            p, q = before[key], after[key]
            distances.append(sum(abs(a - b) for a, b in zip(p, q)))
        return fmean(distances) if distances else 0.0

    async def run(
        self,
        prompts: dict[str, str],
        config: PipelineConfig | None = None,
    ) -> list[dict[str, float | int | str]]:
        cfg = config or PipelineConfig()
        history: list[dict[str, float | int | str]] = []
        for round_idx in range(cfg.rounds):
            before = {k: self.backend.distribution(k)[:] for k in prompts}
            requests = [
                RolloutRequest(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    samples=cfg.samples_per_prompt,
                    temperature=cfg.temperature,
                    seed=cfg.base_seed + round_idx * 10_000 + i,
                )
                for i, (prompt_id, prompt) in enumerate(prompts.items())
            ]
            generations = await self.rollouts.run(requests)
            verified = await self._verify_all(generations)
            train_metrics = train_toy_policy(self.backend, verified, self.grpo)
            after = {k: self.backend.distribution(k)[:] for k in prompts}
            movement = self._movement(before, after)
            should_refresh, reason = self.refresh.observe(movement)
            if should_refresh:
                self.verifier.refresh()
            reward = fmean(x.reward for x in verified) if verified else 0.0
            self.telemetry.observe("train.reward", reward)
            self.telemetry.observe("train.policy_movement_l1", movement)
            history.append(
                {
                    "round": round_idx,
                    "reward": reward,
                    "movement_l1": movement,
                    "refresh": int(should_refresh),
                    "refresh_reason": reason,
                    "verifier_version": self.verifier.version,
                    "updates": int(train_metrics["updates"]),
                }
            )
        return history
