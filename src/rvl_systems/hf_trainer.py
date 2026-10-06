from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from .grpo import compute_group_advantages
from .types import VerifiedGeneration


@dataclass(frozen=True)
class HFTTrainerConfig:
    learning_rate: float = 1e-6
    clip_eps: float = 0.2
    max_grad_norm: float = 1.0
    advantage_eps: float = 1e-6
    clip_advantage: float = 5.0


class HFCausalLMGRPOTrainer:
    """Minimal token-level clipped GRPO trainer for a local causal LM."""

    def __init__(self, model: Any, *, config: HFTTrainerConfig | None = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "HFCausalLMGRPOTrainer requires optional systems dependencies"
            ) from exc
        self.torch = torch
        self.model = model
        self.config = config or HFTTrainerConfig()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

    @staticmethod
    def _metadata(sample: VerifiedGeneration) -> tuple[list[int], list[int], list[float]]:
        meta = sample.generation.metadata
        try:
            prompt_ids = [int(x) for x in meta["prompt_token_ids"]]
            response_ids = [int(x) for x in meta["response_token_ids"]]
            old_logps = [float(x) for x in meta["response_token_logprobs"]]
        except KeyError as exc:
            raise ValueError(
                "generation lacks token metadata; use HFLocalBackend for local training"
            ) from exc
        if not prompt_ids or not response_ids:
            raise ValueError("prompt and response token sequences must be non-empty")
        if len(response_ids) != len(old_logps):
            raise ValueError("response token ids/logprobs length mismatch")
        if not all(math.isfinite(x) for x in old_logps):
            raise FloatingPointError("old token log-probabilities contain non-finite values")
        return prompt_ids, response_ids, old_logps

    def _sample_objective(
        self,
        sample: VerifiedGeneration,
        advantage: float,
    ):
        torch = self.torch
        prompt_ids, response_ids, old_logps = self._metadata(sample)
        device = next(self.model.parameters()).device
        sequence = torch.tensor(
            [prompt_ids + response_ids],
            dtype=torch.long,
            device=device,
        )
        logits = self.model(input_ids=sequence).logits[0]
        start = len(prompt_ids) - 1
        end = start + len(response_ids)
        response_logits = logits[start:end]
        targets = torch.tensor(response_ids, dtype=torch.long, device=device)
        current_logps = torch.log_softmax(response_logits.float(), dim=-1).gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        if not torch.isfinite(current_logps).all():
            raise FloatingPointError("current token log-probabilities contain non-finite values")
        old = torch.tensor(old_logps, dtype=current_logps.dtype, device=device)
        log_ratio = torch.clamp(current_logps - old, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(
            ratio,
            1.0 - self.config.clip_eps,
            1.0 + self.config.clip_eps,
        )
        adv = torch.tensor(float(advantage), dtype=current_logps.dtype, device=device)
        surrogate = torch.minimum(ratio * adv, clipped * adv)
        clip_fraction = ((ratio < 1.0 - self.config.clip_eps) | (ratio > 1.0 + self.config.clip_eps)).float().mean()
        return surrogate.mean(), clip_fraction, log_ratio.abs().max()

    def train_step(self, samples: list[VerifiedGeneration]) -> dict[str, float]:
        if not samples:
            return {"loss": 0.0, "mean_reward": 0.0, "samples": 0.0}
        records = compute_group_advantages(
            samples,
            eps=self.config.advantage_eps,
            clip=self.config.clip_advantage,
        )
        if len(records) != len(samples):
            raise RuntimeError("advantage/sample cardinality mismatch")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        rows = [
            self._sample_objective(sample, record.advantage)
            for sample, record in zip(samples, records)
        ]
        objectives = [row[0] for row in rows]
        clip_fractions = [row[1] for row in rows]
        max_log_ratios = [row[2] for row in rows]
        loss = -self.torch.stack(objectives).mean()
        if not self.torch.isfinite(loss):
            raise FloatingPointError("GRPO loss is non-finite")
        loss.backward()
        grad_norm = self.torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "mean_reward": fmean(sample.reward for sample in samples),
            "samples": float(len(samples)),
            "grad_norm": float(grad_norm.detach().cpu())
            if hasattr(grad_norm, "detach")
            else float(grad_norm),
            "clip_fraction": float(self.torch.stack(clip_fractions).mean().detach().cpu()),
            "max_abs_log_ratio": float(self.torch.stack(max_log_ratios).max().detach().cpu()),
        }
