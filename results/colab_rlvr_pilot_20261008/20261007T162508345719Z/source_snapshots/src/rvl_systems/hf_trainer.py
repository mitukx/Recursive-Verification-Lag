from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from .grpo import compute_group_advantages
from .triton_grpo import grpo_surrogate, validate_objective_backend
from .types import VerifiedGeneration


@dataclass(frozen=True)
class HFTTrainerConfig:
    learning_rate: float = 1e-6
    clip_eps: float = 0.2
    max_grad_norm: float = 1.0
    advantage_eps: float = 1e-6
    clip_advantage: float = 5.0
    disable_dropout: bool = True
    objective_backend: str = "torch"


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
        validate_objective_backend(self.config.objective_backend)
        if self.config.disable_dropout:
            for module in self.model.modules():
                if isinstance(module,torch.nn.Dropout):
                    module.p = 0.0
                if isinstance(getattr(module,"attention_dropout",None),(float,int)):
                    module.attention_dropout = 0.0
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
        with torch.autograd.profiler.record_function("rvl.grpo.model_forward"):
            logits = self.model(input_ids=sequence).logits[0]
        start = len(prompt_ids) - 1
        end = start + len(response_ids)
        response_logits = logits[start:end]
        temperature = float(sample.generation.metadata.get("sampling_temperature", 1.0))
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("training requires a finite positive sampling temperature; greedy scores are not behavior probabilities")
        targets = torch.tensor(response_ids, dtype=torch.long, device=device)
        with torch.autograd.profiler.record_function("rvl.grpo.logprob_objective"):
            current_logps = torch.log_softmax(response_logits.float() / temperature, dim=-1).gather(
                1, targets.unsqueeze(1)
            ).squeeze(1)
        if not torch.isfinite(current_logps).all():
            raise FloatingPointError("current token log-probabilities contain non-finite values")
        old = torch.tensor(
            old_logps,
            dtype=current_logps.dtype,
            device=device,
        )
        advantages = torch.full_like(
            current_logps,
            float(advantage),
        )
        surrogate, clip_mask, abs_log_ratio, behavior_kl = grpo_surrogate(
            current_logps,
            old,
            advantages,
            clip_eps=self.config.clip_eps,
            backend=self.config.objective_backend,
        )
        return (
            surrogate.mean(),
            clip_mask.float().mean(),
            abs_log_ratio.max(),
            behavior_kl.mean(),
        )

    def snapshot_training_state(self) -> dict[str, Any]:
        """Deep CPU snapshot of model/optimizer/RNG for transactional updates."""
        torch = self.torch
        model = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        optimizer = self.optimizer.state_dict()
        # copy.deepcopy preserves optimizer tensors and nested scalar state
        import copy
        optimizer = copy.deepcopy(optimizer)
        state = {
            "model": model,
            "optimizer": optimizer,
            "cpu_rng": torch.get_rng_state().cpu().clone(),
        }
        if torch.cuda.is_available():
            state["cuda_rng"] = [x.cpu().clone() for x in torch.cuda.get_rng_state_all()]
        return state

    def restore_training_state(self, state: dict[str, Any]) -> None:
        """Restore model/optimizer/RNG after a rejected candidate update."""
        torch = self.torch
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        if "cuda_rng" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])

    def train_step(self, samples: list[VerifiedGeneration], *, advantages: list[float] | None = None) -> dict[str, float]:
        if not samples:
            return {"loss": 0.0, "mean_reward": 0.0, "samples": 0.0}
        with self.torch.autograd.profiler.record_function("rvl.grpo.advantages"):
            records = compute_group_advantages(
                samples,
                eps=self.config.advantage_eps,
                clip=self.config.clip_advantage,
            )
        if len(records) != len(samples):
            raise RuntimeError("advantage/sample cardinality mismatch")

        # Training mode permits activation checkpointing; dropout was disabled
        # explicitly so behavior ratios do not include random dropout masks.
        self.model.train()
        with self.torch.autograd.profiler.record_function("rvl.grpo.zero_grad"):
            self.optimizer.zero_grad(set_to_none=True)
        if advantages is None:
            advantages = [record.advantage for record in records]
        if len(advantages) != len(samples) or not all(math.isfinite(x) for x in advantages):
            raise ValueError("invalid supplied advantages")
        losses, clip_fractions, max_log_ratios, kl_estimates = [], [], [], []
        # Keep one response graph at a time, not a graph for the whole batch.
        for index, (sample, advantage) in enumerate(zip(samples, advantages)):
            sync = (index == len(samples)-1)
            context = nullcontext() if sync or not hasattr(self.model, "no_sync") else self.model.no_sync()
            with context:
                with self.torch.autograd.profiler.record_function("rvl.grpo.sample_forward_backward"):
                    objective, fraction, ratio, kl = self._sample_objective(sample, advantage)
                    loss = -objective / len(samples)
                    if not self.torch.isfinite(loss):
                        raise FloatingPointError("GRPO loss is non-finite")
                    with self.torch.autograd.profiler.record_function("rvl.grpo.backward"):
                        loss.backward()
            losses.append(float(loss.detach().cpu()))
            clip_fractions.append(float(fraction.detach().cpu()))
            max_log_ratios.append(float(ratio.detach().cpu()))
            kl_estimates.append(float(kl.detach().cpu()))
        with self.torch.autograd.profiler.record_function("rvl.grpo.grad_clip"):
            if hasattr(self.model, "clip_grad_norm_"):
                grad_norm = self.model.clip_grad_norm_(self.config.max_grad_norm)
            else:
                grad_norm = self.torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
        if not self.torch.isfinite(self.torch.as_tensor(grad_norm)):
            raise FloatingPointError("non-finite gradient norm")
        with self.torch.autograd.profiler.record_function("rvl.grpo.optimizer_step"):
            self.optimizer.step()
        return {
            "loss": sum(losses),
            "mean_reward": fmean(sample.reward for sample in samples),
            "samples": float(len(samples)),
            "grad_norm": float(grad_norm.detach().cpu())
            if hasattr(grad_norm, "detach")
            else float(grad_norm),
            "clip_fraction": fmean(clip_fractions),
            "max_abs_log_ratio": max(max_log_ratios),
            "behavior_kl_estimate": fmean(kl_estimates),
        }
