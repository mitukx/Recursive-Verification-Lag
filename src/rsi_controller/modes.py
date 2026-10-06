from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .models import RSIMode


class RSIExperimentBackend(Protocol):
    mode: RSIMode

    def capability(self) -> dict[str, Any]: ...


@dataclass
class HarnessRSIBackend:
    """Frozen-weight mode used by the reference controller."""

    mode: RSIMode = RSIMode.HARNESS

    def capability(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "weight_updates": False,
            "mutable_components": ["H", "selected V composition/config"],
            "validated": True,
        }


@dataclass
class AdapterRSIBackend:
    """Capability probe/config builder for parameter-efficient adaptation.

    The repository has no native PEFT dependency, so this backend fails closed
    unless `peft` is installed by an explicit experiment environment. It does not
    silently install packages or download models.
    """

    adapter_rank: int = 8
    alpha: int = 16
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    mode: RSIMode = RSIMode.ADAPTER

    def capability(self) -> dict[str, Any]:
        try:
            import peft  # noqa: F401
            available = True
        except ImportError:
            available = False
        return {
            "mode": self.mode.value,
            "weight_updates": True,
            "parameter_efficient": True,
            "peft_available": available,
            "validated": False,
        }

    def build_lora_config(self):
        try:
            from peft import LoraConfig
        except ImportError as exc:
            raise RuntimeError("Adapter RSI requires an explicitly installed PEFT environment") from exc
        return LoraConfig(
            r=self.adapter_rank,
            lora_alpha=self.alpha,
            target_modules=list(self.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )


@dataclass
class RLRsiBackend:
    """Bridge to the repository's existing RVL-aware asynchronous GRPO loop.

    Callers provide already-constructed trusted backend/verifier objects and
    prompts. This wrapper adds no networking, credentials, or deployment step.
    Promotion remains a separate RSI-controller responsibility.
    """

    mode: RSIMode = RSIMode.RL

    def capability(self) -> dict[str, Any]:
        try:
            from src.rvl_systems.lab.lm_runtime import AsyncHFLab  # noqa: F401
            available = True
        except ImportError:
            available = False
        return {
            "mode": self.mode.value,
            "weight_updates": True,
            "uses_existing_grpo": True,
            "uses_existing_rvl_interventions": True,
            "available": available,
            "validated_by_rsi_controller": False,
        }

    async def run_candidate(
        self,
        *,
        root: str | Path,
        backend: Any,
        verifier: Any,
        prompts: Mapping[str, str],
        samples: int,
        seed: int,
        learning_rate: float,
        max_policy_lag: int = 4,
        capacity: int = 4,
        control: Any = None,
    ) -> dict[str, Any]:
        from src.rvl_systems.lab.lm_runtime import AsyncHFLab

        lab = AsyncHFLab(
            root,
            backend,
            verifier,
            learning_rate=learning_rate,
            max_policy_lag=max_policy_lag,
            capacity=capacity,
            control=control,
        )
        try:
            return await lab.run(dict(prompts), samples=samples, seed=seed)
        finally:
            lab.close()
