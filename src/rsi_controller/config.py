from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .models import RSIMode, ResourceLimits


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromotionThresholds:
    min_promotion_gain: float = 0.005
    min_sealed_gain: float = 0.0
    min_trusted_gain: float = 0.0
    max_verification_gap: float = 0.08
    max_failure_rate_increase: float = 0.03
    max_latency_p95_ratio: float = 1.25
    min_verifier_agreement: float = 0.8
    confidence_z: float = 1.64


@dataclass(frozen=True)
class VerifierTrustThresholds:
    uncertain_disagreement: float = 0.10
    low_disagreement: float = 0.20
    uncertain_reward_truth_gap: float = 0.08
    low_reward_truth_gap: float = 0.15
    uncertain_distribution_shift: float = 0.15
    low_distribution_shift: float = 0.30
    uncertain_stale_age: int = 2
    low_stale_age: int = 4


@dataclass(frozen=True)
class MutationPolicy:
    allowed_harness_keys: tuple[str, ...] = (
        "reasoning_budget", "retry_limit", "context_window", "memory_slots",
        "tool_policy", "sampling_temperature", "curriculum_level", "format_guard",
    )
    allowed_training_keys: tuple[str, ...] = (
        "learning_rate", "batch_size", "adapter_rank", "clip_ratio",
    )
    allowed_verifier_keys: tuple[str, ...] = (
        "threshold", "ensemble_size", "refresh_cadence", "trusted_fraction",
    )
    code_patch_allowlist: tuple[str, ...] = ("src/rsi_controller/mutable_harness/",)
    allow_code_patches: bool = False


@dataclass(frozen=True)
class RSIConfig:
    mode: RSIMode = RSIMode.HARNESS
    seed: int = 17
    generations: int = 4
    output_dir: str = "artifacts/rsi-baseline"
    promotion: PromotionThresholds = field(default_factory=PromotionThresholds)
    verifier_trust: VerifierTrustThresholds = field(default_factory=VerifierTrustThresholds)
    mutation: MutationPolicy = field(default_factory=MutationPolicy)
    resources: ResourceLimits = field(default_factory=ResourceLimits)


def _tuple_fields(cls, raw: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(raw)
    for name, field_info in cls.__dataclass_fields__.items():
        if name in result and str(field_info.type).startswith("tuple"):
            result[name] = tuple(result[name])
    return result


def load_config(path: str | Path) -> RSIConfig:
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError("config must be JSON-compatible YAML unless PyYAML is installed") from exc
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("RSI config root must be a mapping")
    return RSIConfig(
        mode=RSIMode(raw.get("mode", "harness")),
        seed=int(raw.get("seed", 17)),
        generations=int(raw.get("generations", 4)),
        output_dir=str(raw.get("output_dir", "artifacts/rsi-baseline")),
        promotion=PromotionThresholds(**raw.get("promotion", {})),
        verifier_trust=VerifierTrustThresholds(**raw.get("verifier_trust", {})),
        mutation=MutationPolicy(**_tuple_fields(MutationPolicy, raw.get("mutation", {}))),
        resources=ResourceLimits(**raw.get("resources", {})),
    )
