"""Reusable learned verifier head for verification-aware RL experiments.

Architecture and training match the prospectively locked Qwen learned-verifier
bridge: mean-pooled final-layer response hidden states followed by

    LayerNorm(hidden) -> Linear(hidden, 128) -> GELU -> Linear(128, 1).

Artifacts are stored as compressed NumPy arrays with JSON metadata and are loaded
with allow_pickle=False. The embedding application supplies the frozen causal-LM
backbone separately, so deployment never executes code from the artifact.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .types import Generation, VerifiedGeneration


ARTIFACT_FORMAT = "rvl.learned_verifier_head_v1"
ARCHITECTURE = "LayerNorm(hidden_size)->Linear(hidden_size,128)->GELU->Linear(128,1)"


@dataclass(frozen=True)
class LearnedVerifierHeadConfig:
    hidden_width: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    full_batch_steps: int = 300
    minimum_positive_labels: int = 8
    minimum_negative_labels: int = 8

    def __post_init__(self) -> None:
        if self.hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if self.full_batch_steps <= 0:
            raise ValueError("full_batch_steps must be positive")
        if self.minimum_positive_labels <= 0 or self.minimum_negative_labels <= 0:
            raise ValueError("minimum class counts must be positive")


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "learned_verifier_head_v1 requires torch; install requirements-systems.txt"
        ) from exc
    return torch


def _make_head(hidden_size: int, width: int):
    torch = _torch()
    return torch.nn.Sequential(
        torch.nn.LayerNorm(hidden_size),
        torch.nn.Linear(hidden_size, width),
        torch.nn.GELU(),
        torch.nn.Linear(width, 1),
    )


def _tokens(generation: Generation) -> tuple[list[int], list[int]]:
    prompt = generation.metadata.get("prompt_token_ids")
    response = generation.metadata.get("response_token_ids")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("learned verifier requires nonempty prompt_token_ids")
    if not isinstance(response, list) or not response:
        raise ValueError("learned verifier requires nonempty response_token_ids")
    if any(not isinstance(x, int) or x < 0 for x in prompt + response):
        raise ValueError("token IDs must be nonnegative integers")
    if generation.token_count != len(response):
        raise ValueError("response token count disagrees with generation metadata")
    return [int(x) for x in prompt], [int(x) for x in response]


def extract_response_feature(backbone: Any, generation: Generation) -> np.ndarray:
    """Extract one mean-pooled final-layer response representation."""
    torch = _torch()
    prompt, response = _tokens(generation)
    try:
        device = next(backbone.parameters()).device
    except StopIteration as exc:
        raise ValueError("backbone has no parameters") from exc
    was_training = bool(backbone.training)
    backbone.eval()
    try:
        with torch.inference_mode():
            ids = torch.tensor(
                [prompt + response],
                dtype=torch.long,
                device=device,
            )
            output = backbone(input_ids=ids, output_hidden_states=True)
            hidden_states = getattr(output, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError("backbone did not return hidden states")
            hidden = hidden_states[-1][0]
            start = len(prompt)
            stop = start + len(response)
            if hidden.shape[0] < stop:
                raise RuntimeError("backbone hidden sequence is shorter than token metadata")
            feature = hidden[start:stop].float().mean(dim=0)
            if not torch.isfinite(feature).all():
                raise FloatingPointError("non-finite learned-verifier representation")
            return feature.detach().cpu().numpy().astype(np.float32, copy=False)
    finally:
        if was_training:
            backbone.train()


def extract_feature_matrix(
    backbone: Any,
    generations: Iterable[Generation],
) -> np.ndarray:
    rows = [extract_response_feature(backbone, generation) for generation in generations]
    if not rows:
        raise ValueError("at least one generation is required")
    width = rows[0].shape
    if any(row.shape != width for row in rows):
        raise ValueError("backbone hidden dimensions changed across samples")
    matrix = np.stack(rows, axis=0)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("finite 2D learned-verifier feature matrix required")
    return matrix


def _trusted_labels(samples: Iterable[VerifiedGeneration]) -> tuple[list[VerifiedGeneration], np.ndarray]:
    rows = list(samples)
    if not rows:
        raise ValueError("trusted verifier-fit samples are empty")
    labels = np.asarray([float(row.reward) for row in rows], dtype=np.float32)
    if not np.isfinite(labels).all() or np.any(labels < 0) or np.any(labels > 1):
        raise ValueError("trusted verifier-fit labels must be finite values in [0,1]")
    return rows, labels


def train_head(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    config: LearnedVerifierHeadConfig | None = None,
):
    """Train the locked full-batch class-balanced MLP head on CPU."""
    config = config or LearnedVerifierHeadConfig()
    if (
        features.ndim != 2
        or labels.ndim != 1
        or features.shape[0] != labels.shape[0]
        or not np.isfinite(features).all()
        or not np.isfinite(labels).all()
    ):
        raise ValueError("finite aligned verifier features/labels required")
    positives = int(np.sum(labels > 0.5))
    negatives = int(np.sum(labels <= 0.5))
    if positives < config.minimum_positive_labels:
        raise ValueError(
            f"learned verifier needs >= {config.minimum_positive_labels} positive labels; got {positives}"
        )
    if negatives < config.minimum_negative_labels:
        raise ValueError(
            f"learned verifier needs >= {config.minimum_negative_labels} negative labels; got {negatives}"
        )

    torch = _torch()
    torch.manual_seed(int(seed))
    head = _make_head(int(features.shape[1]), config.hidden_width).cpu()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    losses: list[float] = []
    head.train()
    for _ in range(config.full_batch_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = head(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite learned-verifier training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    head.eval()
    report = {
        "architecture": ARCHITECTURE,
        "config": asdict(config),
        "seed": int(seed),
        "examples": int(len(labels)),
        "positives": positives,
        "negatives": negatives,
        "pos_weight": float(negatives / positives),
        "hidden_size": int(features.shape[1]),
        "parameters": int(sum(parameter.numel() for parameter in head.parameters())),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }
    return head, report


class LearnedVerifierHeadV1:
    """Frozen-backbone neural verifier compatible with the common Verifier protocol."""

    def __init__(
        self,
        backbone: Any,
        head: Any,
        *,
        model_identity: str,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> None:
        if not model_identity:
            raise ValueError("model_identity is required")
        self.backbone = backbone
        self.head = head.cpu().eval()
        self.model_identity = str(model_identity)
        self.artifact_metadata = dict(artifact_metadata or {})
        self._version = 0
        self._lock = asyncio.Lock()

    @property
    def version(self) -> int:
        return self._version

    def refresh(self) -> None:
        raise RuntimeError(
            "learned verifier head is immutable; deploy a newly trained artifact"
        )

    def _score_sync(self, generation: Generation) -> tuple[float, float]:
        torch = _torch()
        started = time.perf_counter()
        feature = extract_response_feature(self.backbone, generation)
        with torch.inference_mode():
            x = torch.tensor(feature[None, :], dtype=torch.float32)
            score = float(torch.sigmoid(self.head(x).squeeze()).item())
        if not np.isfinite(score) or score < 0.0 or score > 1.0:
            raise FloatingPointError("learned verifier emitted an invalid probability")
        return score, time.perf_counter() - started

    async def verify(self, generation: Generation) -> VerifiedGeneration:
        # A single backbone instance is intentionally serialized. Scale-out occurs
        # by deploying multiple verifier workers, not by concurrent mutation/use
        # of one model object.
        async with self._lock:
            score, elapsed = await asyncio.to_thread(self._score_sync, generation)
        return VerifiedGeneration(
            generation=generation,
            reward=score,
            verifier_latency_s=elapsed,
            verifier_version=self.version,
            metadata={
                "verifier": "learned_verifier_head_v1",
                "model_identity": self.model_identity,
                "artifact_format": ARTIFACT_FORMAT,
            },
        )


def fit_learned_verifier(
    backbone: Any,
    trusted_samples: Iterable[VerifiedGeneration],
    *,
    model_identity: str,
    seed: int,
    config: LearnedVerifierHeadConfig | None = None,
) -> tuple[LearnedVerifierHeadV1, dict[str, Any]]:
    rows, labels = _trusted_labels(trusted_samples)
    features = extract_feature_matrix(backbone, [row.generation for row in rows])
    head, report = train_head(features, labels, seed=seed, config=config)
    verifier = LearnedVerifierHeadV1(
        backbone,
        head,
        model_identity=model_identity,
        artifact_metadata={"training": report},
    )
    return verifier, report


def save_learned_verifier_artifact(
    path: str | Path,
    verifier: LearnedVerifierHeadV1,
    *,
    training_report: dict[str, Any],
) -> Path:
    """Save only numeric head tensors + JSON metadata; no executable pickle."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refuse to overwrite learned verifier artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = verifier.head.state_dict()
    arrays: dict[str, np.ndarray] = {
        f"param::{name}": tensor.detach().cpu().numpy()
        for name, tensor in state.items()
    }
    metadata = {
        "format": ARTIFACT_FORMAT,
        "architecture": ARCHITECTURE,
        "model_identity": verifier.model_identity,
        "training_report": training_report,
        "state_keys": list(state.keys()),
    }
    arrays["__metadata_json__"] = np.asarray(
        json.dumps(metadata, sort_keys=True), dtype=np.str_
    )
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return path


def load_learned_verifier_artifact(
    path: str | Path,
    *,
    backbone: Any,
    expected_model_identity: str,
) -> LearnedVerifierHeadV1:
    """Load a non-pickle head artifact and bind it to an explicit backbone."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if "__metadata_json__" not in archive:
            raise ValueError("learned verifier artifact metadata missing")
        metadata = json.loads(str(archive["__metadata_json__"].item()))
        if metadata.get("format") != ARTIFACT_FORMAT:
            raise ValueError("unknown learned verifier artifact format")
        if metadata.get("architecture") != ARCHITECTURE:
            raise ValueError("learned verifier architecture mismatch")
        if metadata.get("model_identity") != expected_model_identity:
            raise ValueError("learned verifier backbone identity mismatch")
        keys = [str(key) for key in metadata.get("state_keys") or []]
        if not keys:
            raise ValueError("learned verifier artifact state key list is empty")
        first_weight = np.asarray(archive[f"param::{keys[0]}"])
        # LayerNorm weight is the first state item and determines hidden size.
        if first_weight.ndim != 1 or first_weight.size == 0:
            raise ValueError("invalid learned verifier hidden dimension")
        hidden_size = int(first_weight.shape[0])
        training = dict(metadata.get("training_report") or {})
        config_raw = dict(training.get("config") or {})
        config = LearnedVerifierHeadConfig(**config_raw)
        head = _make_head(hidden_size, config.hidden_width)
        torch = _torch()
        expected = head.state_dict()
        if set(expected) != set(keys):
            raise ValueError("learned verifier state schema mismatch")
        loaded = {}
        for name, template in expected.items():
            key = f"param::{name}"
            if key not in archive:
                raise ValueError(f"learned verifier tensor missing: {name}")
            array = np.asarray(archive[key])
            tensor = torch.from_numpy(array.copy())
            if tuple(tensor.shape) != tuple(template.shape):
                raise ValueError(f"learned verifier tensor shape mismatch: {name}")
            loaded[name] = tensor.to(dtype=template.dtype)
        head.load_state_dict(loaded, strict=True)
        head.eval()
    return LearnedVerifierHeadV1(
        backbone,
        head,
        model_identity=expected_model_identity,
        artifact_metadata=metadata,
    )
