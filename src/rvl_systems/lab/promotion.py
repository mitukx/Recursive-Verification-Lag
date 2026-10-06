from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import canonical, digest


@dataclass(frozen=True)
class EvalSample:
    """One frozen held-out comparison between incumbent and candidate."""

    task_id: str
    family: int
    incumbent_reward: float
    candidate_reward: float

    def __post_init__(self):
        for value in (self.incumbent_reward, self.candidate_reward):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("evaluation rewards must be finite probabilities")


@dataclass(frozen=True)
class PromotionPolicy:
    """Fail-closed deterministic deployment gate.

    This is a regression gate, not a claim of statistical significance. The
    frozen paired task/seed suite is intended to make each candidate comparable
    to the exact currently-serving incumbent.
    """

    min_samples: int = 48
    min_mean_delta: float = 0.0
    max_family_regression: float = 0.10
    max_uncompensated_new_failures: int = 2

    def __post_init__(self):
        if self.min_samples <= 0 or self.max_uncompensated_new_failures < 0:
            raise ValueError("invalid promotion sample/failure limits")
        if not -1.0 <= self.min_mean_delta <= 1.0:
            raise ValueError("invalid mean delta threshold")
        if not 0.0 <= self.max_family_regression <= 1.0:
            raise ValueError("invalid family regression threshold")


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    incumbent_version: int
    candidate_version: int
    incumbent_mean: float
    candidate_mean: float
    mean_delta: float
    family_deltas: dict[str, float]
    wins: int
    losses: int
    ties: int
    uncompensated_new_failures: int
    reasons: tuple[str, ...]
    evidence_sha256: str

    def to_dict(self):
        return asdict(self)


def evaluate_promotion(
    samples: list[EvalSample],
    *,
    incumbent_version: int,
    candidate_version: int,
    policy: PromotionPolicy,
) -> PromotionDecision:
    if incumbent_version < 0 or candidate_version < 0:
        raise ValueError("invalid policy versions")
    if len(samples) < policy.min_samples:
        raise ValueError("insufficient held-out promotion samples")
    if len({sample.task_id for sample in samples}) != len(samples):
        raise ValueError("duplicate held-out task IDs")

    incumbent_mean = sum(s.incumbent_reward for s in samples) / len(samples)
    candidate_mean = sum(s.candidate_reward for s in samples) / len(samples)
    mean_delta = candidate_mean - incumbent_mean

    grouped: dict[int, list[EvalSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.family, []).append(sample)
    family_deltas = {}
    for family, rows in sorted(grouped.items()):
        baseline = sum(x.incumbent_reward for x in rows) / len(rows)
        candidate = sum(x.candidate_reward for x in rows) / len(rows)
        family_deltas[str(family)] = candidate - baseline

    wins = sum(s.candidate_reward > s.incumbent_reward for s in samples)
    losses = sum(s.candidate_reward < s.incumbent_reward for s in samples)
    ties = len(samples) - wins - losses
    uncompensated = max(0, losses - wins)

    reasons: list[str] = []
    if candidate_version <= incumbent_version:
        reasons.append("candidate_version_not_newer")
    if mean_delta < policy.min_mean_delta:
        reasons.append("mean_reward_regression")
    if family_deltas and min(family_deltas.values()) < -policy.max_family_regression:
        reasons.append("family_slice_regression")
    if uncompensated > policy.max_uncompensated_new_failures:
        reasons.append("too_many_uncompensated_new_failures")

    evidence = {
        "incumbent_version": incumbent_version,
        "candidate_version": candidate_version,
        "policy": asdict(policy),
        "samples": [asdict(sample) for sample in samples],
        "summary": {
            "incumbent_mean": incumbent_mean,
            "candidate_mean": candidate_mean,
            "mean_delta": mean_delta,
            "family_deltas": family_deltas,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "uncompensated_new_failures": uncompensated,
            "reasons": reasons,
        },
    }
    return PromotionDecision(
        accepted=not reasons,
        incumbent_version=incumbent_version,
        candidate_version=candidate_version,
        incumbent_mean=incumbent_mean,
        candidate_mean=candidate_mean,
        mean_delta=mean_delta,
        family_deltas=family_deltas,
        wins=wins,
        losses=losses,
        ties=ties,
        uncompensated_new_failures=uncompensated,
        reasons=tuple(reasons),
        evidence_sha256=digest(evidence),
    )


class PromotionLedger:
    """Append-only hash-chained promotion history.

    The chain is tamper-evident, not a cryptographic signature. The single lab
    driver owns writes; every restart verifies the full chain before training.
    """

    GENESIS = "0" * 64

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq, self.head = self.verify()

    def verify(self) -> tuple[int, str]:
        if not self.path.exists():
            return 0, self.GENESIS
        previous = self.GENESIS
        expected_seq = 1
        with self.path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                record = json.loads(raw)
                if record.get("seq") != expected_seq:
                    raise ValueError("promotion ledger sequence mismatch")
                if record.get("prev_sha256") != previous:
                    raise ValueError("promotion ledger chain mismatch")
                core = {
                    "seq": record["seq"],
                    "prev_sha256": record["prev_sha256"],
                    "decision": record["decision"],
                }
                if record.get("record_sha256") != digest(core):
                    raise ValueError("promotion ledger checksum mismatch")
                previous = record["record_sha256"]
                expected_seq += 1
        return expected_seq - 1, previous

    def append(self, decision: PromotionDecision) -> str:
        seq = self.seq + 1
        core = {
            "seq": seq,
            "prev_sha256": self.head,
            "decision": decision.to_dict(),
        }
        record = {**core, "record_sha256": digest(core)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.seq = seq
        self.head = record["record_sha256"]
        return self.head
