from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Any

from .models import FailureCategory, FailureCluster, FailureEvidence


@dataclass(frozen=True)
class TaskObservation:
    task_id: str
    success: bool
    reward: float
    trusted: float
    verifier_score: float
    latency_s: float
    timeout: bool = False
    tool_error: bool = False
    selected_wrong_tool: bool = False
    context_overflow: bool = False
    stale_policy_age: int = 0
    stale_verifier_age: int = 0
    metadata: Mapping[str, Any] | None = None


class FailureAnalyzer:
    """Evidence-based failure clustering; model explanations are never authoritative."""

    def analyze(self, rows: Iterable[TaskObservation]) -> list[FailureCluster]:
        grouped: dict[FailureCategory, list[tuple[TaskObservation, FailureEvidence]]] = defaultdict(list)
        rows = list(rows)
        for row in rows:
            if row.timeout:
                grouped[FailureCategory.TIMEOUT].append((row, FailureEvidence("runtime", "deadline exceeded", row.latency_s)))
            if row.tool_error:
                grouped[FailureCategory.INFRASTRUCTURE_FAILURE].append((row, FailureEvidence("tool_runtime", "tool execution error")))
            if row.selected_wrong_tool:
                grouped[FailureCategory.TOOL_SELECTION_ERROR].append((row, FailureEvidence("tool_trace", "selected tool did not match task requirement")))
            if row.context_overflow:
                grouped[FailureCategory.CONTEXT_FAILURE].append((row, FailureEvidence("harness", "context budget exhausted")))
            gap = row.reward - row.trusted
            if gap >= 0.20:
                grouped[FailureCategory.REWARD_HACKING_CANDIDATE].append((row, FailureEvidence("independent_eval", "reward exceeds trusted correctness", gap)))
            if row.verifier_score >= 0.7 and row.trusted < 0.5:
                grouped[FailureCategory.VERIFICATION_FALSE_POSITIVE].append((row, FailureEvidence("trusted_label", "verifier accepted trusted failure", row.verifier_score - row.trusted)))
            if row.verifier_score < 0.5 and row.trusted >= 0.8:
                grouped[FailureCategory.VERIFICATION_FALSE_NEGATIVE].append((row, FailureEvidence("trusted_label", "verifier rejected trusted success", row.trusted - row.verifier_score)))
            if max(row.stale_policy_age, row.stale_verifier_age) >= 2 and abs(row.verifier_score - row.trusted) >= 0.10:
                grouped[FailureCategory.POLICY_VERIFIER_DISTRIBUTION_SHIFT].append(
                    (row, FailureEvidence("versioning", "disagreement co-occurs with verifier/policy staleness", max(row.stale_policy_age, row.stale_verifier_age)))
                )
            if not row.success and not any([
                row.timeout, row.tool_error, row.selected_wrong_tool, row.context_overflow,
                gap >= 0.20, row.verifier_score >= 0.7 and row.trusted < 0.5,
            ]):
                grouped[FailureCategory.REASONING_ERROR].append((row, FailureEvidence("trusted_eval", "incorrect without a more specific observed mechanism", row.trusted)))

        clusters: list[FailureCluster] = []
        total = max(1, len(rows))
        for category, items in grouped.items():
            evidence = tuple(ev for _, ev in items[:8])
            severity = min(1.0, len(items) / total + 0.25 * max((abs(float(ev.value)) for _, ev in items if isinstance(ev.value, (int, float))), default=0.0))
            clusters.append(
                FailureCluster(
                    category=category,
                    count=len(items),
                    severity=severity,
                    evidence=evidence,
                    task_ids=tuple(sorted({row.task_id for row, _ in items})[:32]),
                )
            )
        clusters.sort(key=lambda c: (-c.severity, -c.count, c.category.value))
        return clusters
