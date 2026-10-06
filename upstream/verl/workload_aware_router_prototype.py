"""Prototype for verl workload-aware rollout routing.

This mirrors the current verl RequestLoadBalancer contract closely enough to
validate accounting and scheduling policy before porting into an upstream fork.

It is deliberately Ray-free so the policy can be tested and benchmarked on CPU.
"""
from __future__ import annotations

import random
from typing import Any

from cachetools import LRUCache


DEFAULT_ROUTING_CACHE_SIZE = 10_000


class WorkloadAwareRequestLoadBalancer:
    """Sticky routing with outstanding predicted token-work balancing.

    Unlike least-inflight routing, each admitted request contributes an
    estimated amount of work. The estimate uses prompt length plus the requested
    generation budget. release_server() removes exactly the work recorded at
    acquire time, so callers do not resend large prompt payloads on release.

    This is a prototype for verl upstream; it does not claim GPU performance.
    """

    def __init__(
        self,
        servers: dict[str, Any],
        max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE,
        *,
        prefill_weight: float = 1.0,
        decode_weight: float = 1.0,
        default_decode_tokens: int = 1,
        full_determinism: bool = False,
    ):
        if prefill_weight < 0 or decode_weight < 0:
            raise ValueError("work weights must be non-negative")
        if default_decode_tokens < 0:
            raise ValueError("default_decode_tokens must be non-negative")
        self._servers = dict(servers)
        self._inflight_requests = {sid: 0 for sid in servers}
        self._outstanding_work = {sid: 0.0 for sid in servers}
        self._request_id_to_server = LRUCache(maxsize=max_cache_size)
        # In-flight accounting must never be evicted by the sticky-session LRU.
        self._request_work: dict[str, tuple[str, float]] = {}
        self._prefill_weight = float(prefill_weight)
        self._decode_weight = float(decode_weight)
        self._default_decode_tokens = int(default_decode_tokens)
        self._full_determinism = bool(full_determinism)

    def require_acquire_fields(self) -> list[str]:
        return ["prompt_tokens", "decode_budget"]

    def require_release_fields(self) -> list[str]:
        return ["request_id"]

    def _estimate_work(
        self,
        prompt_tokens: int | None,
        decode_budget: int | None,
    ) -> float:
        try:
            prompt_tokens = max(0, int(prompt_tokens or 0))
        except (TypeError, ValueError):
            prompt_tokens = 0
        if decode_budget is None:
            decode_budget = self._default_decode_tokens
        try:
            decode_budget = max(0, int(decode_budget))
        except (TypeError, ValueError):
            decode_budget = self._default_decode_tokens
        return (
            self._prefill_weight * prompt_tokens
            + self._decode_weight * decode_budget
        )

    def acquire_server(
        self,
        request_id: str,
        prompt_tokens: int | None = None,
        decode_budget: int | None = None,
    ) -> tuple[str, Any]:
        if not self._inflight_requests:
            raise RuntimeError("No available servers in load balancer")

        work = self._estimate_work(prompt_tokens, decode_budget)

        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            if server_id not in self._inflight_requests:
                del self._request_id_to_server[request_id]
            else:
                self._record_acquire(request_id, server_id, work)
                return server_id, self._servers[server_id]

        if self._full_determinism:
            server_ids = list(self._servers)
            server_id = server_ids[hash(request_id) % len(server_ids)]
        else:
            min_work = min(self._outstanding_work.values())
            work_candidates = [
                sid for sid, value in self._outstanding_work.items()
                if value == min_work
            ]
            min_inflight = min(self._inflight_requests[sid] for sid in work_candidates)
            candidates = [
                sid for sid in work_candidates
                if self._inflight_requests[sid] == min_inflight
            ]
            server_id = random.choice(candidates)

        self._request_id_to_server[request_id] = server_id
        self._record_acquire(request_id, server_id, work)
        return server_id, self._servers[server_id]

    def _record_acquire(self, request_id: str, server_id: str, work: float) -> None:
        # Concurrent duplicate request IDs are not a supported sticky-session
        # pattern in verl. Fail closed instead of corrupting release accounting.
        if request_id in self._request_work:
            raise RuntimeError(f"request_id already in flight: {request_id}")
        self._request_work[request_id] = (server_id, work)
        self._inflight_requests[server_id] += 1
        self._outstanding_work[server_id] += work

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        if server_id not in self._inflight_requests:
            return
        if request_id is None:
            raise ValueError("workload-aware release requires request_id")
        record = self._request_work.get(request_id)
        if record is None:
            return
        recorded_server, work = record
        if recorded_server != server_id:
            raise ValueError(
                f"release server mismatch for {request_id}: "
                f"{server_id} != {recorded_server}"
            )
        del self._request_work[request_id]
        if self._inflight_requests[server_id] > 0:
            self._inflight_requests[server_id] -= 1
        self._outstanding_work[server_id] = max(
            0.0, self._outstanding_work[server_id] - work
        )

    def add_servers(self, servers: dict[str, Any]) -> None:
        for sid, handle in servers.items():
            self._servers[sid] = handle
            self._inflight_requests[sid] = 0
            self._outstanding_work[sid] = 0.0

    def remove_servers(self, server_ids: list[str]) -> None:
        doomed = set(server_ids)
        for sid in doomed:
            self._servers.pop(sid, None)
            self._inflight_requests.pop(sid, None)
            self._outstanding_work.pop(sid, None)
        for request_id, sid in list(self._request_id_to_server.items()):
            if sid in doomed:
                del self._request_id_to_server[request_id]
        # In-flight work on a removed server cannot be reassigned safely by the
        # router; drop local accounting because future release is ignored just
        # like verl's current default balancer does for removed servers.
        for request_id, (sid, _) in list(self._request_work.items()):
            if sid in doomed:
                del self._request_work[request_id]

    def get_all_servers(self) -> list[str]:
        return list(self._servers)

    def clear_sticky_cache(self) -> dict:
        cleared = len(self._request_id_to_server)
        self._request_id_to_server.clear()
        return {
            "cleared_entries": cleared,
            "server_loads": dict(self._inflight_requests),
            "server_work": dict(self._outstanding_work),
        }

    def get_total_inflight(self) -> int:
        return sum(self._inflight_requests.values())

    def get_status(self) -> dict:
        return {
            "servers": dict(self._inflight_requests),
            "server_work": dict(self._outstanding_work),
            "total_inflight": self.get_total_inflight(),
            "total_predicted_work": sum(self._outstanding_work.values()),
            "active_servers": len(self._servers),
        }
