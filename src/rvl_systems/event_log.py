from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ControlPlaneEvent:
    seq: int
    kind: str
    request_id: str
    workload_id: str
    worker: str | None = None
    attempt: int | None = None
    data: dict[str, object] | None = None


class ControlPlaneEventLog:
    """Deterministic semantic event log for scheduler replay/debugging."""

    def __init__(self) -> None:
        self.events: list[ControlPlaneEvent] = []

    def append(
        self,
        kind: str,
        *,
        request_id: str,
        workload_id: str,
        worker: str | None = None,
        attempt: int | None = None,
        **data: object,
    ) -> None:
        self.events.append(
            ControlPlaneEvent(
                seq=len(self.events),
                kind=kind,
                request_id=request_id,
                workload_id=workload_id,
                worker=worker,
                attempt=attempt,
                data=data or None,
            )
        )

    def semantic_digest(self) -> str:
        """Hash semantics independent of cross-request interleaving."""
        grouped: dict[str, list[dict]] = {}
        for event in self.events:
            raw = asdict(event)
            raw.pop("seq", None)
            grouped.setdefault(event.request_id, []).append(raw)
        payload = [
            {"request_id": request_id, "events": grouped[request_id]}
            for request_id in sorted(grouped)
        ]
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate(self) -> None:
        starts: set[str] = set()
        terminals: set[str] = set()
        selected: dict[str, set[str]] = {}

        for index, event in enumerate(self.events):
            if event.seq != index:
                raise ValueError("event sequence is not contiguous")
            rid = event.request_id
            if event.kind == "request_started":
                if rid in starts:
                    raise ValueError(f"duplicate request_started for {rid}")
                starts.add(rid)
            elif rid not in starts:
                raise ValueError(f"event before request_started for {rid}")

            if rid in terminals:
                raise ValueError(f"event after terminal state for {rid}")

            if event.kind == "worker_selected":
                if event.worker is None:
                    raise ValueError("worker_selected requires worker")
                used = selected.setdefault(rid, set())
                if event.worker in used:
                    raise ValueError(
                        f"worker {event.worker} reused for request {rid}"
                    )
                used.add(event.worker)

            if event.kind in {"request_completed", "request_failed"}:
                terminals.add(rid)

        missing = starts - terminals
        if missing:
            raise ValueError(
                "requests missing terminal event: "
                + ",".join(sorted(missing))
            )

    def summary(self) -> dict[str, object]:
        self.validate()
        by_kind: dict[str, int] = {}
        attempts: dict[str, int] = {}
        outcomes: dict[str, str] = {}
        for event in self.events:
            by_kind[event.kind] = by_kind.get(event.kind, 0) + 1
            if event.kind == "worker_selected":
                attempts[event.request_id] = attempts.get(event.request_id, 0) + 1
            if event.kind == "request_completed":
                outcomes[event.request_id] = "completed"
            elif event.kind == "request_failed":
                outcomes[event.request_id] = "failed"
        return {
            "events": len(self.events),
            "requests": len(outcomes),
            "event_counts": by_kind,
            "attempts_by_request": attempts,
            "outcomes": outcomes,
            "semantic_sha256": self.semantic_digest(),
        }

    def write_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(
                json.dumps(asdict(event), sort_keys=True) + "\n"
                for event in self.events
            ),
            encoding="utf-8",
        )

    @classmethod
    def read_jsonl(cls, path: str | Path) -> "ControlPlaneEventLog":
        log = cls()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            log.events.append(ControlPlaneEvent(**json.loads(line)))
        return log
