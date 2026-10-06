from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .contracts import Trajectory, Verdict, canonical, digest


class LeaseLost(RuntimeError):
    pass


class ReplayStore:
    """WAL replay with idempotent insertion, expiring leases and fenced commits.

    SQLite supports independent actor/learner processes on one machine. It is
    NOT a multi-node database: replace this interface with a network store there.
    """

    def __init__(self, path: str | Path, capacity: int = 4096):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS replay(
          id TEXT PRIMARY KEY, payload TEXT NOT NULL, hash TEXT NOT NULL,
          policy_version INTEGER NOT NULL, verdict TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'ready', lease_token TEXT,
          lease_until REAL, attempts INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS labels(
          id TEXT PRIMARY KEY REFERENCES replay(id), reward REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS events(
          seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, payload TEXT NOT NULL);
        """)

    def close(self):
        self.db.close()

    def event(self, kind, **data):
        self.db.execute("INSERT INTO events(kind,payload) VALUES(?,?)", (kind, canonical(data)))

    def events(self):
        return [(kind, json.loads(payload)) for kind, payload in
                self.db.execute("SELECT kind,payload FROM events ORDER BY seq")]

    def put(self, trajectory: Trajectory, verdict: Verdict) -> bool:
        raw = trajectory.to_dict()
        # Timing is measured evidence, not part of idempotency identity.
        identity = dict(raw)
        identity.pop("elapsed_s")
        checksum = digest(identity)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute("SELECT hash FROM replay WHERE id=?",
                                       (trajectory.trajectory_id,)).fetchone()
            if existing:
                if existing[0] != checksum:
                    raise ValueError("trajectory ID collision")
                self.db.execute("COMMIT")
                return False
            active = self.db.execute("SELECT COUNT(*) FROM replay WHERE status IN ('ready','leased')").fetchone()[0]
            if active >= self.capacity:
                self.db.execute("COMMIT")
                return False
            self.db.execute("INSERT INTO replay(id,payload,hash,policy_version,verdict) VALUES(?,?,?,?,?)",
                            (trajectory.trajectory_id, canonical(raw), checksum,
                             trajectory.policy_version, canonical(asdict(verdict))))
            self.event("insert", id=trajectory.trajectory_id, policy_version=trajectory.policy_version)
            self.db.execute("COMMIT")
            return True
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def claim(self, limit: int, current_version: int, max_lag: int,
              lease_s: float = 30, now: float | None = None):
        if limit <= 0 or max_lag < 0 or lease_s <= 0:
            raise ValueError("invalid lease limits")
        now = time.time() if now is None else now
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("UPDATE replay SET status='ready',lease_token=NULL WHERE status='leased' AND lease_until<=?", (now,))
            self.db.execute("UPDATE replay SET status='stale' WHERE status='ready' AND policy_version<?",
                            (current_version - max_lag,))
            rows = self.db.execute(
                "SELECT id,payload,verdict FROM replay WHERE status='ready' AND policy_version<=? ORDER BY rowid LIMIT ?",
                (current_version, limit)).fetchall()
            result = []
            for rid, payload, verdict in rows:
                token = uuid.uuid4().hex
                self.db.execute("UPDATE replay SET status='leased',lease_token=?,lease_until=?,attempts=attempts+1 WHERE id=?",
                                (token, now + lease_s, rid))
                data = json.loads(verdict)
                data["scores"] = tuple(data["scores"])
                result.append((token, Trajectory.from_dict(json.loads(payload)), Verdict(**data)))
            self.db.execute("COMMIT")
            return result
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def finish(self, rid, token, *, status="consumed", now=None):
        if status not in ("consumed", "ready", "quarantined"):
            raise ValueError("invalid terminal status")
        now = time.time() if now is None else now
        cur = self.db.execute(
            "UPDATE replay SET status=?,lease_token=NULL WHERE id=? AND status='leased' AND lease_token=? AND lease_until>?",
            (status, rid, token, now))
        if cur.rowcount != 1:
            raise LeaseLost(rid)

    def relabel(self, rid: str, verdict: Verdict):
        # Reward version changes; behavior policy/logprobs NEVER change.
        self.db.execute("UPDATE replay SET verdict=? WHERE id=?", (canonical(asdict(verdict)), rid))

    def trusted_label(self, rid, reward):
        if not 0 <= reward <= 1:
            raise ValueError("invalid trusted reward")
        self.db.execute("INSERT INTO labels(id,reward) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET reward=excluded.reward", (rid, reward))

    def labeled(self):
        return [(Trajectory.from_dict(json.loads(p)), float(y)) for p, y in
                self.db.execute("SELECT r.payload,l.reward FROM replay r JOIN labels l ON r.id=l.id ORDER BY r.rowid")]

    def candidates(self, limit=1024):
        return [(Trajectory.from_dict(json.loads(p)), Verdict(**{**json.loads(v), "scores": tuple(json.loads(v)["scores"])}))
                for p, v in self.db.execute(
                    "SELECT payload,verdict FROM replay WHERE status IN ('ready','stale') ORDER BY rowid DESC LIMIT ?", (limit,))]

    def counts(self):
        return dict(self.db.execute("SELECT status,COUNT(*) FROM replay GROUP BY status"))

    def save_state(self, key, payload):
        self.db.execute("INSERT INTO state(key,payload) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",
                        (key, canonical(payload)))

    def load_state(self, key):
        row = self.db.execute("SELECT payload FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None
