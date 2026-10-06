"""Durable grouped LM experiences with token-exact immutable behavior data."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict

from ..types import Generation, VerifiedGeneration
from .contracts import canonical, digest


class TokenReplay:
    """Durable behavior replay with a separate verification-admission stage.

    Behavior tokens/logprobs are immutable. Verification payloads may be
    refreshed, but a learner can only consume groups in ready state whose
    policy and verifier versions satisfy the configured lag bounds.
    """

    ACTIVE_STATUSES = ("pending_verification", "verifying", "ready")

    def __init__(self,path,capacity=16):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.db = sqlite3.connect(str(path),isolation_level=None,timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.capacity = capacity
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS groups(
          id TEXT PRIMARY KEY, policy_version INTEGER NOT NULL, payload TEXT NOT NULL,
          hash TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'ready',
          verifier_version INTEGER NOT NULL DEFAULT -1,
          generated_at REAL NOT NULL DEFAULT 0,
          verified_at REAL,
          verification_token TEXT,
          verification_lease_until REAL,
          verification_attempts INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS checkpoint(id INTEGER PRIMARY KEY CHECK(id=1), path TEXT NOT NULL, version INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS configuration(id INTEGER PRIMARY KEY CHECK(id=1), hash TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS verification_configuration(id INTEGER PRIMARY KEY CHECK(id=1), hash TEXT NOT NULL);
        """)
        self._migrate_schema()
        self._backfill_verifier_versions()

    def _migrate_schema(self):
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(groups)")}
        additions = {
            "verifier_version": "INTEGER NOT NULL DEFAULT -1",
            "generated_at": "REAL NOT NULL DEFAULT 0",
            "verified_at": "REAL",
            "verification_token": "TEXT",
            "verification_lease_until": "REAL",
            "verification_attempts": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, ddl in additions.items():
            if name not in columns:
                self.db.execute(f"ALTER TABLE groups ADD COLUMN {name} {ddl}")

    @staticmethod
    def _generation_semantic(generation):
        raw = asdict(generation) if isinstance(generation, Generation) else dict(generation)
        raw = json.loads(canonical(raw))
        raw.pop("latency_s", None)
        return raw

    @classmethod
    def _behavior_digest(cls, generations):
        return digest([cls._generation_semantic(g) for g in generations])

    @staticmethod
    def _decode_generations(raw):
        rows = json.loads(raw) if isinstance(raw, str) else raw
        result = []
        for entry in rows:
            payload = entry.get("generation", entry)
            result.append(Generation(**payload))
        return result

    @staticmethod
    def _decode_verified(raw):
        samples = []
        for entry in json.loads(raw):
            entry["generation"] = Generation(**entry["generation"])
            samples.append(VerifiedGeneration(**entry))
        return samples

    def _backfill_verifier_versions(self):
        rows = self.db.execute(
            "SELECT id,payload FROM groups WHERE verifier_version<0 AND status IN ('ready','consumed','stale')"
        ).fetchall()
        for rid, raw in rows:
            try:
                payload = json.loads(raw)
                versions = {int(entry["verifier_version"]) for entry in payload if "verifier_version" in entry}
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if len(versions) == 1:
                self.db.execute(
                    "UPDATE groups SET verifier_version=? WHERE id=?",
                    (versions.pop(), rid),
                )

    def bind(self,config):
        checksum = digest(config)
        row = self.db.execute("SELECT hash FROM configuration WHERE id=1").fetchone()
        if row and row[0] != checksum:
            raise ValueError("LM resume configuration or prompt set mismatch")
        self.db.execute("INSERT OR IGNORE INTO configuration VALUES(1,?)",(checksum,))

    def bind_verification(self,config):
        checksum = digest(config)
        row = self.db.execute("SELECT hash FROM verification_configuration WHERE id=1").fetchone()
        if row and row[0] != checksum:
            raise ValueError("verification pipeline resume configuration mismatch")
        self.db.execute("INSERT OR IGNORE INTO verification_configuration VALUES(1,?)",(checksum,))

    def _active_count(self):
        return int(self.db.execute(
            "SELECT COUNT(*) FROM groups WHERE status IN ('pending_verification','verifying','ready')"
        ).fetchone()[0])

    def active_count(self):
        return self._active_count()

    def put_pending(self,rid,version,generations,*,now=None):
        if not generations:
            raise ValueError("empty behavior group")
        now = time.time() if now is None else now
        raw = [asdict(g) for g in generations]
        checksum = self._behavior_digest(generations)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute("SELECT hash FROM groups WHERE id=?",(rid,)).fetchone()
            if existing:
                if existing[0] != checksum:
                    raise ValueError("token replay ID collision")
                self.db.execute("COMMIT")
                return True
            if self._active_count() >= self.capacity:
                self.db.execute("COMMIT")
                return False
            self.db.execute(
                """INSERT INTO groups(
                    id,policy_version,payload,hash,status,verifier_version,generated_at
                ) VALUES(?,?,?,?, 'pending_verification', -1, ?)""",
                (rid,version,canonical(raw),checksum,now),
            )
            self.db.execute("COMMIT")
            return True
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def put(self,rid,version,samples,*,now=None):
        """Compatibility path for callers that already produced verified samples."""
        if not samples:
            raise ValueError("empty verified group")
        now = time.time() if now is None else now
        raw = [asdict(s) for s in samples]
        semantic = []
        for s in raw:
            d = json.loads(canonical(s))
            d["generation"].pop("latency_s")
            d.pop("verifier_latency_s")
            semantic.append(d)
        checksum = digest(semantic)
        versions = {int(s.verifier_version) for s in samples}
        if len(versions) != 1:
            raise ValueError("verified group mixes verifier versions")
        verifier_version = versions.pop()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute("SELECT hash FROM groups WHERE id=?",(rid,)).fetchone()
            if existing:
                if existing[0] != checksum:
                    raise ValueError("token replay ID collision")
                self.db.execute("COMMIT")
                return True
            if self._active_count() >= self.capacity:
                self.db.execute("COMMIT")
                return False
            self.db.execute(
                """INSERT INTO groups(
                    id,policy_version,payload,hash,status,verifier_version,generated_at,verified_at
                ) VALUES(?,?,?,?, 'ready', ?, ?, ?)""",
                (rid,version,canonical(raw),checksum,verifier_version,now,now),
            )
            self.db.execute("COMMIT")
            return True
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def _requeue_stale_verifications_unlocked(self,current_verifier_version,max_verifier_lag):
        threshold = current_verifier_version-max_verifier_lag
        rows = self.db.execute(
            "SELECT id,payload FROM groups WHERE status='ready' AND verifier_version<? ORDER BY rowid",
            (threshold,),
        ).fetchall()
        for rid, raw in rows:
            generations = self._decode_generations(raw)
            self.db.execute(
                """UPDATE groups SET payload=?,status='pending_verification',verifier_version=-1,
                   verified_at=NULL,verification_token=NULL,verification_lease_until=NULL
                   WHERE id=? AND status='ready'""",
                (canonical([asdict(g) for g in generations]),rid),
            )
        return len(rows)

    def requeue_stale_verifications(self,current_verifier_version,max_verifier_lag):
        if current_verifier_version < 0 or max_verifier_lag < 0:
            raise ValueError("invalid verifier lag bound")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            count = self._requeue_stale_verifications_unlocked(
                current_verifier_version,max_verifier_lag
            )
            self.db.execute("COMMIT")
            return count
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def claim_verification(self,current_version,max_policy_lag,current_verifier_version,
                           max_verifier_lag=0,lease_s=30,now=None):
        if min(current_version,current_verifier_version,max_policy_lag,max_verifier_lag) < 0 or lease_s <= 0:
            raise ValueError("invalid verification lease limits")
        now = time.time() if now is None else now
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                """UPDATE groups SET status='pending_verification',verification_token=NULL,
                   verification_lease_until=NULL
                   WHERE status='verifying' AND verification_lease_until<=?""",
                (now,),
            )
            self.db.execute(
                """UPDATE groups SET status='stale',verification_token=NULL,verification_lease_until=NULL
                   WHERE status IN ('pending_verification','ready')
                   AND policy_version<?""",
                (current_version-max_policy_lag,),
            )
            self._requeue_stale_verifications_unlocked(
                current_verifier_version,max_verifier_lag
            )
            row = self.db.execute(
                """SELECT id,policy_version,payload FROM groups
                   WHERE status='pending_verification' AND policy_version<=?
                   ORDER BY rowid LIMIT 1""",
                (current_version,),
            ).fetchone()
            if not row:
                self.db.execute("COMMIT")
                return None
            rid,version,raw = row
            token = uuid.uuid4().hex
            cur = self.db.execute(
                """UPDATE groups SET status='verifying',verification_token=?,
                   verification_lease_until=?,verification_attempts=verification_attempts+1
                   WHERE id=? AND status='pending_verification'""",
                (token,now+lease_s,rid),
            )
            if cur.rowcount != 1:
                raise RuntimeError("verification claim lost")
            generations = self._decode_generations(raw)
            self.db.execute("COMMIT")
            return token,rid,version,generations
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def complete_verification(self,rid,token,samples,*,now=None):
        if not samples:
            raise ValueError("empty verification result")
        now = time.time() if now is None else now
        versions = {int(s.verifier_version) for s in samples}
        if len(versions) != 1:
            raise ValueError("verification result mixes verifier versions")
        verifier_version = versions.pop()
        row = self.db.execute(
            """SELECT payload FROM groups WHERE id=? AND status='verifying'
               AND verification_token=? AND verification_lease_until>?""",
            (rid,token,now),
        ).fetchone()
        if not row:
            raise ValueError("verification lease lost")
        original = self._decode_generations(row[0])
        if self._behavior_digest(original) != self._behavior_digest([s.generation for s in samples]):
            raise ValueError("verification changed immutable behavior data")
        cur = self.db.execute(
            """UPDATE groups SET payload=?,status='ready',verifier_version=?,verified_at=?,
               verification_token=NULL,verification_lease_until=NULL
               WHERE id=? AND status='verifying' AND verification_token=?""",
            (canonical([asdict(s) for s in samples]),verifier_version,now,rid,token),
        )
        if cur.rowcount != 1:
            raise ValueError("verification lease lost")

    def retry_verification(self,rid,token,*,penalize=False):
        row = self.db.execute(
            "SELECT verification_attempts FROM groups WHERE id=? AND status='verifying' AND verification_token=?",
            (rid,token),
        ).fetchone()
        if not row:
            raise ValueError("verification lease lost")
        attempts = int(row[0]) if penalize else max(0,int(row[0])-1)
        self.db.execute(
            """UPDATE groups SET status='pending_verification',verification_token=NULL,
               verification_lease_until=NULL,verification_attempts=? WHERE id=?""",
            (attempts,rid),
        )

    def fail_verification(self,rid,token,*,max_attempts=3):
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        row = self.db.execute(
            "SELECT verification_attempts FROM groups WHERE id=? AND status='verifying' AND verification_token=?",
            (rid,token),
        ).fetchone()
        if not row:
            raise ValueError("verification lease lost")
        status = "quarantined" if int(row[0]) >= max_attempts else "pending_verification"
        self.db.execute(
            """UPDATE groups SET status=?,verification_token=NULL,verification_lease_until=NULL
               WHERE id=? AND status='verifying' AND verification_token=?""",
            (status,rid,token),
        )
        return status

    def rewrite_verified(self,rid,samples,*,now=None):
        if not samples:
            raise ValueError("empty verified group")
        now = time.time() if now is None else now
        verifier_version = min(int(s.verifier_version) for s in samples)
        cur = self.db.execute(
            """UPDATE groups SET payload=?,verifier_version=?,verified_at=?
               WHERE id=? AND status='ready'""",
            (canonical([asdict(s) for s in samples]),verifier_version,now,rid),
        )
        if cur.rowcount != 1:
            raise ValueError("verified group is not ready")

    def next(self,current_version,max_lag,current_verifier_version=None,max_verifier_lag=0):
        if current_version < 0 or max_lag < 0 or max_verifier_lag < 0:
            raise ValueError("invalid replay lag bound")
        self.db.execute(
            "UPDATE groups SET status='stale' WHERE status='ready' AND policy_version<?",
            (current_version-max_lag,),
        )
        params = [current_version]
        clause = ""
        if current_verifier_version is not None:
            if current_verifier_version < 0:
                raise ValueError("invalid verifier version")
            self.requeue_stale_verifications(current_verifier_version,max_verifier_lag)
            clause = " AND verifier_version>=?"
            params.append(current_verifier_version-max_verifier_lag)
        row = self.db.execute(
            f"""SELECT id,policy_version,payload FROM groups
                WHERE status='ready' AND policy_version<=?{clause}
                ORDER BY rowid LIMIT 1""",
            tuple(params),
        ).fetchone()
        if not row:
            return None
        rid,version,raw = row
        return rid,version,self._decode_verified(raw)

    def verification_metadata(self,rid):
        row = self.db.execute(
            """SELECT policy_version,verifier_version,generated_at,verified_at,
               verification_attempts,status FROM groups WHERE id=?""",
            (rid,),
        ).fetchone()
        if not row:
            raise KeyError(rid)
        return {
            "policy_version":int(row[0]),
            "verifier_version":int(row[1]),
            "generated_at":float(row[2]),
            "verified_at":None if row[3] is None else float(row[3]),
            "verification_attempts":int(row[4]),
            "status":row[5],
        }

    def verification_backlog(self,current_verifier_version,max_verifier_lag=0):
        threshold = current_verifier_version-max_verifier_lag
        return int(self.db.execute(
            """SELECT COUNT(*) FROM groups WHERE status IN ('pending_verification','verifying')
               OR (status='ready' AND verifier_version<?)""",
            (threshold,),
        ).fetchone()[0])

    def commit(self,rid,path,version):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cur = self.db.execute("UPDATE groups SET status='consumed' WHERE id=? AND status='ready'",(rid,))
            if cur.rowcount != 1:
                raise ValueError("experience already consumed")
            self.db.execute("INSERT INTO checkpoint VALUES(1,?,?) ON CONFLICT(id) DO UPDATE SET path=excluded.path,version=excluded.version",
                            (str(path),version))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def checkpoint(self):
        return self.db.execute("SELECT path,version FROM checkpoint WHERE id=1").fetchone()

    def counts(self):
        return dict(self.db.execute("SELECT status,COUNT(*) FROM groups GROUP BY status"))

    def close(self):
        self.db.close()
