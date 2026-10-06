"""Durable grouped LM experiences with token-exact immutable behavior data."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict

from ..types import Generation, VerifiedGeneration
from .contracts import canonical, digest


class TokenReplay:
    def __init__(self,path,capacity=16):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.db = sqlite3.connect(str(path),isolation_level=None,timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.capacity = capacity
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS groups(
          id TEXT PRIMARY KEY, policy_version INTEGER NOT NULL, payload TEXT NOT NULL,
          hash TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'ready');
        CREATE TABLE IF NOT EXISTS checkpoint(id INTEGER PRIMARY KEY CHECK(id=1), path TEXT NOT NULL, version INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS configuration(id INTEGER PRIMARY KEY CHECK(id=1), hash TEXT NOT NULL);
        """)

    def bind(self,config):
        checksum = digest(config)
        row = self.db.execute("SELECT hash FROM configuration WHERE id=1").fetchone()
        if row and row[0] != checksum:
            raise ValueError("LM resume configuration or prompt set mismatch")
        self.db.execute("INSERT OR IGNORE INTO configuration VALUES(1,?)",(checksum,))

    def put(self,rid,version,samples):
        raw = [asdict(s) for s in samples]
        semantic = []
        for s in raw:
            d = json.loads(canonical(s))
            d["generation"].pop("latency_s")
            d.pop("verifier_latency_s")
            semantic.append(d)
        checksum = digest(semantic)
        existing = self.db.execute("SELECT hash FROM groups WHERE id=?",(rid,)).fetchone()
        if existing:
            if existing[0] != checksum:
                raise ValueError("token replay ID collision")
            return True
        if self.db.execute("SELECT COUNT(*) FROM groups WHERE status='ready'").fetchone()[0] >= self.capacity:
            return False
        self.db.execute("INSERT INTO groups(id,policy_version,payload,hash) VALUES(?,?,?,?)",
                        (rid,version,canonical(raw),checksum))
        return True

    def next(self,current_version,max_lag):
        self.db.execute("UPDATE groups SET status='stale' WHERE status='ready' AND policy_version<?",
                        (current_version-max_lag,))
        row = self.db.execute("SELECT id,policy_version,payload FROM groups WHERE status='ready' AND policy_version<=? ORDER BY rowid LIMIT 1",
                              (current_version,)).fetchone()
        if not row:
            return None
        rid,version,raw = row
        samples = []
        for s in json.loads(raw):
            s["generation"] = Generation(**s["generation"])
            samples.append(VerifiedGeneration(**s))
        return rid,version,samples

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

    def close(self):
        self.db.close()
