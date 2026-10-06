from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import canonical
from .models import Candidate, ChampionSnapshot, ImprovementProposal, PromotionDecision

IMMUTABLE_TABLES = ("events", "proposals", "candidates", "evaluations", "decisions", "champions", "lessons")


class ResearchMemory:
    """Append-only experiment memory with a mutable current-champion pointer only."""
    def __init__(self, path: str | Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL"); self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,ts REAL NOT NULL,kind TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS proposals(id TEXT PRIMARY KEY,generation INTEGER NOT NULL,parent_champion_id TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY,proposal_id TEXT NOT NULL,parent_champion_id TEXT NOT NULL,payload TEXT NOT NULL,FOREIGN KEY(proposal_id) REFERENCES proposals(id));
        CREATE TABLE IF NOT EXISTS evaluations(seq INTEGER PRIMARY KEY AUTOINCREMENT,candidate_id TEXT NOT NULL,split TEXT NOT NULL,suite_digest TEXT NOT NULL,payload TEXT NOT NULL,FOREIGN KEY(candidate_id) REFERENCES candidates(id));
        CREATE TABLE IF NOT EXISTS decisions(candidate_id TEXT PRIMARY KEY,accepted INTEGER NOT NULL,payload TEXT NOT NULL,FOREIGN KEY(candidate_id) REFERENCES candidates(id));
        CREATE TABLE IF NOT EXISTS champions(id TEXT PRIMARY KEY,generation INTEGER NOT NULL,parent_champion_id TEXT,policy_version INTEGER NOT NULL,verifier_version INTEGER NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS lessons(seq INTEGER PRIMARY KEY AUTOINCREMENT,generation INTEGER NOT NULL,candidate_id TEXT NOT NULL,lesson TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS controller_state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        """)
        for table in IMMUTABLE_TABLES:
            self.db.execute(f"CREATE TRIGGER IF NOT EXISTS no_update_{table} BEFORE UPDATE ON {table} BEGIN SELECT RAISE(ABORT, 'append-only table'); END")
            self.db.execute(f"CREATE TRIGGER IF NOT EXISTS no_delete_{table} BEFORE DELETE ON {table} BEGIN SELECT RAISE(ABORT, 'append-only table'); END")

    def close(self): self.db.close()
    def event(self, kind: str, **payload: Any): self.db.execute("INSERT INTO events(ts,kind,payload) VALUES(?,?,?)",(time.time(),kind,canonical(payload)))
    def record_proposal(self,generation,proposal):
        self.db.execute("INSERT INTO proposals(id,generation,parent_champion_id,payload) VALUES(?,?,?,?)",(proposal.id,generation,proposal.parent_champion_id,canonical(proposal.to_dict()))); self.event("proposal",generation=generation,proposal_id=proposal.id)
    def record_candidate(self,candidate):
        payload=asdict(candidate); payload["target_component"]=candidate.target_component.value
        self.db.execute("INSERT INTO candidates(id,proposal_id,parent_champion_id,payload) VALUES(?,?,?,?)",(candidate.candidate_id,candidate.proposal_id,candidate.parent_champion_id,canonical(payload))); self.event("candidate",candidate_id=candidate.candidate_id,proposal_id=candidate.proposal_id)
    def record_evaluation(self,candidate_id,split,suite_digest,payload): self.db.execute("INSERT INTO evaluations(candidate_id,split,suite_digest,payload) VALUES(?,?,?,?)",(candidate_id,split,suite_digest,canonical(payload)))
    def record_decision(self,decision):
        self.db.execute("INSERT INTO decisions(candidate_id,accepted,payload) VALUES(?,?,?)",(decision.candidate_id,int(decision.accepted),canonical(asdict(decision)))); self.event("promotion_decision",candidate_id=decision.candidate_id,accepted=decision.accepted,reasons=list(decision.reasons))
    def add_champion(self,champion,*,activate=True):
        self.db.execute("INSERT INTO champions(id,generation,parent_champion_id,policy_version,verifier_version,payload) VALUES(?,?,?,?,?,?)",(champion.champion_id,champion.generation,champion.parent_champion_id,champion.policy_version,champion.verifier_version,canonical(asdict(champion))))
        if activate: self.set_current_champion(champion.champion_id)
        self.event("champion_recorded",champion_id=champion.champion_id,generation=champion.generation,activate=activate)
    def set_current_champion(self,champion_id):
        if self.db.execute("SELECT 1 FROM champions WHERE id=?",(champion_id,)).fetchone() is None: raise KeyError(f"unknown champion: {champion_id}")
        self.db.execute("INSERT INTO controller_state(key,value) VALUES('current_champion',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(champion_id,))
    def rollback(self,champion_id,reason):
        current=self.current_champion(); target=self.get_champion(champion_id); self.set_current_champion(champion_id); self.event("rollback",from_champion=current.champion_id,to_champion=champion_id,reason=reason); return target
    def current_champion(self):
        row=self.db.execute("SELECT value FROM controller_state WHERE key='current_champion'").fetchone()
        if not row: raise RuntimeError("no active champion")
        return self.get_champion(row[0])
    def get_champion(self,champion_id):
        row=self.db.execute("SELECT payload FROM champions WHERE id=?",(champion_id,)).fetchone()
        if not row: raise KeyError(champion_id)
        return ChampionSnapshot(**json.loads(row[0]))
    def add_lesson(self,generation,candidate_id,lesson): self.db.execute("INSERT INTO lessons(generation,candidate_id,lesson) VALUES(?,?,?)",(generation,candidate_id,lesson))
    def commit_generation(self,decision,generation,candidate_id,lesson,champion=None):
        """Atomically commit the promotion decision, optional activation, and lesson."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.record_decision(decision)
            if champion is not None:
                self.add_champion(champion)
            self.add_lesson(generation,candidate_id,lesson)
            self.event(
                "generation_committed",
                generation=generation,
                candidate_id=candidate_id,
                accepted=decision.accepted,
                champion_id=champion.champion_id if champion is not None else None,
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
    def rejected_hypotheses(self):
        rows=self.db.execute("SELECT p.payload FROM proposals p JOIN decisions d ON d.candidate_id IN (SELECT id FROM candidates WHERE proposal_id=p.id) WHERE d.accepted=0 ORDER BY p.generation").fetchall()
        return [json.loads(r[0])["hypothesis"] for r in rows]
    def candidate_diffs(self): return [json.loads(r[0])["diff"] for r in self.db.execute("SELECT payload FROM candidates ORDER BY rowid").fetchall()]
    def recent_events(self,limit=100):
        rows=self.db.execute("SELECT seq,ts,kind,payload FROM events ORDER BY seq DESC LIMIT ?",(limit,)).fetchall()
        return [{"seq":s,"ts":ts,"kind":k,**json.loads(p)} for s,ts,k,p in reversed(rows)]
    def integrity_check(self): return str(self.db.execute("PRAGMA integrity_check").fetchone()[0])
