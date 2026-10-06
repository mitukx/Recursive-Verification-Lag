from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.rsi_controller.candidates import CandidateGenerator, MutationRejected
from src.rsi_controller.config import MutationPolicy, RSIConfig, VerifierTrustThresholds
from src.rsi_controller.controller import DEFAULT_STATE, RSIController
from src.rsi_controller.evaluation import EvaluationStack, synthetic_experiment_entrypoint
from src.rsi_controller.memory import ResearchMemory
from src.rsi_controller.models import ChampionSnapshot, Component, EvaluationPlan, ExpectedEffect, ImprovementProposal, PromotionDecision, ResourceLimits, RSIMode
from src.rsi_controller.promotion import PromotionGate, detect_false_progress
from src.rsi_controller.sandbox import SandboxRunner
from src.rsi_controller.verifier_lag import RecursiveVerificationLagMonitor


def proposal(patch=None, component=Component.HARNESS):
    return ImprovementProposal(
        id="p1", hypothesis="bounded hypothesis", target_component=component,
        patch_or_config_change=patch or {"reasoning_budget":3},
        expected_effects=(ExpectedEffect("trusted_score","increase",.01),),
        risk_factors=("latency",), evaluation_plan=EvaluationPlan(),
        rollback_plan="restore c0", parent_champion_id="c0", deterministic_seed=17,
        resource_limits=ResourceLimits(wall_time_s=2,cpu_time_s=1,memory_mb=1024),
    )


class ProposalMutationTests(unittest.TestCase):
    def test_schema_and_allowlist(self):
        c=CandidateGenerator(MutationPolicy(),RSIMode.HARNESS).generate(proposal(),DEFAULT_STATE)
        self.assertEqual(c.full_state["H"]["reasoning_budget"],3)
        with self.assertRaises(MutationRejected):
            CandidateGenerator(MutationPolicy(),RSIMode.HARNESS).generate(proposal({"promotion_rules":"weaken"}),DEFAULT_STATE)

    def test_eval_code_and_training_mutation_blocked(self):
        gen=CandidateGenerator(MutationPolicy(allow_code_patches=True),RSIMode.HARNESS)
        with self.assertRaises(MutationRejected):
            gen.generate(proposal({"__code_patch__":{"src/rsi_controller/evaluation.py":"tamper"}}),DEFAULT_STATE)
        with self.assertRaises(MutationRejected):
            CandidateGenerator(MutationPolicy(),RSIMode.HARNESS).generate(proposal({"learning_rate":.1},Component.TRAINING),DEFAULT_STATE)
        patching=CandidateGenerator(MutationPolicy(allow_code_patches=True),RSIMode.HARNESS)
        for escaped in (
            "src/rsi_controller/mutable_harness/../evaluation.py",
            "../src/rsi_controller/mutable_harness/x.py",
            "src\\rsi_controller\\mutable_harness\\x.py",
        ):
            with self.assertRaises(MutationRejected):
                patching.generate(
                    proposal({"__code_patch__":{escaped:"tamper"}}),
                    DEFAULT_STATE,
                )


class EvaluationPromotionTests(unittest.TestCase):
    def setUp(self):
        self.stack=EvaluationStack(17)
        self.base=self.stack.evaluate_state(DEFAULT_STATE,seed=17,policy_version=0,verifier_version=0,policy_verifier_age=0)

    def test_hidden_eval_separation(self):
        desc=self.stack.public_description()
        self.assertEqual(desc["sealed_contents"],"unavailable to improvement planner/candidates")
        self.assertIn("terminal audit only",desc["sealed_access_policy"])
        self.assertIsNone(self.base.sealed)
        before=dict(desc["suite_digests"])
        state=json.loads(json.dumps(DEFAULT_STATE)); state["H"]["reasoning_budget"]=3
        adaptive=self.stack.evaluate_state(state,seed=17,policy_version=1,verifier_version=0,policy_verifier_age=1)
        self.assertIsNone(adaptive.sealed)
        terminal=self.stack.evaluate_state(
            state,seed=17,policy_version=1,verifier_version=0,
            policy_verifier_age=1,include_sealed=True,
        )
        self.assertIsNotNone(terminal.sealed)
        self.assertEqual(before,self.stack.public_description()["suite_digests"])

    def test_reward_hacking_rejected(self):
        state=json.loads(json.dumps(DEFAULT_STATE)); state["H"]["format_guard"]="reward_optimized"
        cand=self.stack.evaluate_state(state,seed=17,policy_version=1,verifier_version=0,policy_verifier_age=1)
        hacking=detect_false_progress(cand,self.base)
        lag=RecursiveVerificationLagMonitor(VerifierTrustThresholds()).assess(cand,self.base)
        decision=PromotionGate(RSIConfig().promotion).decide("hack",cand,self.base,hacking,lag)
        self.assertTrue(hacking.flagged); self.assertFalse(decision.accepted)

    def test_capability_gain_promotes(self):
        state=json.loads(json.dumps(DEFAULT_STATE)); state["H"]["reasoning_budget"]=3
        cand=self.stack.evaluate_state(state,seed=17,policy_version=1,verifier_version=0,policy_verifier_age=1)
        hacking=detect_false_progress(cand,self.base); lag=RecursiveVerificationLagMonitor(VerifierTrustThresholds()).assess(cand,self.base)
        self.assertTrue(PromotionGate(RSIConfig().promotion).decide("good",cand,self.base,hacking,lag).accepted)

    def test_deterministic_seed_and_stale_verifier(self):
        raw={"suite_seed":17,"candidate_seed":17,"candidate_state":DEFAULT_STATE,"policy_version":0,"verifier_version":0,"policy_verifier_age":0}
        self.assertEqual(synthetic_experiment_entrypoint(raw),synthetic_experiment_entrypoint(raw))
        stale=self.stack.evaluate_state(DEFAULT_STATE,seed=17,policy_version=5,verifier_version=0,policy_verifier_age=5)
        self.assertEqual(RecursiveVerificationLagMonitor(VerifierTrustThresholds()).assess(stale,self.base).trust_level,"low")


class MemorySandboxTests(unittest.TestCase):
    def test_append_only_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=ResearchMemory(Path(tmp)/"memory.sqlite")
            try:
                m.add_champion(ChampionSnapshot("c0",0,DEFAULT_STATE,0,0))
                m.add_champion(ChampionSnapshot("c1",1,DEFAULT_STATE,1,0,"c0"))
                self.assertEqual(m.rollback("c0","regression").champion_id,"c0")
                with self.assertRaises(sqlite3.DatabaseError): m.db.execute("UPDATE champions SET generation=99 WHERE id='c0'")
                self.assertEqual(m.integrity_check(),"ok")
            finally: m.close()

    def test_generation_commit_is_atomic_on_champion_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=ResearchMemory(Path(tmp)/"memory.sqlite")
            try:
                m.add_champion(ChampionSnapshot("c0",0,DEFAULT_STATE,0,0))
                p=proposal()
                m.record_proposal(1,p)
                cand=CandidateGenerator(MutationPolicy(),RSIMode.HARNESS).generate(p,DEFAULT_STATE)
                m.record_candidate(cand)
                decision=PromotionDecision(cand.candidate_id,True,("ok",),{},{"gate":True})
                duplicate=ChampionSnapshot("c0",1,DEFAULT_STATE,1,0,"c0")
                with self.assertRaises(sqlite3.IntegrityError):
                    m.commit_generation(
                        decision,1,cand.candidate_id,"lesson",champion=duplicate
                    )
                self.assertEqual(
                    m.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0],0
                )
                self.assertEqual(
                    m.db.execute("SELECT COUNT(*) FROM lessons").fetchone()[0],0
                )
                self.assertEqual(m.current_champion().champion_id,"c0")
            finally:
                m.close()

    def test_timeout_recovery_and_kill_switch(self):
        runner=SandboxRunner(("src.rsi_controller.sandbox:timeout_test_entrypoint","src.rsi_controller.evaluation:synthetic_experiment_entrypoint"))
        limits=ResourceLimits(wall_time_s=.15,cpu_time_s=1,memory_mb=1024)
        self.assertTrue(runner.run("src.rsi_controller.sandbox:timeout_test_entrypoint",{"sleep_s":1},limits).timed_out)
        ok=runner.run("src.rsi_controller.evaluation:synthetic_experiment_entrypoint",{"suite_seed":17,"candidate_seed":17,"candidate_state":DEFAULT_STATE,"policy_version":0,"verifier_version":0,"policy_verifier_age":0},replace(limits,wall_time_s=3,cpu_time_s=5))
        self.assertTrue(ok.ok,ok.error)
        runner.kill_switch()
        with self.assertRaises(RuntimeError): runner.run("src.rsi_controller.evaluation:synthetic_experiment_entrypoint",{},limits)


class EndToEndTests(unittest.TestCase):
    def test_multigeneration_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=replace(RSIConfig(),generations=3,output_dir=tmp,resources=ResourceLimits(wall_time_s=4,cpu_time_s=2,memory_mb=1024,max_output_bytes=100000))
            c=RSIController(cfg)
            try:
                summary=c.run(3); rows=c.metrics.read(); outcomes=[r.promotion_decision for r in rows]
                self.assertEqual(len(rows),4)
                self.assertIn("PROMOTED",outcomes); self.assertIn("REJECTED",outcomes)
                self.assertGreaterEqual(summary["promoted"],1); self.assertGreaterEqual(summary["rejected"],1)
                self.assertTrue((Path(tmp)/"research_memory.sqlite").exists())
                self.assertEqual(len(list((Path(tmp)/"metrics").glob("*.svg"))),6)
                self.assertIn("final_sealed_audit",summary)
                self.assertIn("never used",summary["final_sealed_audit"]["access_policy"])
                events=c.memory.recent_events(500)
                self.assertTrue(any(e["kind"]=="rvl_assessment" for e in events))
                decisions=[e["seq"] for e in events if e["kind"]=="promotion_decision"]
                sealed=[e["seq"] for e in events if e["kind"]=="sealed_final_audit"]
                self.assertEqual(len(sealed),1)
                self.assertTrue(decisions and max(decisions)<sealed[0])
                splits={
                    row[0] for row in c.memory.db.execute(
                        "SELECT DISTINCT split FROM evaluations"
                    ).fetchall()
                }
                self.assertEqual(splits,{"evolution","development","promotion"})
                with self.assertRaises(RuntimeError):
                    c.run(4)
            finally: c.close()


if __name__=="__main__":
    unittest.main()
