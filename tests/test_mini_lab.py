import asyncio
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.rvl_systems.lab.agent import (
    Snapshot, TabularLearner, WeightRegistry, public_reward, rollout, trusted_reward)
from src.rvl_systems.lab.contracts import Step, Task, Trajectory, Verdict
from src.rvl_systems.lab.control import ControlConfig, RVLControlPlane, Curriculum
from src.rvl_systems.lab.promotion import PromotionDecision
from src.rvl_systems.lab.runtime import LabConfig, MiniLab, percentile
from src.rvl_systems.lab.store import LeaseLost, ReplayStore
from src.rvl_systems.lab.verification import VerifierEnsemble
from src.rvl_systems.lab.verification_debt import (
    VerificationDebtConfig, VerificationDebtController, VerificationDebtSignals)


def trajectory(i="one", version=0, program=(0,2), action=2):
    return Trajectory(i,Task("task",0,1,2),version,
                      (Step("0:0:0",action,math.log(1/7),"edit"),),program,True,17)


def verdict():
    return Verdict(1,0,(1,.5),"cell",.5)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/"replay.sqlite"
        self.store = ReplayStore(self.path,capacity=2)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_restart_idempotency_and_collision(self):
        t = trajectory()
        self.assertTrue(self.store.put(t,verdict()))
        self.assertFalse(self.store.put(replace(t,elapsed_s=4),verdict()))
        with self.assertRaises(ValueError):
            self.store.put(replace(t,program=(1,2)),verdict())
        self.store.close()
        self.store = ReplayStore(self.path,capacity=2)
        self.assertEqual(self.store.counts(),{"ready":1})

    def test_capacity_and_future_policy_fence(self):
        self.store.put(trajectory("a",version=2),verdict())
        self.store.put(trajectory("b",version=2),verdict())
        self.assertFalse(self.store.put(trajectory("c"),verdict()))
        self.assertEqual(self.store.claim(2,1,2,now=10),[])

    def test_lease_expiry_recovery_and_fenced_ack(self):
        self.store.put(trajectory(),verdict())
        old,t,_ = self.store.claim(1,0,2,lease_s=2,now=10)[0]
        self.assertEqual(self.store.claim(1,0,2,now=11),[])
        new,_,_ = self.store.claim(1,0,2,lease_s=5,now=13)[0]
        with self.assertRaises(LeaseLost):
            self.store.finish(t.trajectory_id,old,now=14)
        self.store.finish(t.trajectory_id,new,now=14)
        self.assertEqual(self.store.counts(),{"consumed":1})

    def test_stale_relabel_preserves_behavior(self):
        t = trajectory()
        self.store.put(t,verdict())
        self.assertEqual(self.store.claim(1,4,1,now=10),[])
        self.assertEqual(self.store.counts(),{"stale":1})
        v = Verdict(0,3,(1,0),"cell",0,True)
        self.store.relabel(t.trajectory_id,v)
        recovered,updated = self.store.candidates()[0]
        self.assertEqual(recovered,t)
        self.assertEqual(updated.verifier_version,3)
        self.assertTrue(updated.trusted)

    def test_two_connections_cannot_claim_same_work(self):
        self.store.put(trajectory(),verdict())
        other = ReplayStore(self.path)
        try:
            self.assertEqual(len(self.store.claim(1,0,2,now=10)),1)
            self.assertEqual(other.claim(1,0,2,now=10),[])
        finally:
            other.close()

    def test_transaction_rollback_restores_state_and_consumption(self):
        self.store.put(trajectory(),verdict())
        token,t,_ = self.store.claim(1,0,2,now=10)[0]
        self.store.db.execute("BEGIN IMMEDIATE")
        self.store.finish(t.trajectory_id,token,now=11)
        self.store.save_state("learner",{"version":1})
        self.store.db.execute("ROLLBACK")
        self.assertIsNone(self.store.load_state("learner"))
        self.assertEqual(self.store.counts(),{"leased":1})


class PolicyTests(unittest.TestCase):
    def test_real_gradient_increases_good_action(self):
        l = TabularLearner()
        before = l.snapshot().probabilities("0:0:0")[1]
        l.train([(trajectory(program=(1,2),action=1),1)])
        self.assertGreater(l.snapshot().probabilities("0:0:0")[1],before)
        self.assertEqual(l.version,1)

    def test_clipped_off_policy_gradient_does_not_reinforce_far_positive_sample(self):
        l = TabularLearner()
        t = trajectory(action=1)
        t = replace(t,steps=(replace(t.steps[0],behavior_logprob=math.log(.001)),))
        metrics = l.train([(t,1)])
        self.assertEqual(l.snapshot().probabilities("0:0:0"),[1/7]*7)
        self.assertEqual(metrics["clip_fraction"],1)

    def test_snapshot_does_not_follow_learner_mutation(self):
        l = TabularLearner()
        s = l.snapshot()
        l.train([(trajectory(action=1),1)])
        self.assertEqual(s.version,0)
        self.assertEqual(s.probabilities("0:0:0"),[1/7]*7)

    def test_weight_checksum_and_monotonic_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = WeightRegistry(tmp)
            s = TabularLearner().snapshot()
            r.publish(s)
            with self.assertRaises(ValueError):
                r.publish(s)
            self.assertEqual(WeightRegistry(tmp).load(),s)
            raw = json.loads((Path(tmp)/"active.json").read_text())
            (Path(tmp)/raw["artifact"]).write_text("{}")
            with self.assertRaises(ValueError):
                WeightRegistry(tmp).load()

    def test_eval_task_cannot_enter_training_replay(self):
        with self.assertRaises(ValueError):
            replace(trajectory(),task=Task("eval",0,1,2,split="eval"))


class VerificationTests(unittest.TestCase):
    def test_executable_proxy_exploitation_and_critic_learning(self):
        v = VerifierEnsemble()
        attack = trajectory()
        good = trajectory("good",program=(1,2))
        self.assertEqual(public_reward(attack.task,attack.program),1)
        self.assertEqual(trusted_reward(attack.task,attack.program),0)
        before = v.score(attack).reward
        labels = [(attack,0)]*12+[(good,1)]*12
        v.fit(labels)
        self.assertLess(v.score(attack).reward,before)
        self.assertGreater(v.score(good).reward,v.score(attack).reward)
        self.assertEqual(v.audit(attack).reward,0)
        self.assertEqual(v.exploits,1)

    def test_critic_cannot_reward_known_executable_failure_in_colliding_cell(self):
        v = VerifierEnsemble()
        good = trajectory("good",program=(1,2))
        bad = trajectory("bad",program=(1,3))
        v.fit([(good,1)]*100)
        self.assertEqual(v.score(good).cell,v.score(bad).cell)
        self.assertGreater(v.score(bad).scores[1],.9)
        self.assertEqual(public_reward(bad.task,bad.program),0)
        self.assertEqual(v.score(bad).reward,0)

    def test_controller_budget_and_fixed_cadence(self):
        v,l = VerifierEnsemble(),TabularLearner()
        rows = [(trajectory(str(i)),verdict()) for i in range(4)]
        c = RVLControlPlane(ControlConfig(audit_budget=3,audit_per_batch=2,risk_threshold=0))
        self.assertEqual(len(c.select_audits(rows,v,l.snapshot(),0)),2)
        self.assertEqual(len(c.select_audits(rows,v,l.snapshot(),1)),1)
        self.assertEqual(c.select_audits(rows,v,l.snapshot(),2),[])
        fixed = RVLControlPlane(ControlConfig(mode="fixed",cadence=3))
        self.assertEqual(fixed.select_audits(rows,v,l.snapshot(),0),[])
        self.assertEqual(len(fixed.select_audits(rows,v,l.snapshot(),2)),2)

    def test_error_geometry_changes_priority_at_same_policy_shift(self):
        v,l = VerifierEnsemble(),TabularLearner()
        a,b = trajectory("a"),trajectory("b",program=(1,2))
        v.fit([(a,0)]*50+[(b,1)]*50)
        c = RVLControlPlane(ControlConfig())
        self.assertGreater(c.priority(a,v.score(a),v,l.snapshot()),
                           c.priority(b,v.score(b),v,l.snapshot()))

    def test_active_search_is_not_false_behavior_policy_data(self):
        c = Curriculum()
        attack = c.attack(trajectory().task,TabularLearner().snapshot(),0)
        self.assertEqual(attack.steps,())
        self.assertEqual(trusted_reward(attack.task,attack.program),0)

    def test_failure_task_replay_is_fresh_and_bounded(self):
        c = Curriculum()
        t = trajectory()
        for _ in range(100):
            c.observe(t,0)
        self.assertEqual(len(c.failed_tasks),64)
        replayed = [c.generate(i,i) for i in range(100)]
        self.assertEqual(len({t.task_id for t in replayed}),100)
        self.assertTrue(any(t.coefficient==1 and t.bias==2 for t in replayed))

    def test_invalid_verdict_nan_and_config(self):
        with self.assertRaises(ValueError):
            Verdict(float("nan"),0,(),"cell",0)
        with self.assertRaises(ValueError):
            LabConfig(actors=2,deterministic=True)
        with self.assertRaises(ValueError):
            ControlConfig(mode="unknown")


class VerificationDebtTests(unittest.TestCase):
    def test_debt_levels_are_monotonic_and_bounded(self):
        controller = VerificationDebtController(
            VerificationDebtConfig(soft_limit=2.0,hard_limit=4.0)
        )
        low = controller.assess(VerificationDebtSignals(pending=1))
        elevated = controller.assess(VerificationDebtSignals(pending=2))
        high = controller.assess(VerificationDebtSignals(pending=4))
        self.assertEqual(low.action,"admit_generation")
        self.assertEqual(elevated.action,"throttle_generation")
        self.assertEqual(high.action,"pause_generation")
        self.assertLess(low.score,elevated.score)
        self.assertLess(elevated.score,high.score)

    def test_replay_debt_tracks_queue_age_and_stale_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            from src.rvl_systems.lab.token_replay import TokenReplay
            from src.rvl_systems.types import Generation, VerifiedGeneration
            store = TokenReplay(Path(tmp)/"replay.sqlite",capacity=4)
            try:
                g = Generation("p","prompt","response",0.0,1,0.0,{})
                store.put_pending("pending",0,[g],now=10)
                signals = store.verification_debt_signals(2,1,now=20)
                self.assertEqual(signals.pending,1)
                self.assertEqual(signals.max_policy_lag,2)
                self.assertEqual(signals.oldest_unverified_age_s,10)
                token,rid,_,rows = store.claim_verification(2,4,1,1,now=20)
                store.complete_verification(
                    rid,token,[VerifiedGeneration(rows[0],1.0,0.0,0)],now=21
                )
                signals = store.verification_debt_signals(2,1,now=22)
                self.assertEqual(signals.stale_rewards,1)
                self.assertEqual(signals.max_verifier_lag,1)
            finally:
                store.close()


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_step_checkpoint_recovers_exact_action_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/"episode.json"
            s = TabularLearner().snapshot()
            t = trajectory().task
            first = await rollout(t,s,17,"episode",max_steps=4,checkpoint=checkpoint)
            resumed = await rollout(t,s,17,"episode",max_steps=24,checkpoint=checkpoint)
            fresh = await rollout(t,s,17,"episode",max_steps=24)
            self.assertEqual(resumed.steps,fresh.steps)
            self.assertEqual(resumed.program,fresh.program)
            self.assertEqual(resumed.steps[:len(first.steps)],first.steps)
            with self.assertRaises(ValueError):
                await rollout(t,s,99,"episode",checkpoint=checkpoint)

    async def test_wall_clock_deadline_is_enforced(self):
        with self.assertRaises(TimeoutError):
            await rollout(trajectory().task,TabularLearner().snapshot(),17,"episode",
                          tool_latency_s=1,deadline_s=.001)

    async def test_end_to_end_real_updates_refit_budget_and_drained_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab = MiniLab(tmp,LabConfig(episodes=64,actors=3,capacity=8,batch_size=4,tool_latency_s=0),
                          ControlConfig(audit_budget=16,refit_labels=4,risk_threshold=0))
            try:
                report = await asyncio.wait_for(lab.run(),30)
                self.assertGreater(report["batches"],0)
                self.assertTrue(lab.learner.logits)
                self.assertGreater(lab.verifier.version,0)
                self.assertLessEqual(report["trusted_calls"],16)
                self.assertGreater(report["exploits_detected"],0)
                self.assertEqual(report["replay_counts"].get("ready",0),0)
                self.assertEqual(report["replay_counts"].get("leased",0),0)
                self.assertTrue((Path(tmp)/"report.json").exists())
                versions = [d["max_policy_lag"] for d in report["history"]]
                self.assertTrue(all(0 <= v <= lab.cfg.max_policy_lag for v in versions))
            finally:
                lab.close()

    async def test_deterministic_semantics_two_fresh_runs(self):
        reports = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                lab = MiniLab(tmp,LabConfig(episodes=32,actors=1,batch_size=4,deterministic=True,tool_latency_s=0),
                              ControlConfig(audit_budget=8,refit_labels=4,risk_threshold=0))
                try:
                    reports.append(await asyncio.wait_for(lab.run(),30))
                finally:
                    lab.close()
        self.assertEqual(reports[0]["semantic_sha256"],reports[1]["semantic_sha256"])

    async def test_restart_extends_without_retraining_consumed_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = LabConfig(episodes=16,actors=1,batch_size=4,deterministic=True,tool_latency_s=0)
            control = ControlConfig(audit_budget=8,refit_labels=4,risk_threshold=0)
            lab = MiniLab(tmp,cfg,control)
            try:
                first = await asyncio.wait_for(lab.run(),30)
            finally:
                lab.close()
            lab = MiniLab(tmp,replace(cfg,episodes=32),control)
            try:
                second = await asyncio.wait_for(lab.run(),30)
                self.assertGreater(second["batches"],first["batches"])
                self.assertEqual(sum(second["replay_counts"].values()),32)
                self.assertEqual(len(lab.store.labeled()),len({t.trajectory_id for t,_ in lab.store.labeled()}))
            finally:
                lab.close()

    async def test_regression_gate_rolls_back_rejected_learner_and_serving(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = LabConfig(episodes=8,actors=1,batch_size=4,deterministic=True,tool_latency_s=0)
            lab = MiniLab(tmp,cfg)
            try:
                for i in range(4):
                    lab.store.put(trajectory(str(i),program=(1,2),action=1),verdict())
                lab._promotion_decision = lambda incumbent,candidate: PromotionDecision(
                    True,incumbent.version,candidate.version,0.5,0.75,0.25,{"0":0.25},
                    4,0,44,0,(),"a"*64)
                accepted = lab._train_batch(lab.store.claim(4,0,16))
                self.assertTrue(accepted)
                served_version = lab.serving.version
                served_state = json.loads(json.dumps(lab.learner.state()))
                for i in range(4,8):
                    lab.store.put(trajectory(str(i),program=(1,2),action=1),verdict())
                lab._promotion_decision = lambda incumbent,candidate: PromotionDecision(
                    False,incumbent.version,candidate.version,0.75,0.0,-0.75,{"0":-0.75},
                    0,4,44,4,("mean_reward_regression",),"b"*64)
                self.assertFalse(lab._train_batch(lab.store.claim(4,served_version,16)))
                self.assertEqual(lab.serving.version,served_version)
                self.assertEqual(lab.learner.state(),served_state)
                self.assertEqual(lab.promotion_ledger.seq,2)
            finally:
                lab.close()
            restarted = MiniLab(tmp,cfg)
            try:
                self.assertEqual(restarted.registry.active.version,served_version)
                self.assertEqual(restarted.learner.version,served_version)
                self.assertEqual(restarted.promotion_ledger.seq,2)
            finally:
                restarted.close()

    async def test_reserved_episode_is_recovered_on_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = LabConfig(episodes=8,actors=1,batch_size=4,deterministic=True,tool_latency_s=0)
            lab = MiniLab(tmp,cfg)
            self.assertIsNotNone(lab._reserve())
            lab.close()
            lab = MiniLab(tmp,cfg)
            try:
                r = await asyncio.wait_for(lab.run(),30)
                self.assertEqual(sum(r["replay_counts"].values()),8)
            finally:
                lab.close()


class MeasurementTests(unittest.TestCase):
    def test_tail_percentiles(self):
        self.assertAlmostEqual(percentile([0,1,2,3],.99),2.97)
        self.assertEqual(percentile([],.95),0)


if __name__ == "__main__":
    unittest.main()
