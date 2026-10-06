import asyncio
import unittest

from src.rvl_systems.eval_fabric import (
    EvalTask,
    EvaluationFabric,
    exact_two_sided_sign_test,
)
from src.rvl_systems.rollout import AsyncRolloutEngine
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.types import Generation


class DeterministicBackend:
    def __init__(self, threshold, *, delay=0.0, fail_task=None):
        self.threshold = threshold
        self.delay = delay
        self.fail_task = fail_task
        self.seeds = {}

    async def generate(self,prompt_id,prompt,*,n,temperature,seed):
        if self.fail_task == prompt_id:
            raise ConnectionError("injected")
        if self.delay:
            await asyncio.sleep(self.delay)
        self.seeds[prompt_id] = seed
        correct = (seed % 10) < self.threshold
        return [Generation(
            prompt_id,prompt,"correct" if correct else "wrong",
            0.0,1,self.delay,{"seed":seed},
        )]


async def reward(task,generation):
    return float(generation.response == "correct")


class EvalFabricTests(unittest.IsolatedAsyncioTestCase):
    async def test_paired_seed_model_comparison_over_scheduler_and_rollout(self):
        a_backend = DeterministicBackend(4)
        b_backend0 = DeterministicBackend(8)
        b_backend1 = DeterministicBackend(8)
        runners = {
            "model-a":AsyncRolloutEngine(a_backend,max_concurrency=2),
            "model-b":LeastLoadedScheduler([
                WorkerSlot("b0",b_backend0,max_inflight=2),
                WorkerSlot("b1",b_backend1,max_inflight=2),
            ],queue_limit=8),
        }
        tasks = [
            EvalTask(f"t{i}",f"prompt {i}",slice="even" if i%2==0 else "odd")
            for i in range(20)
        ]
        report = await EvaluationFabric(
            runners,reward,max_inflight=6,seed=100
        ).run(tasks)
        self.assertTrue(report["complete"])
        self.assertEqual(report["successful_results"],40)
        comp = report["comparisons"][0]
        self.assertEqual(comp["complete_pairs"],20)
        self.assertGreater(comp["mean_delta_b_minus_a"],0)
        self.assertGreater(comp["wins_b"],comp["wins_a"])
        self.assertEqual(set(comp["slice_deltas_b_minus_a"]),{"even","odd"})
        for i,task in enumerate(tasks):
            expected = 100+i
            self.assertEqual(a_backend.seeds[task.task_id],expected)
            seen = b_backend0.seeds.get(
                task.task_id,b_backend1.seeds.get(task.task_id)
            )
            self.assertEqual(seen,expected)
        self.assertEqual(len(report["semantic_sha256"]),64)
        self.assertEqual(len(report["suite_sha256"]),64)

    async def test_infra_failure_is_not_converted_to_bad_model_reward(self):
        runners = {
            "a":AsyncRolloutEngine(DeterministicBackend(5,fail_task="t1")),
            "b":AsyncRolloutEngine(DeterministicBackend(5)),
        }
        tasks = [EvalTask("t0","p0"),EvalTask("t1","p1")]
        report = await EvaluationFabric(runners,reward).run(tasks)
        self.assertFalse(report["complete"])
        self.assertEqual(report["failure_count"],1)
        failed = [r for r in report["results"] if r["status"]!="ok"][0]
        self.assertIsNone(failed["reward"])
        self.assertEqual(failed["status"],"infra_failure")
        self.assertEqual(report["comparisons"][0]["incomplete_pairs"],1)

    async def test_evaluator_failure_is_explicit(self):
        async def broken(task,generation):
            if task.task_id=="bad":
                return float("nan")
            return 1.0
        runners = {
            "a":AsyncRolloutEngine(DeterministicBackend(10)),
            "b":AsyncRolloutEngine(DeterministicBackend(10)),
        }
        report = await EvaluationFabric(runners,broken).run([
            EvalTask("ok","p"),EvalTask("bad","p")
        ])
        self.assertFalse(report["complete"])
        failures = [
            r for r in report["results"]
            if r["status"]=="evaluator_failure"
        ]
        self.assertEqual(len(failures),2)
        self.assertTrue(all(r["reward"] is None for r in failures))

    async def test_duplicate_and_training_tasks_fail_closed(self):
        runners = {
            "a":AsyncRolloutEngine(DeterministicBackend(5)),
            "b":AsyncRolloutEngine(DeterministicBackend(5)),
        }
        fabric = EvaluationFabric(runners,reward)
        with self.assertRaises(ValueError):
            await fabric.run([EvalTask("x","p"),EvalTask("x","q")])
        with self.assertRaises(ValueError):
            EvalTask("train","p",split="train")

    def test_exact_sign_test(self):
        self.assertIsNone(exact_two_sided_sign_test(0,0))
        self.assertEqual(exact_two_sided_sign_test(5,5),1.0)
        self.assertLess(exact_two_sided_sign_test(0,10),0.01)


if __name__=="__main__":
    unittest.main()
