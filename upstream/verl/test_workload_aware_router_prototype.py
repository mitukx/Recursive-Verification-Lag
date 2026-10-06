import unittest

from upstream.verl.workload_aware_router_prototype import (
    WorkloadAwareRequestLoadBalancer,
)
from upstream.verl.benchmark_workload_aware_router import run


class WorkloadAwareRouterTests(unittest.TestCase):
    def test_routes_by_outstanding_work_not_request_count(self):
        lb = WorkloadAwareRequestLoadBalancer({"a": None, "b": None})
        a, _ = lb.acquire_server(
            "long", prompt_ids=[0] * 1000,
            sampling_params={"max_tokens": 1000},
        )
        b, _ = lb.acquire_server(
            "short", prompt_ids=[0] * 10,
            sampling_params={"max_tokens": 10},
        )
        self.assertNotEqual(a, b)
        # Both servers now have one request. Least-inflight would see a tie,
        # while work-aware routing should choose the short-loaded server.
        c, _ = lb.acquire_server(
            "next", prompt_ids=[0] * 20,
            sampling_params={"max_tokens": 20},
        )
        self.assertEqual(c, b)

    def test_release_restores_exact_accounting(self):
        lb = WorkloadAwareRequestLoadBalancer({"a": None, "b": None})
        sid, _ = lb.acquire_server(
            "r", prompt_ids=[1] * 123,
            sampling_params={"max_new_tokens": 77},
        )
        self.assertEqual(lb.get_status()["total_predicted_work"], 200)
        lb.release_server(sid, request_id="r")
        self.assertEqual(lb.get_status()["total_predicted_work"], 0)
        self.assertEqual(lb.get_total_inflight(), 0)

    def test_duplicate_inflight_request_id_fails_closed(self):
        lb = WorkloadAwareRequestLoadBalancer({"a": None})
        lb.acquire_server("same", prompt_ids=[1], sampling_params={})
        with self.assertRaises(RuntimeError):
            lb.acquire_server("same", prompt_ids=[1], sampling_params={})

    def test_removed_server_does_not_leave_accounting(self):
        lb = WorkloadAwareRequestLoadBalancer({"a": None, "b": None})
        sid, _ = lb.acquire_server(
            "r", prompt_ids=[1] * 10,
            sampling_params={"max_tokens": 10},
        )
        lb.remove_servers([sid])
        self.assertNotIn(sid, lb.get_status()["server_work"])
        lb.release_server(sid, request_id="r")

    def test_trace_benchmark_improves_heterogeneous_balance(self):
        report = run(n=256, servers=8, seed=17)
        self.assertLess(
            report["workload_aware"]["work_imbalance_ratio"],
            report["least_inflight"]["work_imbalance_ratio"],
        )
        self.assertLess(
            report["workload_aware"]["makespan_s"],
            report["least_inflight"]["makespan_s"],
        )


if __name__ == "__main__":
    unittest.main()
