import unittest

from upstream.verl.workload_aware_router_prototype import (
    WorkloadAwareRequestLoadBalancer,
)
from upstream.verl.benchmark_workload_aware_router import run


class WorkloadAwareRouterTests(unittest.TestCase):
    def test_routes_by_outstanding_work_not_request_count(self):
        lb = WorkloadAwareRequestLoadBalancer({"a": None, "b": None})
        a, _ = lb.acquire_server(
            "long", prompt_tokens=1000, decode_budget=1000,
        )
        b, _ = lb.acquire_server(
            "short", prompt_tokens=10, decode_budget=10,
        )
        self.assertNotEqual(a, b)
        # Both servers now have one request. Least-inflight would see a tie,
        # while work-aware routing should choose the short-loaded server.
        c, _ = lb.acquire_server(
            "next", prompt_tokens=20, decode_budget=20,
        )
        self.assertEqual(c, b)

    def test_release_restores_exact_accounting(self):
        lb = WorkloadAwareRequestLoadBalancer({"a": None, "b": None})
        sid, _ = lb.acquire_server(
            "r", prompt_tokens=123, decode_budget=77,
        )
        self.assertEqual(lb.get_status()["total_predicted_work"], 200)
        lb.release_server(sid, request_id="r")
        self.assertEqual(lb.get_status()["total_predicted_work"], 0)
        self.assertEqual(lb.get_total_inflight(), 0)

    def test_duplicate_inflight_request_id_fails_closed(self):
        lb = WorkloadAwareRequestLoadBalancer({"a": None})
        lb.acquire_server("same", prompt_tokens=1, decode_budget=1)
        with self.assertRaises(RuntimeError):
            lb.acquire_server("same", prompt_tokens=1, decode_budget=1)

    def test_inflight_accounting_is_not_evicted_by_sticky_cache(self):
        lb = WorkloadAwareRequestLoadBalancer(
            {"a": None, "b": None}, max_cache_size=2
        )
        assignments = []
        for i in range(8):
            sid, _ = lb.acquire_server(
                f"r{i}", prompt_tokens=i + 1, decode_budget=1
            )
            assignments.append((sid, f"r{i}"))
        self.assertEqual(lb.get_total_inflight(), 8)
        self.assertGreater(lb.get_status()["total_predicted_work"], 0)
        for sid, request_id in assignments:
            lb.release_server(sid, request_id=request_id)
        self.assertEqual(lb.get_total_inflight(), 0)
        self.assertEqual(lb.get_status()["total_predicted_work"], 0)

    def test_release_mismatch_preserves_accounting(self):
        lb = WorkloadAwareRequestLoadBalancer({"a": None, "b": None})
        sid, _ = lb.acquire_server("r", prompt_tokens=100, decode_budget=20)
        wrong = "b" if sid == "a" else "a"
        with self.assertRaises(ValueError):
            lb.release_server(wrong, request_id="r")
        self.assertEqual(lb.get_total_inflight(), 1)
        self.assertEqual(lb.get_status()["total_predicted_work"], 120)
        lb.release_server(sid, request_id="r")
        self.assertEqual(lb.get_total_inflight(), 0)

    def test_removed_server_does_not_leave_accounting(self):
        lb = WorkloadAwareRequestLoadBalancer({"a": None, "b": None})
        sid, _ = lb.acquire_server(
            "r", prompt_tokens=10, decode_budget=10,
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
