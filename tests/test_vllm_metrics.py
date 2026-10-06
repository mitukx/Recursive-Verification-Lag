import unittest

from src.rvl_systems.vllm_metrics import (
    counter_delta,
    max_metric,
    parse_prometheus_samples,
    sum_metric,
)


class VLLMMetricsTest(unittest.TestCase):
    def test_parses_labels_and_aggregates(self):
        text = """
# HELP ignored help
vllm:generation_tokens_total{model_name="a"} 10
vllm:generation_tokens_total{model_name="b"} 5
vllm:kv_cache_usage_perc{model_name="a"} 0.25
"""
        samples = parse_prometheus_samples(text)
        self.assertEqual(
            sum_metric(samples, "vllm:generation_tokens_total"),
            15,
        )
        self.assertEqual(
            max_metric(samples, "vllm:kv_cache_usage_perc"),
            0.25,
        )

    def test_counter_delta_never_goes_negative(self):
        before = {"x": [10.0]}
        after = {"x": [3.0]}
        self.assertEqual(counter_delta(before, after, "x"), 0.0)


if __name__ == "__main__":
    unittest.main()
