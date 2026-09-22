import unittest

import pandas as pd

from src.candidate_bank_experiment import Config, run


class CandidateBankExperimentTest(unittest.TestCase):
    def test_smoke(self):
        df = pd.DataFrame(
            {
                "task_id": ["a", "a", "b", "b"],
                "candidate_id": ["a0", "a1", "b0", "b1"],
                "trusted_score": [0.0, 1.0, 0.0, 1.0],
                "base_logprob": [0.0, 0.0, 0.0, 0.0],
                "f::public_score": [0.0, 1.0, 0.0, 1.0],
            }
        )
        out = run(df, Config(eta=0.5, refresh_interval=1, rounds=3, audit_per_refresh=8, seed=1))
        self.assertEqual(len(out), 3)
        self.assertTrue({"true_reward", "proxy_reward", "kl_from_previous"}.issubset(out.columns))
        self.assertTrue(out["true_reward"].notna().all())


if __name__ == "__main__":
    unittest.main()
