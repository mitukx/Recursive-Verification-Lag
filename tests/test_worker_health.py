import unittest

from src.rvl_systems.worker_health import WorkerHealth


class WorkerHealthTest(unittest.TestCase):
    def test_quarantines_after_threshold_and_recovers(self):
        health = WorkerHealth(failure_threshold=2)
        self.assertFalse(health.record_failure("w0"))
        self.assertTrue(health.is_available("w0"))
        self.assertTrue(health.record_failure("w0"))
        self.assertFalse(health.is_available("w0"))
        health.recover("w0")
        self.assertTrue(health.is_available("w0"))
        self.assertEqual(
            health.consecutive_failures["w0"],
            0,
        )

    def test_success_resets_failure_streak(self):
        health = WorkerHealth(failure_threshold=2)
        health.record_failure("w0")
        health.record_success("w0")
        self.assertFalse(
            health.record_failure("w0")
        )
        self.assertTrue(
            health.is_available("w0")
        )

    def test_half_open_reservation_is_single_probe(self):
        health = WorkerHealth(
            failure_threshold=1,
            quarantine_cooldown_s=0.0,
        )
        self.assertTrue(
            health.record_failure("w0")
        )
        self.assertTrue(
            health.is_available("w0")
        )
        self.assertTrue(
            health.reserve("w0")
        )
        self.assertFalse(
            health.is_available("w0")
        )
        self.assertFalse(
            health.reserve("w0")
        )
        health.cancel_reservation("w0")
        self.assertTrue(
            health.is_available("w0")
        )
        self.assertTrue(
            health.reserve("w0")
        )
        health.record_success("w0")
        self.assertTrue(
            health.is_available("w0")
        )
        self.assertNotIn(
            "w0",
            health.half_open,
        )


if __name__ == "__main__":
    unittest.main()
