import unittest

from src.refresh_controller import (
    FixedCadence,
    GeometryMarginController,
    MovementState,
    ScalarThreshold,
)


class RefreshControllerTest(unittest.TestCase):
    def test_fixed_cadence(self):
        c = FixedCadence(interval=3)
        self.assertFalse(c.should_refresh(MovementState(rounds_since_refresh=2)))
        self.assertTrue(c.should_refresh(MovementState(rounds_since_refresh=3)))

    def test_scalar_threshold(self):
        c = ScalarThreshold(metric="cumulative_kl", threshold=0.5)
        self.assertFalse(c.should_refresh(MovementState(2, cumulative_kl=0.49)))
        self.assertTrue(c.should_refresh(MovementState(2, cumulative_kl=0.5)))

    def test_geometry_margin(self):
        c = GeometryMarginController(threshold=4.0)
        low = MovementState(2, restricted_geometry=0.03, proxy_margin=0.1)
        high = MovementState(2, restricted_geometry=0.05, proxy_margin=0.1)
        self.assertFalse(c.should_refresh(low))
        self.assertTrue(c.should_refresh(high))


if __name__ == "__main__":
    unittest.main()
