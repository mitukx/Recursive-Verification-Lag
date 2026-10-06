import os
import unittest
from unittest import mock

from src.rvl_systems.torch_distributed import context_from_env


class TorchDistributedConfigTest(unittest.TestCase):
    def test_context_from_torchrun_environment(self):
        env = {"RANK": "2", "WORLD_SIZE": "8", "LOCAL_RANK": "2"}
        with mock.patch.dict(os.environ, env, clear=False):
            ctx = context_from_env("nccl")
        self.assertEqual(ctx.rank, 2)
        self.assertEqual(ctx.world_size, 8)
        self.assertEqual(ctx.local_rank, 2)
        self.assertEqual(ctx.backend, "nccl")

    def test_defaults_are_single_process(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            ctx = context_from_env()
        self.assertEqual(ctx.rank, 0)
        self.assertEqual(ctx.world_size, 1)
        self.assertEqual(ctx.local_rank, 0)


if __name__ == "__main__":
    unittest.main()
