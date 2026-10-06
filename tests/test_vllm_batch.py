import asyncio
import json
import unittest
from unittest import mock

from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.vllm_batch import VLLMBatchChatBackend


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class VLLMBatchBackendTest(unittest.TestCase):
    def test_batch_chat_contract_and_index_mapping(self):
        def fake_urlopen(request, timeout):
            self.assertTrue(request.full_url.endswith("/v1/chat/completions/batch"))
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(payload["model"], "model")
            self.assertEqual(payload["n"], 1)
            self.assertEqual(len(payload["messages"]), 2)
            self.assertEqual(payload["temperature"], 0.8)
            self.assertEqual(payload["seed"], 7)
            return FakeResponse({
                "choices": [
                    {
                        "index": 1,
                        "message": {"content": "second"},
                        "token_ids": [12],
                        "logprobs": {"content": [{"token": "second", "logprob": -0.2}]},
                    },
                    {
                        "index": 0,
                        "message": {"content": "first"},
                        "token_ids": [11],
                        "logprobs": {"content": [{"token": "first", "logprob": -0.1}]},
                    },
                ]
            })

        async def run():
            backend = VLLMBatchChatBackend("http://worker", "model")
            requests = [
                RolloutRequest("a", "prompt a", samples=1, temperature=0.8, seed=7),
                RolloutRequest("b", "prompt b", samples=1, temperature=0.8, seed=7),
            ]
            with mock.patch("src.rvl_systems.vllm_batch.urllib.request.urlopen", fake_urlopen):
                groups = await backend.generate_batch(requests)
            self.assertEqual([g[0].response for g in groups], ["first", "second"])
            self.assertEqual(groups[0][0].metadata["batch_size"], 2)
            self.assertAlmostEqual(groups[0][0].logprob, -0.1)

        asyncio.run(run())

    def test_rejects_unsupported_sampling_contract(self):
        async def run():
            backend = VLLMBatchChatBackend("http://worker", "model")
            with self.assertRaises(ValueError):
                await backend.generate_batch([
                    RolloutRequest("a", "a", samples=2, temperature=0.8, seed=1)
                ])
            with self.assertRaises(ValueError):
                await backend.generate_batch([
                    RolloutRequest("a", "a", samples=1, temperature=0.8, seed=1),
                    RolloutRequest("b", "b", samples=1, temperature=0.8, seed=2),
                ])
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
