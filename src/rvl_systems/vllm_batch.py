from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from dataclasses import dataclass

from .rollout import RolloutRequest
from .types import Generation


@dataclass
class VLLMBatchChatBackend:
    """Adapter for vLLM's /v1/chat/completions/batch endpoint.

    Current vLLM batch-chat semantics return one completion per conversation,
    so every RolloutRequest must use samples=1. A DynamicBatcher groups
    compatible temperature/seed requests before calling this adapter.
    """

    endpoint: str
    model: str
    timeout_s: float = 120.0

    async def generate_batch(
        self,
        requests: list[RolloutRequest],
    ) -> list[list[Generation]]:
        return await asyncio.to_thread(self._generate_batch_sync, requests)

    def _generate_batch_sync(
        self,
        requests: list[RolloutRequest],
    ) -> list[list[Generation]]:
        if not requests:
            return []
        if any(request.samples != 1 for request in requests):
            raise ValueError("vLLM batch chat currently requires samples=1")
        temperatures = {float(request.temperature) for request in requests}
        seeds = {request.seed for request in requests}
        if len(temperatures) != 1 or len(seeds) != 1:
            raise ValueError("batched requests must share temperature and seed")

        payload = {
            "model": self.model,
            "messages": [
                [{"role": "user", "content": request.prompt}]
                for request in requests
            ],
            "temperature": next(iter(temperatures)),
            "seed": next(iter(seeds)),
            "n": 1,
            "logprobs": True,
            "return_token_ids": True,
        }
        url = self.endpoint.rstrip("/") + "/v1/chat/completions/batch"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        latency = time.perf_counter() - start

        choices = body.get("choices") or []
        by_index = {int(choice["index"]): choice for choice in choices}
        if set(by_index) != set(range(len(requests))):
            raise RuntimeError("vLLM batch response indices do not match request batch")

        output: list[list[Generation]] = []
        for index, request in enumerate(requests):
            choice = by_index[index]
            message = choice.get("message") or {}
            text = str(message.get("content") or "")
            logprob_rows = ((choice.get("logprobs") or {}).get("content") or [])
            token_logprobs = [
                float(row["logprob"])
                for row in logprob_rows
                if isinstance(row, dict) and row.get("logprob") is not None
            ]
            token_ids = choice.get("token_ids") or []
            token_count = len(token_ids) if token_ids else max(1, len(token_logprobs))
            generation = Generation(
                prompt_id=request.prompt_id,
                prompt=request.prompt,
                response=text,
                logprob=sum(token_logprobs),
                token_count=token_count,
                latency_s=latency,
                metadata={
                    "backend": "vllm-batch-chat",
                    "model": self.model,
                    "batch_size": len(requests),
                    "response_token_ids": [int(x) for x in token_ids],
                    "response_token_logprobs": token_logprobs,
                },
            )
            output.append([generation])
        return output
