from __future__ import annotations

import asyncio
import json
import math
import random
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from .types import Generation


class InferenceBackend(Protocol):
    async def generate(
        self,
        prompt_id: str,
        prompt: str,
        *,
        n: int,
        temperature: float,
        seed: int,
    ) -> list[Generation]:
        ...


@dataclass
class ToyTabularBackend:
    """A tiny trainable policy used to exercise the complete RLVR control path.

    Each prompt has a finite action vocabulary. Logits are updated by the GRPO
    trainer, so tests and Mac/CPU demos exercise real policy improvement rather
    than a fixed mock.
    """

    actions: tuple[str, ...] = ("0", "1", "2", "3")
    logits: dict[str, list[float]] = field(default_factory=dict)
    base_latency_s: float = 0.0

    def _logits(self, prompt_id: str) -> list[float]:
        return self.logits.setdefault(prompt_id, [0.0] * len(self.actions))

    def distribution(self, prompt_id: str, temperature: float = 1.0) -> list[float]:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        z = [x / temperature for x in self._logits(prompt_id)]
        m = max(z)
        exp = [math.exp(x - m) for x in z]
        total = sum(exp)
        return [x / total for x in exp]

    async def generate(
        self,
        prompt_id: str,
        prompt: str,
        *,
        n: int,
        temperature: float,
        seed: int,
    ) -> list[Generation]:
        if n <= 0:
            raise ValueError("n must be positive")
        start = time.perf_counter()
        rng = random.Random(seed)
        probs = self.distribution(prompt_id, temperature)
        out: list[Generation] = []
        for _ in range(n):
            r = rng.random()
            acc = 0.0
            idx = len(probs) - 1
            for j, p in enumerate(probs):
                acc += p
                if r <= acc:
                    idx = j
                    break
            if self.base_latency_s:
                await asyncio.sleep(self.base_latency_s)
            response = self.actions[idx]
            out.append(
                Generation(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    response=response,
                    logprob=math.log(max(probs[idx], 1e-12)),
                    token_count=1,
                    latency_s=time.perf_counter() - start,
                    metadata={"action_index": idx, "backend": "toy-tabular"},
                )
            )
        return out

    def apply_policy_gradient(
        self,
        prompt_id: str,
        action_index: int,
        advantage: float,
        learning_rate: float,
    ) -> None:
        probs = self.distribution(prompt_id)
        logits = self._logits(prompt_id)
        for j in range(len(logits)):
            grad = (1.0 if j == action_index else 0.0) - probs[j]
            logits[j] += learning_rate * advantage * grad


@dataclass
class VLLMHTTPBackend:
    """OpenAI-compatible vLLM/SGLang serving adapter.

    This adapter intentionally speaks HTTP rather than importing vLLM so the
    repository remains runnable on macOS/CPU while the same rollout engine can
    target a remote GPU inference server.
    """

    endpoint: str
    model: str
    timeout_s: float = 120.0

    async def generate(
        self,
        prompt_id: str,
        prompt: str,
        *,
        n: int,
        temperature: float,
        seed: int,
    ) -> list[Generation]:
        return await asyncio.to_thread(
            self._generate_sync, prompt_id, prompt, n, temperature, seed
        )

    def _generate_sync(
        self, prompt_id: str, prompt: str, n: int, temperature: float, seed: int
    ) -> list[Generation]:
        url = self.endpoint.rstrip("/") + "/v1/completions"
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "n": n,
                "temperature": temperature,
                "seed": seed,
                "logprobs": 1,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        latency = time.perf_counter() - start
        choices = body.get("choices", [])
        if len(choices) != n:
            raise RuntimeError(f"expected {n} choices, got {len(choices)}")
        out = []
        for choice in choices:
            text = str(choice.get("text", ""))
            token_logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
            finite = [float(x) for x in token_logprobs if x is not None]
            out.append(
                Generation(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    response=text,
                    logprob=sum(finite),
                    token_count=max(1, len(finite)),
                    latency_s=latency,
                    metadata={"backend": "vllm-http", "model": self.model},
                )
            )
        return out
