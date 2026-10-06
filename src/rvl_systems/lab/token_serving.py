"""Strict vLLM completion adapter for token-exact distributed training replay."""
from __future__ import annotations

import asyncio
import json
import math
import time
import urllib.request
from dataclasses import dataclass

from ..types import Generation


@dataclass
class TokenServingBackend:
    endpoint: str
    model: str
    max_new_tokens: int = 256
    timeout_s: float = 120
    max_bytes: int = 8*1024*1024

    async def generate(self,prompt_id,prompt,*,n,temperature,seed):
        return await asyncio.to_thread(self._generate,prompt_id,prompt,n,temperature,seed)

    def _generate(self,prompt_id,prompt,n,temperature,seed):
        if n <= 0 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("invalid serving request")
        payload = {"model":self.model,"prompt":prompt,"n":n,"temperature":temperature,
                   "seed":seed,"logprobs":1,"return_token_ids":True,
                   "max_tokens":self.max_new_tokens,"top_k":-1,"top_p":1.0}
        request = urllib.request.Request(self.endpoint.rstrip("/")+"/v1/completions",
            data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
        start = time.perf_counter()
        with urllib.request.urlopen(request,timeout=self.timeout_s) as response:
            raw = response.read(self.max_bytes+1)
        if len(raw)>self.max_bytes:
            raise ValueError("oversized token serving response")
        return self.parse(json.loads(raw),prompt_id,prompt,n,time.perf_counter()-start)

    def parse(self,body,prompt_id,prompt,n,elapsed_s):
        if body.get("model") != self.model:
            raise ValueError("serving model identity mismatch")
        prompt_ids = body.get("prompt_token_ids")
        if not isinstance(prompt_ids,list) or not prompt_ids or not all(type(x) is int and x >= 0 for x in prompt_ids):
            raise ValueError("server must return exact prompt token IDs; client retokenization is forbidden")
        choices = body.get("choices",[])
        if len(choices)!=n or sorted(c.get("index",-1) for c in choices)!=list(range(n)):
            raise ValueError("response choice cardinality/index mismatch")
        out = []
        for choice in sorted(choices,key=lambda c:c["index"]):
            ids = choice.get("token_ids")
            logps = (choice.get("logprobs") or {}).get("token_logprobs")
            if not isinstance(ids,list) or not ids or not all(type(x) is int and x >= 0 for x in ids):
                raise ValueError("missing response token IDs")
            if not isinstance(logps,list) or len(ids)!=len(logps):
                raise ValueError("behavior token/logprob alignment mismatch")
            if not all(type(x) in (int,float) and math.isfinite(x) and x <= 0 for x in logps):
                raise ValueError("invalid behavior logprob")
            out.append(Generation(prompt_id,prompt,choice["text"],sum(logps),len(ids),elapsed_s,
                       {"backend":"token-exact-http","model":self.model,
                        "prompt_token_ids":prompt_ids,"response_token_ids":ids,
                        "response_token_logprobs":logps}))
        return out
