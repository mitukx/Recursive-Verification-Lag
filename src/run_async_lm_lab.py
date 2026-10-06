"""Asynchronous real LM mechanics smoke; its parity reward is not a capability benchmark."""
import argparse
import asyncio
import json

from .rvl_systems.hf_backend import HFLocalBackend
from .rvl_systems.lab.lm_runtime import AsyncHFLab
from .rvl_systems.verifier import FunctionalVerifier


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",default="sshleifer/tiny-gpt2")
    p.add_argument("--output",default="artifacts/async-lm-lab")
    p.add_argument("--groups",type=int,default=4)
    p.add_argument("--samples",type=int,default=8)
    p.add_argument("--device",default="cpu")
    a = p.parse_args()
    backend = HFLocalBackend(a.model,max_new_tokens=4,device=a.device,precision="fp32")
    verifier = FunctionalVerifier(lambda g: float(g.metadata["response_token_ids"][0]%2))
    lab = AsyncHFLab(a.output,backend,verifier)
    try:
        report = asyncio.run(lab.run({f"task-{i}":"Write a short Python function:" for i in range(a.groups)},samples=a.samples))
        if not report["version"]:
            raise AssertionError("no learner updates")
        if report["parameter_l1_change"] <= 0:
            raise AssertionError("no real model parameter change")
        print(json.dumps(report,sort_keys=True))
    finally:
        lab.close()


if __name__ == "__main__":
    main()
