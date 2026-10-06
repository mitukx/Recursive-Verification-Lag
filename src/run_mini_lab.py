"""Run the persistent asynchronous CPU lab on Linux or macOS."""
import argparse
import asyncio
import json

from .rvl_systems.lab.control import ControlConfig
from .rvl_systems.lab.runtime import LabConfig, MiniLab


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="artifacts/mini-lab")
    p.add_argument("--episodes",type=int,default=256)
    p.add_argument("--actors",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--mode",choices=["adaptive","fixed","never"],default="adaptive")
    p.add_argument("--cadence",type=int,default=8)
    p.add_argument("--audit-budget",type=int,default=64)
    p.add_argument("--max-steps",type=int,default=24)
    p.add_argument("--episode-deadline-s",type=float,default=7200)
    p.add_argument("--tool-latency-ms",type=float,default=1)
    p.add_argument("--deterministic",action="store_true")
    a = p.parse_args()
    lab = MiniLab(a.output,LabConfig(episodes=a.episodes,actors=a.actors,
        batch_size=a.batch_size,seed=a.seed,max_steps=a.max_steps,
        episode_deadline_s=a.episode_deadline_s,tool_latency_s=a.tool_latency_ms/1000,
        deterministic=a.deterministic),
        ControlConfig(mode=a.mode,cadence=a.cadence,audit_budget=a.audit_budget))
    try:
        result = asyncio.run(lab.run())
        print(json.dumps({k:v for k,v in result.items() if k != "history"},sort_keys=True))
    finally:
        lab.close()


if __name__ == "__main__":
    main()
