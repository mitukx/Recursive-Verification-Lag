"""Paired-seed adaptive/fixed/never interventions, with explicit label cost."""
import argparse
import asyncio
import json
import statistics
from pathlib import Path

from .rvl_systems.lab.control import ControlConfig
from .rvl_systems.lab.runtime import LabConfig, MiniLab


async def benchmark(root, seeds, episodes):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    runs = []
    for seed in seeds:
        for mode in ("adaptive","fixed","never"):
            output = root/f"{mode}-seed-{seed}"
            if (output/"replay.sqlite").exists():
                raise ValueError(f"benchmark requires a fresh directory: {output}")
            cfg = LabConfig(episodes=episodes,actors=1,batch_size=8,
                            seed=seed,tool_latency_s=0,deterministic=True)
            control = ControlConfig(mode=mode,cadence=2,audit_budget=32,
                                    audit_per_batch=2,refit_labels=4,risk_threshold=.30)
            lab = MiniLab(output,cfg,control)
            try:
                r = await lab.run()
            finally:
                lab.close()
            runs.append({"seed":seed,"mode":mode,"final_reward":r["final_eval_reward"],
                         "initial_reward":r["initial_eval_reward"],"audits":r["trusted_calls"],
                         "verifier_version":r["history"][-1]["verifier_version"],
                         "exploits":r["exploits_detected"],"elapsed_s":r["elapsed_s"],
                         "semantic_sha256":r["semantic_sha256"],
                         "p99_s":r["latency_s"]["p99"]})
    pairs = []
    for seed in seeds:
        a = next(r for r in runs if r["seed"]==seed and r["mode"]=="adaptive")
        f = next(r for r in runs if r["seed"]==seed and r["mode"]=="fixed")
        pairs.append({"seed":seed,"reward_difference":a["final_reward"]-f["final_reward"],
                      "adaptive_audits":a["audits"],"fixed_audits":f["audits"],
                      "cost_matched":a["audits"]==f["audits"]})
    differences = [p["reward_difference"] for p in pairs]
    report = {"schema":1,"experiment":"CPU restricted-DSL RVL intervention ablation",
              "seeds":seeds,"episodes":episodes,"runs":runs,"paired_comparisons":pairs,
              "mean_paired_difference":statistics.fmean(differences),
              "paired_standard_error":statistics.stdev(differences)/len(differences)**.5 if len(differences)>1 else None,
              "all_cost_matched":all(p["cost_matched"] for p in pairs),
              "claims":["reported differences may be negative or zero",
                        "reused online gate suite is not an independent final capability benchmark",
                        "no inference from this DSL to pretrained LM scaling"]}
    (root/"comparison.json").write_text(json.dumps(report,indent=2,sort_keys=True))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="artifacts/mini-lab-ablation")
    p.add_argument("--seeds",default="17,29,43")
    p.add_argument("--episodes",type=int,default=256)
    a = p.parse_args()
    print(json.dumps(asyncio.run(benchmark(a.output,[int(x) for x in a.seeds.split(",")],a.episodes)),sort_keys=True))


if __name__ == "__main__":
    main()
