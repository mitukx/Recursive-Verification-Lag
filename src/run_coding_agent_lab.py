"""Model-driven asynchronous coding RL with isolated executable rewards."""
import argparse
import asyncio
import ast
import json

from .rvl_systems.hf_backend import HFLocalBackend
from .rvl_systems.lab.coding import CodingTask,DockerCodingGrader,IOTest
from .rvl_systems.lab.coding_lm import CodingAsyncHFLab
from .rvl_systems.lab.judges import CalibratedMultiVerifier
from .rvl_systems.types import Generation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True)
    p.add_argument("--image",required=True,help="Pinned Docker image ID, never a mutable tag")
    p.add_argument("--platform",choices=["linux/arm64","linux/amd64"],default="linux/amd64")
    p.add_argument("--device",default="cpu")
    p.add_argument("--output",default="artifacts/coding-agent-lab")
    p.add_argument("--samples",type=int,default=4)
    p.add_argument("--max-steps",type=int,default=32)
    p.add_argument("--deadline-s",type=float,default=7200)
    a = p.parse_args()
    tasks = {}
    for i in range(8):
        k,b = i%3+1,i
        task = CodingTask(f"affine-{i}",f"Implement f(x)={k}*x+{b} in Python as def f(x).",
                          "f",(IOTest((0,),b),),(IOTest((-3,),-3*k+b),IOTest((7,),7*k+b)))
        tasks[task.task_id] = task
    public = DockerCodingGrader(tasks,image=a.image,platform=a.platform)
    hidden = DockerCodingGrader(tasks,image=a.image,hidden=True,platform=a.platform)
    def candidate(g):
        return Generation(g.metadata["coding_task_id"],tasks[g.metadata["coding_task_id"]].prompt,
                          g.metadata["final_source"],0,0,0)
    async def public_score(g):
        return await public(candidate(g)) if g.metadata["completed"] else 0.0
    async def trusted_score(g):
        return await hidden(candidate(g)) if g.metadata["completed"] else 0.0
    def feature(g):
        try:
            tree = ast.parse(g.metadata["final_source"])
            # Deliberately coarse, auditable calibrator feature cells.
            return "ast:"+str(sum(isinstance(n,ast.BinOp) for n in ast.walk(tree)))
        except SyntaxError:
            return "invalid-source"
    verifier = CalibratedMultiVerifier({"public_tests":public_score},trusted=trusted_score,feature=feature)
    backend = HFLocalBackend(a.model,max_new_tokens=128,device=a.device,precision="auto")
    lab = CodingAsyncHFLab(a.output,backend,verifier,tasks,public,max_steps=a.max_steps,deadline_s=a.deadline_s)
    try:
        report = asyncio.run(lab.run_tasks(samples=a.samples))
        print(json.dumps(report,sort_keys=True))
    finally:
        lab.close()


if __name__ == "__main__":
    main()
