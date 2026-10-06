"""Evaluator-owned input/output tests; generated Python executes only in Docker."""
from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass

# Reuse the repository's pinned-image, bounded-output, disposable-container
# transport. Comparison with expected results happens OUTSIDE candidate Python.
from ...score_mbppplus_docker import docker_command, run_capped, sanitize


WORKER = r"""
import contextlib,io,json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(4,4))
request=json.load(sys.stdin)
results=[]
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
    try:
        namespace={"__name__":"__candidate__"}
        exec(compile(request["source"],"candidate.py","exec"),namespace)
        function=namespace[request["entry_point"]]
        for args in request["inputs"]:
            try:
                value=function(*args)
                # Force JSON serializability before sending to the evaluator.
                json.dumps(value,allow_nan=False)
                results.append({"ok":True,"value":value})
            except BaseException:
                results.append({"ok":False,"value":None})
    except BaseException:
        results=[{"ok":False,"value":None} for _ in request["inputs"]]
print(json.dumps({"results":results},allow_nan=False),flush=True)
"""


@dataclass(frozen=True)
class IOTest:
    arguments: tuple
    expected: object


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    prompt: str
    entry_point: str
    public: tuple[IOTest,...]
    hidden: tuple[IOTest,...]


class DockerCodingGrader:
    def __init__(self,tasks,*,image,hidden=False,platform="linux/amd64",timeout_s=10):
        self.tasks,self.image,self.hidden = tasks,image,hidden
        self.platform,self.timeout_s = platform,timeout_s
        # Validate immutable image and isolation flags before any evaluation.
        docker_command(image,"validation",WORKER,platform)
        self.last_failure = None

    async def __call__(self,generation):
        return await asyncio.to_thread(self._score,generation)

    def _score(self,generation):
        task = self.tasks[generation.prompt_id]
        tests = task.hidden if self.hidden else task.public
        if not tests:
            raise ValueError("empty executable test suite")
        name = "rvl-code-"+uuid.uuid4().hex
        # Expected labels are never included in the container request.
        payload = {"source":sanitize(generation.response),"entry_point":task.entry_point,
                   "inputs":[list(t.arguments) for t in tests]}
        command = docker_command(self.image,name,WORKER,self.platform)
        try:
            result = run_capped(command,payload,name,timeout=self.timeout_s)
            rows = result["results"]
            if not isinstance(rows,list) or len(rows)!=len(tests):
                raise ValueError("candidate output cardinality mismatch")
            passes = []
            for row,test in zip(rows,tests):
                if not isinstance(row,dict) or set(row)!={"ok","value"} or type(row["ok"]) is not bool:
                    raise ValueError("invalid candidate output")
                # JSON canonical comparison prevents True == 1 from passing.
                from .contracts import canonical
                passes.append(row["ok"] and canonical(row["value"])==canonical(test.expected))
            return sum(passes)/len(tests)
        except (RuntimeError,TimeoutError,ValueError,KeyError,TypeError):
            self.last_failure = "execution_or_protocol_failure"
            return 0.0
