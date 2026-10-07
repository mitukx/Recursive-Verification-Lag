"""Connect the existing durable coding agent to a real-repository operator task."""
from __future__ import annotations

import asyncio
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from .evaluator import TASK, load_task, run


@dataclass(frozen=True)
class MLSysCodingTask:
    task_id: str
    prompt: str


def coding_task():
    spec, source = load_task()
    return MLSysCodingTask(spec["task_id"], (TASK / "README.md").read_text() +
                           "\nInitial solution.py:\n```python\n" + source.decode() + "```\n")


class MLSysPublicGrader:
    """Public correctness feedback only; terminal specs belong to the evaluator.

    Pass to CodingToolAgent(public_grader). Model-generated code always runs in
    Docker. The trusted-local smoke mode is deliberately absent from this API.
    Full public reports live in evaluator storage, outside model context.
    """
    def __init__(self, image: str, artifact_root: Path):
        self.image = image
        self.artifact_root = artifact_root
        artifact_root.mkdir(parents=True, exist_ok=True)

    async def __call__(self, generation):
        if not isinstance(generation.response, str) or len(generation.response.encode()) > 32000:
            return 0.0
        def grade():
            with tempfile.TemporaryDirectory(prefix="rvl-mlsys-agent-") as tmp:
                candidate = Path(tmp) / "solution.py"
                candidate.write_text(generation.response)
                output = self.artifact_root / uuid.uuid4().hex
                report = run(candidate, output, image=self.image, correctness_only=True)
                return float(report["summary"]["correctness_passed"])
        return await asyncio.to_thread(grade)
