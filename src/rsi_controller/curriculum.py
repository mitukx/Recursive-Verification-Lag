from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    difficulty: int
    requires_tools: bool = False
    multi_file: bool = False


DEFAULT_STAGES = (
    CurriculumStage("simple_code_completion", 1),
    CurriculumStage("bug_fixing", 2),
    CurriculumStage("tests", 3),
    CurriculumStage("multi_step_debugging", 4, requires_tools=True),
    CurriculumStage("optimization", 5, requires_tools=True),
    CurriculumStage("multi_file_modification", 6, requires_tools=True, multi_file=True),
    CurriculumStage("tool_use_tasks", 7, requires_tools=True),
    CurriculumStage("agent_harness_improvement", 8, requires_tools=True, multi_file=True),
)


class BoundedCurriculum:
    """Selects a validated stage just above reliable competence.

    Task generation and authoritative answer production are deliberately separate:
    generated tasks must pass `validator`, and this class never supplies labels.
    """

    def __init__(self, stages: Sequence[CurriculumStage] = DEFAULT_STAGES, reliability_target: float = 0.78):
        if not stages or not 0 < reliability_target < 1:
            raise ValueError("invalid curriculum")
        self.stages = tuple(stages)
        self.reliability_target = reliability_target

    def select_stage(self, performance_by_stage: dict[str, float]) -> CurriculumStage:
        reliable = [s for s in self.stages if performance_by_stage.get(s.name, 0.0) >= self.reliability_target]
        if not reliable:
            return self.stages[0]
        level = min(len(self.stages) - 1, max(s.difficulty for s in reliable))
        # difficulty is 1-indexed and stages are ordered.
        return self.stages[level]

    def validate_generated_task(self, task: object, validator: Callable[[object], bool]) -> bool:
        return bool(validator(task))
