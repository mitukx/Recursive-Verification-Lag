from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


@dataclass(frozen=True)
class RLVRTask:
    task_id: str
    prompt: str
    answer: str
    split: str


def normalize_numeric_answer(value: str) -> str | None:
    cleaned = value.strip().replace(",", "")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))
    normalized = format(number.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def extract_final_numeric_answer(text: str) -> str | None:
    marker = re.findall(
        r"####\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
        text,
    )
    if marker:
        return normalize_numeric_answer(marker[-1])
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    return normalize_numeric_answer(matches[-1])


def gsm8k_reference_answer(answer_text: str) -> str:
    extracted = extract_final_numeric_answer(answer_text)
    if extracted is None:
        raise ValueError("GSM8K reference answer has no numeric final answer")
    return extracted


def build_math_prompt(question: str) -> str:
    return (
        "Solve the following math problem carefully. "
        "End your response with a final line exactly in the form "
        "'#### <number>'.\n\n"
        f"Problem: {question.strip()}"
    )


def response_reward(response: str, expected: str) -> float:
    predicted = extract_final_numeric_answer(response)
    return float(predicted is not None and predicted == expected)


def load_jsonl_tasks(path: str | Path) -> list[RLVRTask]:
    tasks: list[RLVRTask] = []
    for index, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        task_id = str(raw.get("id", f"task-{index}"))
        prompt = str(raw["prompt"])
        answer = str(raw["answer"])
        split = str(raw.get("split", "train"))
        tasks.append(
            RLVRTask(
                task_id=task_id,
                prompt=prompt,
                answer=answer,
                split=split,
            )
        )
    if not tasks:
        raise ValueError("task file is empty")
    return tasks


def load_gsm8k_tasks(
    *,
    train_limit: int,
    eval_limit: int,
    seed: int,
    revision: str | None = None,
) -> tuple[list[RLVRTask], list[RLVRTask]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "GSM8K loading requires the optional 'datasets' package"
        ) from exc

    if train_limit <= 0 or eval_limit <= 0:
        raise ValueError("train_limit and eval_limit must be positive")

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        revision=revision,
    )
    train_raw = dataset["train"].shuffle(seed=seed)
    eval_raw = dataset["test"].shuffle(seed=seed + 1)

    train = [
        RLVRTask(
            task_id=f"gsm8k-train-{index}",
            prompt=build_math_prompt(str(row["question"])),
            answer=gsm8k_reference_answer(str(row["answer"])),
            split="train",
        )
        for index, row in enumerate(
            train_raw.select(range(min(train_limit, len(train_raw))))
        )
    ]
    evaluation = [
        RLVRTask(
            task_id=f"gsm8k-test-{index}",
            prompt=build_math_prompt(str(row["question"])),
            answer=gsm8k_reference_answer(str(row["answer"])),
            split="eval",
        )
        for index, row in enumerate(
            eval_raw.select(range(min(eval_limit, len(eval_raw))))
        )
    ]
    return train, evaluation
