from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .types import Generation

_SHA = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN = re.compile(
    r"\b(sorry|admit|axiom|unsafe|run_tac|elab|macro_rules|open\s+System|IO\.)\b",
    re.IGNORECASE,
)
_SPLITS = {"development", "heldout"}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class OpenAIMathTask:
    task_id: str
    family: int
    title: str
    comparator_path: str
    theorem_name: str
    split: str
    mechanism: str
    blob_sha: str

    def validate(self) -> None:
        path = Path(self.comparator_path)
        if self.family <= 0:
            raise ValueError("family must be positive")
        if self.split not in _SPLITS:
            raise ValueError("invalid split")
        if (
            path.is_absolute()
            or ".." in path.parts
            or not self.comparator_path.startswith("lean/ComparatorChallenges/")
            or path.suffix != ".lean"
        ):
            raise ValueError("unsafe comparator path")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_'.]*", self.theorem_name):
            raise ValueError("invalid theorem name")
        if not _SHA.fullmatch(self.blob_sha):
            raise ValueError("invalid source blob SHA")


@dataclass(frozen=True)
class OpenAIMathManifest:
    version: int
    upstream_repo: str
    upstream_sha: str
    lean_toolchain: str
    tasks: tuple[OpenAIMathTask, ...]

    @classmethod
    def load(cls, path: str | Path) -> "OpenAIMathManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = cls(
            int(raw["version"]),
            str(raw["upstream_repo"]),
            str(raw["upstream_sha"]),
            str(raw["lean_toolchain"]),
            tuple(OpenAIMathTask(**row) for row in raw["tasks"]),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.version != 1 or self.upstream_repo != "openai/math":
            raise ValueError("unsupported manifest")
        if not _SHA.fullmatch(self.upstream_sha):
            raise ValueError("invalid upstream SHA")
        if not self.lean_toolchain.startswith("leanprover/lean4:"):
            raise ValueError("unexpected Lean toolchain")
        ids: set[str] = set()
        paths: set[str] = set()
        family_split: dict[int, str] = {}
        for task in self.tasks:
            task.validate()
            if task.task_id in ids or task.comparator_path in paths:
                raise ValueError("duplicate task or comparator path")
            prior = family_split.setdefault(task.family, task.split)
            if prior != task.split:
                raise ValueError(f"family {task.family} crosses split boundary")
            ids.add(task.task_id)
            paths.add(task.comparator_path)
        if {task.split for task in self.tasks} != _SPLITS:
            raise ValueError("manifest must contain development and heldout tasks")

    def select(self, split: str) -> "OpenAIMathManifest":
        if split not in _SPLITS:
            raise ValueError("unknown split")
        return OpenAIMathManifest(
            self.version,
            self.upstream_repo,
            self.upstream_sha,
            self.lean_toolchain,
            tuple(task for task in self.tasks if task.split == split),
        )

    def task_map(self) -> dict[str, OpenAIMathTask]:
        return {task.task_id: task for task in self.tasks}

    @property
    def fingerprint(self) -> str:
        return sha256_text(
            canonical(
                {
                    "version": self.version,
                    "upstream_repo": self.upstream_repo,
                    "upstream_sha": self.upstream_sha,
                    "lean_toolchain": self.lean_toolchain,
                    "tasks": [asdict(task) for task in self.tasks],
                }
            )
        )


@dataclass(frozen=True)
class LeanCheckResult:
    task_id: str
    passed: bool
    returncode: int
    latency_s: float
    source_sha256: str
    candidate_sha256: str
    source_blob_sha: str
    stdout_tail: str
    stderr_tail: str


class OpenAIMathCheckout:
    def __init__(self, root: str | Path, manifest: OpenAIMathManifest):
        self.root = Path(root).resolve()
        self.manifest = manifest

    def _git(self, *args: str) -> str:
        proc = subprocess.run(
            ("git", "-C", str(self.root), *args),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if proc.returncode:
            raise RuntimeError(proc.stderr.strip() or "git provenance check failed")
        return proc.stdout.strip()

    def verify(self) -> None:
        if self._git("rev-parse", "HEAD") != self.manifest.upstream_sha:
            raise ValueError("OpenAI Math checkout does not match pinned commit")
        if self._git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError("OpenAI Math checkout has tracked modifications")
        toolchain = (self.root / "lean" / "lean-toolchain").read_text().strip()
        if toolchain != self.manifest.lean_toolchain:
            raise ValueError("Lean toolchain drift")
        for task in self.manifest.tasks:
            blob = self._git("rev-parse", f"HEAD:{task.comparator_path}")
            if blob != task.blob_sha:
                raise ValueError(f"source drift for {task.task_id}")


class LeanTrustedVerifier:
    """Exact Lean-kernel reward for pinned OpenAI Math Comparator challenges."""

    def __init__(
        self,
        checkout: str | Path,
        manifest: OpenAIMathManifest,
        *,
        timeout_s: float = 180.0,
        max_candidate_bytes: int = 131072,
        max_output_bytes: int = 32768,
        sandbox_prefix: tuple[str, ...] = (),
        allow_unsandboxed: bool = False,
        verify_provenance: bool = True,
    ):
        if not sandbox_prefix and not allow_unsandboxed:
            raise ValueError("external sandbox required for untrusted Lean candidate")
        self.checkout = OpenAIMathCheckout(checkout, manifest)
        self.manifest = manifest
        self.tasks = manifest.task_map()
        self.timeout_s = timeout_s
        self.max_candidate_bytes = max_candidate_bytes
        self.max_output_bytes = max_output_bytes
        self.sandbox_prefix = tuple(sandbox_prefix)
        if verify_provenance:
            self.checkout.verify()

    def source(self, task_id: str) -> str:
        return (self.checkout.root / self.tasks[task_id].comparator_path).read_text(
            encoding="utf-8"
        )

    @staticmethod
    def inject_candidate(source: str, theorem_name: str, candidate: str) -> str:
        holes = list(re.finditer(r"\bsorry\b", source))
        if len(holes) != 1:
            raise ValueError(f"strict task requires one proof hole, found {len(holes)}")
        theorem = re.search(
            rf"(?m)^\s*theorem\s+{re.escape(theorem_name)}(?:\b|[.{{])", source
        )
        if theorem is None or theorem.start() > holes[0].start():
            raise ValueError("target theorem does not own proof hole")
        hole = holes[0]
        return source[: hole.start()] + candidate + source[hole.end() :]

    def prompt(self, task_id: str) -> str:
        task = self.tasks[task_id]
        source = self.source(task_id)
        holes = list(re.finditer(r"\bsorry\b", source))
        if len(holes) != 1:
            raise ValueError("not a strict single-hole task")
        visible = source[: holes[0].start()] + "-- PROOF_HOLE" + source[holes[0].end() :]
        return (
            f"OpenAI Math family {task.family}: {task.title}\n"
            f"Target theorem: {task.theorem_name}\n"
            "Return only Lean tactics replacing PROOF_HOLE after := by. "
            "No sorry, admit, axioms, unsafe code, metaprogramming, or IO.\n\n"
            + visible
        )

    def verify_candidate(self, task_id: str, candidate: str) -> LeanCheckResult:
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError("empty candidate")
        if len(candidate.encode("utf-8")) > self.max_candidate_bytes:
            raise ValueError("candidate too large")
        if _FORBIDDEN.search(candidate):
            raise ValueError("forbidden proof escape")
        task = self.tasks[task_id]
        source = self.source(task_id)
        materialized = self.inject_candidate(source, task.theorem_name, candidate.strip())
        started = time.perf_counter()

        with tempfile.TemporaryDirectory(prefix="rvl-oai-math-") as tmp:
            path = Path(tmp) / f"{task_id}.lean"
            path.write_text(materialized, encoding="utf-8")
            command = (*self.sandbox_prefix, "lake", "env", "lean", str(path))
            env = dict(os.environ)
            env["LEAN_ABORT_ON_PANIC"] = "1"
            try:
                proc = subprocess.run(
                    command,
                    cwd=self.checkout.root / "lean",
                    env=env,
                    capture_output=True,
                    check=False,
                    timeout=self.timeout_s,
                )
                code = int(proc.returncode)
                stdout = proc.stdout[-self.max_output_bytes :].decode("utf-8", "replace")
                stderr = proc.stderr[-self.max_output_bytes :].decode("utf-8", "replace")
            except subprocess.TimeoutExpired as exc:
                code = 124
                stdout = (exc.stdout or b"")[-self.max_output_bytes :].decode(
                    "utf-8", "replace"
                )
                stderr = (exc.stderr or b"")[-self.max_output_bytes :].decode(
                    "utf-8", "replace"
                ) + "\nverifier timeout"

        passed = code == 0 and "declaration uses 'sorry'" not in stdout + stderr
        return LeanCheckResult(
            task_id,
            passed,
            code,
            time.perf_counter() - started,
            sha256_text(source),
            sha256_text(candidate.strip()),
            task.blob_sha,
            stdout,
            stderr,
        )


class LeanGenerationGrader:
    """Async trusted-grader adapter for MultiVerifier."""

    def __init__(self, verifier: LeanTrustedVerifier):
        self.verifier = verifier
        self.calls = 0
        self.last_results: dict[str, LeanCheckResult] = {}

    async def __call__(self, generation: Generation) -> float:
        result = await asyncio.to_thread(
            self.verifier.verify_candidate, generation.prompt_id, generation.response
        )
        self.calls += 1
        self.last_results[generation.prompt_id] = result
        return float(result.passed)


def summarize_rows(rows: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold outside [0,1]")
    n = len(rows)
    if not n:
        return {"n": 0}
    proxy = [float(row["proxy_score"]) for row in rows]
    truth = [float(bool(row["trusted_pass"])) for row in rows]
    positive = [i for i, score in enumerate(proxy) if score >= threshold]
    false_pos = sum(1 for i in positive if truth[i] == 0)
    by_task: dict[str, list[dict[str, Any]]] = {}
    by_age: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), []).append(row)
        age = int(row.get("stale_age", 0))
        bucket = "0" if age == 0 else ("1-2" if age <= 2 else ("3-7" if age <= 7 else "8+"))
        by_age.setdefault(bucket, []).append(row)
    false_progress = transitions = 0
    for group in by_task.values():
        ordered = sorted(group, key=lambda r: (int(r.get("policy_version", 0)), int(r.get("sequence", 0))))
        for before, after in zip(ordered, ordered[1:]):
            transitions += 1
            if (
                float(after["proxy_score"]) > float(before["proxy_score"])
                and bool(before["trusted_pass"])
                and not bool(after["trusted_pass"])
            ):
                false_progress += 1

    def rate(group: list[dict[str, Any]]) -> float:
        return sum(bool(row["trusted_pass"]) for row in group) / len(group)

    return {
        "n": n,
        "proxy_mean": sum(proxy) / n,
        "trusted_pass_rate": sum(truth) / n,
        "calibration_mae": sum(abs(a - b) for a, b in zip(proxy, truth)) / n,
        "proxy_positive_n": len(positive),
        "false_positive_n": false_pos,
        "false_positive_rate": false_pos / len(positive) if positive else 0.0,
        "evaluated_transitions": transitions,
        "false_progress_transitions": false_progress,
        "false_progress_transition_rate": false_progress / transitions if transitions else 0.0,
        "trusted_pass_rate_by_stale_age": {
            name: rate(group) for name, group in sorted(by_age.items())
        },
    }


def hash_chain(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    head = "0" * 64
    out = []
    for row in rows:
        payload = dict(row)
        payload["previous_hash"] = head
        head = sha256_text(head + canonical(row))
        payload["row_hash"] = head
        out.append(payload)
    return out, head
