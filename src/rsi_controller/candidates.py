from __future__ import annotations

import copy
from pathlib import PurePosixPath
from typing import Any, Mapping

from .config import MutationPolicy, digest
from .models import Candidate, Component, ImprovementProposal, RSIMode


class MutationRejected(ValueError):
    pass


class CandidateGenerator:
    """Build declarative, bounded candidates; never permits unrestricted mutation."""

    def __init__(self, policy: MutationPolicy, mode: RSIMode):
        self.policy = policy
        self.mode = mode

    def _allowed(self, component: Component) -> set[str]:
        if component == Component.HARNESS:
            return set(self.policy.allowed_harness_keys)
        if component == Component.TRAINING:
            if self.mode == RSIMode.HARNESS:
                return set()
            return set(self.policy.allowed_training_keys)
        if component == Component.VERIFIER:
            return set(self.policy.allowed_verifier_keys)
        if component == Component.POLICY:
            return set() if self.mode == RSIMode.HARNESS else {"adapter_checkpoint", "policy_checkpoint"}
        return set()

    def validate_patch_path(self, path: str) -> None:
        if "\\" in path:
            raise MutationRejected("code patch path must use repository-relative POSIX separators")
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or any(part in {"..", ""} for part in candidate.parts):
            raise MutationRejected("code patch escapes repository")
        if not self.policy.allow_code_patches:
            raise MutationRejected("code patches are disabled")
        roots = [PurePosixPath(prefix.rstrip("/")) for prefix in self.policy.code_patch_allowlist]
        if not any(root in candidate.parents for root in roots):
            raise MutationRejected(f"path is outside code-patch allowlist: {path}")

    def generate(self, proposal: ImprovementProposal, champion_state: Mapping[str, Any]) -> Candidate:
        patch = dict(proposal.patch_or_config_change)
        code_patch = patch.pop("__code_patch__", None)
        if code_patch is not None:
            if not isinstance(code_patch, dict):
                raise MutationRejected("__code_patch__ must map path to content")
            for path in code_patch:
                self.validate_patch_path(path)
        allowed = self._allowed(proposal.target_component)
        unknown = set(patch) - allowed
        if unknown:
            raise MutationRejected(f"mutation outside allowed surface: {sorted(unknown)}")

        section = {
            Component.HARNESS: "H",
            Component.TRAINING: "F",
            Component.VERIFIER: "V",
            Component.POLICY: "theta",
        }[proposal.target_component]
        state = copy.deepcopy(dict(champion_state))
        state.setdefault(section, {})
        if not isinstance(state[section], dict):
            raise MutationRejected(f"component state {section} is not mutable mapping")
        state[section].update(patch)
        candidate_id = "cand-" + digest({
            "proposal": proposal.id,
            "parent": proposal.parent_champion_id,
            "state": state,
            "seed": proposal.deterministic_seed,
        })[:12]
        return Candidate(
            candidate_id=candidate_id,
            proposal_id=proposal.id,
            parent_champion_id=proposal.parent_champion_id,
            target_component=proposal.target_component,
            diff=proposal.patch_or_config_change,
            full_state=state,
            seed=proposal.deterministic_seed,
            resource_limits=proposal.resource_limits,
            dependency_metadata=proposal.dependency_metadata,
        )
