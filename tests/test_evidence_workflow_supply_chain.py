import re
import unittest
from pathlib import Path


PINNED_WORKFLOWS = (
    ".github/workflows/rvl-qwen-alignment-bridge.yml",
    ".github/workflows/rvl-qwen-learned-verifier.yml",
    ".github/workflows/rvl-real-llm-active-audit.yml",
    ".github/workflows/qwen-bridge-contract.yml",
    ".github/workflows/qwen-learned-verifier-contract.yml",
    ".github/workflows/real-llm-active-audit-contract.yml",
    ".github/workflows/rsi-controller.yml",
)
EVIDENCE_WORKFLOWS = PINNED_WORKFLOWS[:3]
ACTION = re.compile(
    r"uses:\s+(actions/(?:checkout|setup-python|upload-artifact|download-artifact))@([^\s#]+)"
)
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


class EvidenceWorkflowSupplyChainTest(unittest.TestCase):
    def test_first_party_actions_are_immutable_commit_pins(self):
        for workflow in PINNED_WORKFLOWS:
            text = Path(workflow).read_text()
            matches = ACTION.findall(text)
            self.assertTrue(matches, workflow)
            for action, ref in matches:
                self.assertRegex(ref, FULL_SHA, f"{workflow}: {action}@{ref}")

    def test_real_evidence_workflows_retain_trigger_provenance(self):
        for workflow in EVIDENCE_WORKFLOWS:
            text = Path(workflow).read_text()
            self.assertIn("provenance/workflow.json", text, workflow)
            self.assertIn('"trigger_sha"', text, workflow)
            self.assertIn('"run_id"', text, workflow)
            self.assertIn('"action_pins"', text, workflow)


if __name__ == "__main__":
    unittest.main()
