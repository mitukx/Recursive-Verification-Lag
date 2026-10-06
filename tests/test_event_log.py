import tempfile
import unittest
from pathlib import Path

from src.rvl_systems.event_log import ControlPlaneEventLog


class EventLogTest(unittest.TestCase):
    def test_roundtrip_digest_and_validation(self):
        log = ControlPlaneEventLog()
        log.append("request_started", request_id="r1", workload_id="train")
        log.append(
            "worker_selected",
            request_id="r1",
            workload_id="train",
            worker="w0",
            attempt=1,
        )
        log.append("request_completed", request_id="r1", workload_id="train")
        digest = log.semantic_digest()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            log.write_jsonl(path)
            loaded = ControlPlaneEventLog.read_jsonl(path)

        loaded.validate()
        self.assertEqual(loaded.semantic_digest(), digest)
        self.assertEqual(loaded.summary()["requests"], 1)

    def test_rejects_worker_reuse(self):
        log = ControlPlaneEventLog()
        log.append("request_started", request_id="r", workload_id="x")
        for attempt in (1, 2):
            log.append(
                "worker_selected",
                request_id="r",
                workload_id="x",
                worker="w0",
                attempt=attempt,
            )
        log.append("request_failed", request_id="r", workload_id="x")
        with self.assertRaisesRegex(ValueError, "reused"):
            log.validate()


if __name__ == "__main__":
    unittest.main()
