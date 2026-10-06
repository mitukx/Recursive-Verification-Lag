import json
import tempfile
import unittest
from pathlib import Path

from src.rvl_systems.rlvr_benchmark import (
    build_math_prompt,
    extract_final_numeric_answer,
    gsm8k_reference_answer,
    load_jsonl_tasks,
    normalize_numeric_answer,
    response_reward,
)


class RLVRBenchmarkTest(unittest.TestCase):
    def test_extract_prefers_final_marker(self):
        text = "I considered 12 and 15. Final answer:\n#### 1,234"
        self.assertEqual(extract_final_numeric_answer(text), "1234")

    def test_numeric_normalization(self):
        self.assertEqual(normalize_numeric_answer("0012.500"), "12.5")
        self.assertEqual(normalize_numeric_answer("-2,000"), "-2000")

    def test_gsm8k_reference(self):
        answer = "Some reasoning with 5.\n#### 42"
        self.assertEqual(gsm8k_reference_answer(answer), "42")
        self.assertEqual(response_reward("work\n#### 42", "42"), 1.0)
        self.assertEqual(response_reward("work\n#### 41", "42"), 0.0)

    def test_jsonl_loader_and_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "a",
                        "prompt": build_math_prompt("What is 1+1?"),
                        "answer": "2",
                        "split": "train",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tasks = load_jsonl_tasks(path)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].answer, "2")
        self.assertIn("#### <number>", tasks[0].prompt)


if __name__ == "__main__":
    unittest.main()
