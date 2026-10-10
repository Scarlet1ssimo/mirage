"""CPU checks for validated/invalid selection and aggregate diagnostics."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from reproduce_superoptimizer import select, summarize, write_json


def observation(latency, valid, status=None):
    return {
        "status": status or ("ok" if valid else "measured_invalid"),
        "latency_ms": {"median": latency},
        "effective_tflops": 1 / latency,
        "correctness": {"passed": valid},
        "post_validation": {"passed": valid},
    }


class ReproductionChecks(unittest.TestCase):
    def test_fastest_invalid_does_not_replace_validated_winner(self):
        bad, good = observation(1, False), observation(2, True)
        winner, fastest = select([bad, good])
        self.assertIs(winner, good)
        self.assertIs(fastest, bad)

    def test_missing_postvalidation_or_failed_run_cannot_win(self):
        incomplete = observation(1, True)
        incomplete.pop("post_validation")
        failed = observation(0.5, True, "failed")
        self.assertEqual(select([incomplete, failed]), (None, incomplete))
        invalid_timing = observation(1, True)
        invalid_timing["latency_ms"]["median"] = float("nan")
        self.assertEqual(select([invalid_timing]), (None, None))

    def test_aggregate_preserves_invalid_speed_and_timeout_without_a_winner(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            case = out / "cases/attention/B1"
            measured = case / "cuda-0"
            timeout = case / "triton-0"
            measured.mkdir(parents=True)
            timeout.mkdir()
            candidate = case / "candidate-0.json"
            write_json(candidate, [])
            write_json(
                case / "case.json",
                {
                    "suite": "ae",
                    "case": "attention",
                    "B": 1,
                    "source": "cache",
                    "status": "timeout",
                    "fp32_accum": False,
                    "fold_zero_epsilon": False,
                    "gpu_uuid": "test-gpu",
                    "candidate_count": 1,
                },
            )
            bad = observation(1, False)
            bad.update(
                candidate_graph=str(candidate), backend="cuda", candidate_index=0
            )
            bad["correctness"].update(
                output_nan=4, output_elements=4, relative_l2=float("nan")
            )
            write_json(measured / "result.json", bad)
            write_json(
                timeout / "result.json", {"status": "timeout", "stage": "compile"}
            )
            rows = summarize(out)
            self.assertEqual(rows[0]["fastest_timed_tflops"], 1)
            self.assertIsNone(rows[0]["validated_tflops"])
            self.assertFalse(rows[0]["fastest_timed_valid"])
            self.assertEqual(rows[0]["output_nan"], 4)
            self.assertIsNone(rows[0]["relative_l2"])
            self.assertFalse((case / "winner").exists())
            self.assertTrue((case / "fastest_timed").exists())
            self.assertEqual(len(json.loads((out / "attempts.json").read_text())), 2)
            summarize(out)
            self.assertEqual(len(json.loads((out / "attempts.json").read_text())), 2)
            with (out / "performance.csv").open() as file:
                self.assertEqual(len(list(csv.DictReader(file))), 1)
            failures = json.loads((out / "failure-distribution.json").read_text())
            self.assertEqual(failures["attempt_statuses"]["timeout"], 1)


if __name__ == "__main__":
    unittest.main()
