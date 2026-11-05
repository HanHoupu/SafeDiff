"""CPU-only contract checks: no model, CUDA, or numerical generation tests."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analysis.run_acceleration import (
    ResultWriter, build_parser, extract_prompt, load_records, run_jobs,
)


class FakeBackend:
    instances = []

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.options = kwargs
        self.calls = []
        self.closed = False
        self.__class__.instances.append(self)

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"response": "fixture", "config": kwargs, "provenance": {"fixture": True}}

    def close(self):
        self.closed = True


class FailingBackend(FakeBackend):
    def generate(self, prompt, **kwargs):
        if kwargs["mode"] == "fastdllm":
            raise RuntimeError("fixture failure")
        return super().generate(prompt, **kwargs)


class IncompleteBackend(FakeBackend):
    def generate(self, prompt, **kwargs):
        result = super().generate(prompt, **kwargs)
        if kwargs["mode"] == "instant":
            result.update(residual_masks=1, completion_status="residual_masks",
                          generated_token_ids=[126336])
        return result


class InputContractTests(unittest.TestCase):
    def test_describe_config_does_not_import_model_dependencies(self):
        script = (
            "import sys; from acceleration import describe_config; "
            "describe_config('instant'); describe_config('fastdllm'); "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        )
        completed = subprocess.run([sys.executable, "-c", script],
                                   cwd=Path(__file__).resolve().parents[1],
                                   text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_existing_prompt_schemas_and_conflicts(self):
        for field in ("text", "prompt", "vanilla prompt"):
            self.assertEqual(extract_prompt({field: " hello \n"}), " hello \n")
        self.assertEqual(extract_prompt({"text": " hello ", "prompt": " hello "}), " hello ")
        for invalid in ({}, {"text": "  "}, {"prompt": None},
                        {"text": "hello", "prompt": "other"},
                        {"text": "hello", "prompt": " hello "},
                        {"text": "hello", "prompt": ""}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                extract_prompt(invalid)

    def test_json_and_jsonl_preserve_metadata(self):
        source = [{"id": "a", "text": "hello", "risk_tier": "HIGH"},
                  {"id": "b", "vanilla prompt": "second"}]
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".json", ".jsonl"):
                path = Path(directory) / ("inputs" + suffix)
                text = json.dumps(source) if suffix == ".json" else "\n".join(map(json.dumps, source))
                path.write_text(text, encoding="utf-8")
                records = load_records(input_path=str(path))
                self.assertEqual([r["prompt"] for r in records], ["hello", "second"])
                self.assertEqual(records[0]["source_record"]["risk_tier"], "HIGH")

    def test_all_inputs_validated_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('[{"prompt":"valid"},{"prompt":""}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "record 1"):
                load_records(input_path=str(path))


class RunContractTests(unittest.TestCase):
    def setUp(self):
        FakeBackend.instances.clear()

    def test_compare_shares_loader_and_generation_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            args = build_parser().parse_args([
                "--mode", "compare", "--prompt", "question", "--output", str(output),
                "--gen-length", "64", "--seed", "7", "--system-prompt", "Be brief.",
            ])
            self.assertEqual(run_jobs(args, load_records(prompt=args.prompt), backend_factory=FakeBackend), 0)
            self.assertEqual(len(FakeBackend.instances), 1)
            backend = FakeBackend.instances[0]
            self.assertTrue(backend.closed)
            self.assertTrue(backend.options["local_files_only"])
            self.assertEqual([call[1]["mode"] for call in backend.calls], ["instant", "fastdllm"])
            normalized = [{k: v for k, v in options.items() if k != "mode"}
                          for _, options in backend.calls]
            self.assertEqual(normalized[0], normalized[1])
            self.assertEqual([p for p, _ in backend.calls], ["question", "question"])
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["mode"] for r in rows], ["instant", "fastdllm"])
            self.assertEqual(rows[0]["run_id"], rows[1]["run_id"])
            self.assertNotIn("safety_score", rows[0])

    def test_failure_keeps_completed_result_and_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            args = build_parser().parse_args(["--mode", "compare", "--output", str(output)])
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                run_jobs(args, load_records(prompt="question"), backend_factory=FailingBackend)
            rows = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([r["status"] for r in rows], ["ok", "error"])
            self.assertEqual(rows[1]["error"]["type"], "RuntimeError")
            self.assertTrue(FakeBackend.instances[-1].closed)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_residual_masks_are_saved_and_comparison_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            args = build_parser().parse_args(["--mode", "compare", "--output", str(output)])
            records = load_records(prompt=" question \n")
            records.append({"index": 1, "id": 1, "prompt": "second",
                            "source_record": {"prompt": "second"}})
            code = run_jobs(args, records, backend_factory=IncompleteBackend)
            self.assertEqual(code, 2)
            rows = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([r["status"] for r in rows], ["incomplete", "ok", "incomplete", "ok"])
            self.assertEqual(rows[0]["generation"]["generated_token_ids"], [126336])
            self.assertEqual(rows[0]["prompt"], " question \n")
            self.assertEqual([prompt for prompt, _ in FakeBackend.instances[-1].calls],
                             [" question \n", " question \n", "second", "second"])
            self.assertTrue(FakeBackend.instances[-1].closed)

    def test_existing_output_rejected_before_backend_load(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            output.write_text("original", encoding="utf-8")
            args = build_parser().parse_args(["--output", str(output)])
            with self.assertRaises(FileExistsError):
                run_jobs(args, load_records(prompt="question"), backend_factory=FakeBackend)
            self.assertEqual(output.read_text(encoding="utf-8"), "original")
            self.assertFalse(FakeBackend.instances)

    def test_json_snapshot_is_readable_after_each_record(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            writer = ResultWriter(str(output))
            try:
                for i in range(2):
                    writer.write({"index": i})
                    self.assertEqual(len(json.loads(output.read_text(encoding="utf-8"))), i + 1)
            finally:
                writer.close()


if __name__ == "__main__":
    unittest.main()
