"""CPU-only contract checks: no model, CUDA, or numerical generation tests."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from analysis.run_acceleration import (
    ResultWriter, build_parser, extract_prompt, load_records, main, run_jobs,
)


class FakeBackend:
    instances = []

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.options = kwargs
        self.calls = []
        self.policy_calls = []
        self.closed = False
        self.__class__.instances.append(self)

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"response": "fixture", "config": kwargs, "provenance": {"fixture": True}}

    def generate_for_policy(self, prompt, *, policy, **kwargs):
        from acceleration import mode_for_policy
        mode = mode_for_policy(policy)
        self.policy_calls.append((prompt, policy, kwargs))
        return {**self.generate(prompt, mode=mode, **kwargs), "policy": policy, "mode": mode}

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
            "import sys; from acceleration import describe_config, mode_for_policy; "
            "describe_config('instant'); describe_config('fastdllm'); describe_config('llada'); "
            "assert mode_for_policy('SAFE') == 'llada'; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        )
        completed = subprocess.run([sys.executable, "-B", "-c", script],
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


class PolicyContractTests(unittest.TestCase):
    def test_exact_policy_mapping_and_invalid_inputs(self):
        from acceleration import MODES, POLICIES, POLICY_TO_MODE, mode_for_policy
        self.assertEqual(MODES, ("instant", "fastdllm", "llada"))
        self.assertEqual(POLICIES, ("FAST", "NORMAL", "SAFE"))
        self.assertEqual(POLICY_TO_MODE, {"FAST": "instant", "NORMAL": "fastdllm", "SAFE": "llada"})
        for policy, mode in POLICY_TO_MODE.items():
            self.assertEqual(mode_for_policy(policy), mode)
        for policy in (None, 1, True, [], {}, "", "fast", "SAFE ", "UNKNOWN"):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                mode_for_policy(policy)

    def test_legacy_config_dicts_are_unchanged(self):
        # Captured before adding the third mode; protects the measured baselines.
        from acceleration import describe_config
        frozen = {
            "instant": {64: "c362507ff479cc4ccc4c9d3734bb96acf689b27847d72144bf1de817820402b6",
                        256: "3b1972de01450d9b2e248dc6c008a20490f1c38ba97cb5df564f1465ecb35acd",
                        288: "a878d12332f1ff9cf98037024a7f7c5bb326f85c4f4a9b0a702a14af8f7c2f01"},
            "fastdllm": {64: "2bf81852bd324388d9c260a221b0b167fcdf3e09ba858b9c5364d1111c3178e8",
                         256: "c5e5454c3ca88e2c71f82082adc0ff5b3313b764b4a6a9d56fa7e925924d7d6b",
                         288: "287bc83642d79637859c5dbd32f83930a2b20d30899f438d0a4c3f987af2216e"},
        }
        for mode, lengths in frozen.items():
            for length, expected in lengths.items():
                with self.subTest(mode=mode, length=length):
                    data = json.dumps(describe_config(mode, length, 42), sort_keys=True, separators=(",", ":")).encode()
                    self.assertEqual(hashlib.sha256(data).hexdigest(), expected)

    def test_llada_is_vanilla_without_acceleration(self):
        from acceleration import describe_config
        config = describe_config("llada", gen_length=64, seed=7)
        self.assertEqual(config["mode"], "llada")
        self.assertEqual(config["generator"], "official_llada_vanilla")
        self.assertEqual(config["steps"], 64)
        self.assertEqual(config["block_length"], 64)
        self.assertEqual(config["temperature"], 0)
        self.assertEqual(config["cfg_scale"], 0)
        self.assertFalse(config["dual_cache"])
        self.assertFalse(config["use_cache"])
        self.assertIsNone(config["threshold"])
        self.assertIsNone(config["berm_mu"])
        self.assertFalse(config["eos_early_stop"])
        self.assertEqual(config["seed"], 7)
        for length in (1, 33):
            vanilla = describe_config("llada", gen_length=length)
            self.assertEqual((vanilla["steps"], vanilla["block_length"]), (length, length))
            for mode in ("instant", "fastdllm"):
                with self.subTest(mode=mode, length=length), self.assertRaises(ValueError):
                    describe_config(mode, gen_length=length)

    def test_backend_policy_wrapper_delegates_and_records_both_names(self):
        from acceleration import LLaDABackend, POLICY_TO_MODE
        backend = LLaDABackend("fixture-model-path")
        for policy, mode in POLICY_TO_MODE.items():
            with self.subTest(policy=policy), mock.patch.object(backend, "generate", return_value={"mode": mode, "response": "fixture"}) as generate:
                result = backend.generate_for_policy(" question ", policy=policy, gen_length=64, seed=7, system_prompt="Be brief.")
                generate.assert_called_once_with(" question ", mode=mode, gen_length=64, seed=7, system_prompt="Be brief.")
                self.assertEqual((result["policy"], result["mode"]), (policy, mode))
                self.assertNotIn("safety_score", result)

    def test_backend_policy_bad_arguments_fail_before_loading(self):
        from acceleration import LLaDABackend
        backend = LLaDABackend("fixture-model-path")
        cases = [
            {"prompt": "question", "policy": "UNKNOWN"},
            {"prompt": "question", "policy": []},
            {"prompt": "", "policy": "SAFE"},
            {"prompt": "question", "policy": "SAFE", "gen_length": 0},
            {"prompt": "question", "policy": "SAFE", "gen_length": 1.5},
            {"prompt": "question", "policy": "SAFE", "seed": -1},
            {"prompt": "question", "policy": "SAFE", "system_prompt": 1},
        ]
        with mock.patch.object(backend, "load", side_effect=AssertionError("must not load")) as load:
            for arguments in cases:
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    backend.generate_for_policy(**arguments)
            load.assert_not_called()


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

    def test_cli_default_mode_and_mutually_exclusive_policy(self):
        self.assertEqual(build_parser().parse_args([]).mode, "instant")
        for arguments in (["--mode", "llada", "--policy", "SAFE"], ["--policy", "safe"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args(arguments)

    def test_compare_all_includes_three_modes_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            args = build_parser().parse_args(["--mode", "compare-all", "--output", str(output), "--gen-length", "64", "--seed", "7"])
            self.assertEqual(run_jobs(args, load_records(prompt=" question \n"), backend_factory=FakeBackend), 0)
            self.assertEqual(len(FakeBackend.instances), 1)
            backend = FakeBackend.instances[0]
            self.assertEqual([options["mode"] for _, options in backend.calls], ["instant", "fastdllm", "llada"])
            self.assertEqual([prompt for prompt, _ in backend.calls], [" question \n"] * 3)
            self.assertEqual([{k: v for k, v in options.items() if k != "mode"} for _, options in backend.calls],
                             [{"gen_length": 64, "seed": 7, "system_prompt": None}] * 3)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["mode"] for row in rows], ["instant", "fastdllm", "llada"])

    def test_policy_dispatch_preserves_names_and_input_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            for policy, mode in (("FAST", "instant"), ("NORMAL", "fastdllm"), ("SAFE", "llada")):
                output = Path(directory) / (policy + ".json")
                args = build_parser().parse_args(["--policy", policy, "--output", str(output)])
                records = load_records(prompt="question")
                records[0]["source_record"]["risk_tier"] = "metadata only"
                self.assertEqual(run_jobs(args, records, backend_factory=FakeBackend), 0)
                backend = FakeBackend.instances[-1]
                self.assertEqual(backend.policy_calls[0][1], policy)
                row = json.loads(output.read_text(encoding="utf-8"))[0]
                self.assertEqual((row["policy"], row["mode"]), (policy, mode))
                self.assertEqual((row["generation"]["policy"], row["generation"]["mode"]), (policy, mode))
                self.assertEqual(row["input"]["risk_tier"], "metadata only")

    def test_describe_policy_and_compare_configs_without_backend(self):
        for arguments, expected_modes in ((["--mode", "compare"], {"instant", "fastdllm"}),
                                          (["--mode", "compare-all"], {"instant", "fastdllm", "llada"})):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main([*arguments, "--describe-config"], backend_factory=FakeBackend), 0)
            self.assertEqual(set(json.loads(stdout.getvalue())), expected_modes)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--policy", "SAFE", "--describe-config"], backend_factory=FakeBackend), 0)
        described = json.loads(stdout.getvalue())
        self.assertEqual((described["policy"], described["mode"]), ("SAFE", "llada"))
        self.assertEqual(described["config"]["mode"], "llada")
        self.assertFalse(FakeBackend.instances)

    def test_invalid_policy_config_creates_no_output_or_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(["--policy", "SAFE", "--gen-length", "0", "--prompt", "question", "--output", str(output)], backend_factory=FakeBackend)
            self.assertEqual(code, 1)
            self.assertFalse(output.exists())
            self.assertFalse(FakeBackend.instances)

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
