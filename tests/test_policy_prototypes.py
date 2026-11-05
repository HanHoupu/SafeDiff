"""CPU-only integration checks for oracle and interactive policy callers."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from acceleration import LLaDABackend, POLICIES, POLICY_TO_MODE, mode_for_policy
from analysis import adaptive_acceleration_prototype as oracle
from analysis import demo_controller as demo
from analysis import prototype_controller as score_demo


class FakeBackend:
    instances = []
    # Exercise the real public policy dispatcher; only numerical generation is fake.
    generate_for_policy = LLaDABackend.generate_for_policy

    def __init__(self, model_path="fixture", **kwargs):
        self.model_path = model_path
        self.options = kwargs
        self.calls = []
        self.load_count = 0
        self.closed = False
        self.__class__.instances.append(self)

    def generate(self, prompt, **kwargs):
        if not self.load_count:
            self.load_count += 1
        self.calls.append((prompt, kwargs))
        return {"response": "fixture response", "mode": kwargs["mode"], "nfe": 3}

    def close(self):
        self.closed = True


class FailingBackend(FakeBackend):
    def generate(self, prompt, **kwargs):
        if prompt == "fail":
            raise RuntimeError("fixture generation failure")
        return super().generate(prompt, **kwargs)


class PrototypeTests(unittest.TestCase):
    def setUp(self):
        FakeBackend.instances.clear()

    def test_import_has_no_model_or_interactive_side_effects(self):
        script = (
            "import sys; import analysis.adaptive_acceleration_prototype; "
            "import analysis.demo_controller; "
            "import analysis.prototype_controller; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-B", "-c", script],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_oracle_labels_and_guard_heuristic_use_public_mapping(self):
        for level, policy in enumerate(POLICIES):
            self.assertEqual(oracle.route({"risk_level": level}),
                             (policy, mode_for_policy(policy)))
        for raw, risk, policy in (("Unsafe", "HIGH", "SAFE"),
                                  ("Controversial", "MEDIUM", "NORMAL"),
                                  ("Safe", "LOW", "FAST"),
                                  ("unparsed", "UNKNOWN", "NORMAL")):
            self.assertEqual(demo.get_policy(raw), (risk, policy, mode_for_policy(policy)))
        with mock.patch.dict(POLICY_TO_MODE, {"FAST": "llada"}):
            self.assertEqual(oracle.route({"label": 0}), ("FAST", "llada"))
            self.assertEqual(demo.get_policy("Safe"), ("LOW", "FAST", "llada"))

    def test_oracle_reuses_backend_and_preserves_inputs(self):
        records = [{"id": n, "risk_level": n, "prompt": f" prompt {n} "}
                   for n in (2, 0, 1)]
        original = json.loads(json.dumps(records))
        backend = FakeBackend()
        rows = oracle.run_records(records, backend, gen_length=96, seed=7,
                                  system_prompt="fixture system")
        self.assertEqual(backend.load_count, 1)
        self.assertEqual([r["id"] for r in rows], [2, 0, 1])
        self.assertEqual([r["mode"] for r in rows], ["llada", "instant", "fastdllm"])
        self.assertEqual(records, original)
        for row, record, (prompt, options) in zip(rows, records, backend.calls):
            self.assertEqual(row["source_record"], record)
            self.assertEqual(row["routing_source"], "oracle_dataset_risk_label")
            self.assertEqual(row["generation"]["policy"], row["policy"])
            self.assertEqual(prompt, record["prompt"])
            self.assertEqual(options["gen_length"], 96)
            self.assertEqual(options["seed"], 7)
            self.assertEqual(options["system_prompt"], "fixture system")
            self.assertNotIn("steps", options)
            self.assertNotIn("adaptive_steps", row)

    def test_all_oracle_records_validated_before_generation(self):
        for bad in ({"prompt": "missing label"}, {"risk_level": 0, "prompt": ""},
                    {"risk_level": 0, "prompt": "one", "text": "other"}):
            backend = FakeBackend()
            with self.assertRaises(ValueError):
                oracle.run_records([{"risk_level": 1, "prompt": "valid"}, bad], backend)
            self.assertEqual(backend.calls, [])

    def test_oracle_cli_constructs_one_backend_and_closes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "oracle.json"
            path.write_text(json.dumps([{"risk_level": n, "prompt": str(n)}
                                        for n in range(3)]), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                oracle.main(["--input", str(path), "--out_dir", str(root / "out")],
                            backend_factory=FakeBackend)
            self.assertEqual(len(FakeBackend.instances), 1)
            backend = FakeBackend.instances[0]
            self.assertTrue(backend.options["local_files_only"])
            self.assertTrue(backend.closed)
            self.assertEqual(backend.load_count, 1)
            self.assertIn("oracle", output.getvalue().lower())
            self.assertNotIn("Budget reduction", output.getvalue())
            rows = json.loads((root / "out" / "adaptive_results.json").read_text())
            self.assertEqual(len(rows), 3)

    def test_demo_dry_run_never_constructs_models(self):
        classifier = mock.Mock()
        controller = demo.DemoController(backend_factory=FakeBackend, classifier=classifier)
        for policy in POLICIES:
            result = controller.run("fixture", policy=policy, dry_run=True)
            self.assertEqual(result["mode"], mode_for_policy(policy))
            self.assertNotIn("generation", result)
        self.assertEqual(controller.run("fixture", guard_output="Unsafe", dry_run=True)["mode"], "llada")
        with self.assertRaises(ValueError):
            controller.run("fixture", dry_run=True)
        self.assertEqual(FakeBackend.instances, [])
        classifier.classify.assert_not_called()

    def test_demo_real_selection_reuses_one_backend(self):
        classifier = mock.Mock()
        classifier.classify.side_effect = ["Safe", "Controversial", "Unsafe"]
        controller = demo.DemoController(backend_factory=FakeBackend, classifier=classifier)
        for policy in POLICIES:
            row = controller.run("fixture", gen_length=64, seed=123)
            self.assertEqual(row["policy"], policy)
            self.assertEqual(row["generation"]["mode"], mode_for_policy(policy))
        self.assertEqual(len(FakeBackend.instances), 1)
        self.assertEqual(controller.backend.load_count, 1)
        self.assertTrue(controller.backend.options["local_files_only"])
        controller.close()
        self.assertTrue(FakeBackend.instances[0].closed)

    def test_allow_download_requires_explicit_opt_in(self):
        controller = demo.DemoController(backend_factory=FakeBackend, allow_download=True)
        controller.run("fixture", policy="FAST")
        self.assertFalse(controller.backend.options["local_files_only"])
        self.assertTrue(demo.GuardClassifier().local_files_only)
        self.assertFalse(demo.GuardClassifier(allow_download=True).local_files_only)

    def test_cli_dry_run_works_outside_repository_without_models(self):
        path = Path(demo.__file__).resolve()
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", str(path), "--dry-run",
                                     "--prompt", "fixture", "--policy", "SAFE"],
                                    cwd=directory, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["mode"], "llada")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            demo.main(["--dry-run"], backend_factory=FakeBackend)
        self.assertEqual(json.loads(output.getvalue()), POLICY_TO_MODE)
        self.assertEqual(FakeBackend.instances, [])

    def test_backend_mapping_mismatch_is_not_silently_reported(self):
        backend = mock.Mock()
        backend.generate_for_policy.return_value = {"policy": "FAST", "mode": "llada"}
        with self.assertRaises(RuntimeError):
            oracle.run_records([{"label": 0, "prompt": "fixture"}], backend)
        controller = demo.DemoController(backend_factory=lambda *a, **k: backend)
        with self.assertRaises(RuntimeError):
            controller.run("fixture", policy="FAST")

    def test_partial_generation_and_error_survive_in_json_and_jsonl(self):
        for suffix in ("json", "jsonl"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "inputs.json"
                source.write_text(json.dumps([
                    {"risk_level": 2, "prompt": "first"},
                    {"risk_level": 0, "prompt": "fail"},
                    {"risk_level": 1, "prompt": "never run"},
                ]), encoding="utf-8")
                output = root / f"merged.{suffix}"
                args = ["--input", str(source), "--out-dir", str(root / "policies"),
                        "--output", str(output)]
                with self.assertRaisesRegex(RuntimeError, "fixture generation failure"):
                    oracle.main(args, backend_factory=FailingBackend)
                data = output.read_text(encoding="utf-8")
                rows = json.loads(data) if suffix == "json" else [json.loads(line) for line in data.splitlines()]
                self.assertEqual([row["status"] for row in rows], ["ok", "error"])
                self.assertEqual(rows[1]["policy"], "FAST")
                self.assertEqual(rows[1]["error"]["type"], "RuntimeError")
                self.assertTrue(FailingBackend.instances[-1].closed)
                before = output.read_bytes()
                FakeBackend.instances.clear()
                with self.assertRaises(FileExistsError):
                    oracle.main(args, backend_factory=FakeBackend)
                self.assertEqual(FakeBackend.instances, [])
                self.assertEqual(output.read_bytes(), before)

    def test_mixed_policy_configs_fail_before_output_or_backend_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "inputs.json"
            source.write_text(json.dumps([
                {"risk_level": 2, "prompt": "SAFE allows length 33"},
                {"risk_level": 0, "prompt": "FAST requires a multiple of 32"},
            ]), encoding="utf-8")
            with self.assertRaises(ValueError):
                oracle.main(["--input", str(source), "--out-dir", str(root / "out"),
                             "--gen-length", "33"], backend_factory=FakeBackend)
            self.assertEqual(FakeBackend.instances, [])
            self.assertFalse((root / "out").exists())
        classifier = mock.Mock()
        controller = demo.DemoController(backend_factory=FakeBackend, classifier=classifier)
        with self.assertRaises(ValueError):
            controller.run("fixture", gen_length=33)
        classifier.classify.assert_not_called()
        self.assertEqual(FakeBackend.instances, [])

    def test_demo_persists_error_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo.jsonl"
            args = ["--prompt", "fail", "--policy", "NORMAL", "--output", str(output)]
            with self.assertRaises(RuntimeError):
                demo.main(args, backend_factory=FailingBackend)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "error")
            self.assertEqual(record["policy"], "NORMAL")
            before = output.read_bytes()
            FakeBackend.instances.clear()
            with self.assertRaises(FileExistsError):
                demo.main(args, backend_factory=FakeBackend)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(FakeBackend.instances, [])

    def test_existing_score_threshold_routes_without_step_claims(self):
        self.assertEqual(score_demo.THRESHOLD, 0.045201)
        self.assertEqual(score_demo.route_score(0), ("LOW", "FAST", "instant"))
        self.assertEqual(score_demo.route_score(score_demo.THRESHOLD), ("HIGH", "SAFE", "llada"))
        with mock.patch.dict(POLICY_TO_MODE, {"SAFE": "fastdllm"}):
            self.assertEqual(score_demo.route_score(1)[2], "fastdllm")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.json"
            path.write_text(json.dumps([
                {"id": 1, "sp_hid_tail": 0.01},
                {"id": 2, "sp_hid_tail": 0.1},
                {"id": 3, "sp_hid_tail": None},
            ]), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                score_demo.main(["--scores", str(path)])
        self.assertIn("mode=instant", output.getvalue())
        self.assertIn("mode=llada", output.getvalue())
        self.assertNotIn("steps=", output.getvalue())


if __name__ == "__main__":
    unittest.main()
