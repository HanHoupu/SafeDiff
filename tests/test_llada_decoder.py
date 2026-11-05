"""Pinned-source and tiny CPU decoder checks; no checkpoint or GPU is loaded."""

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


VENDOR = Path(__file__).resolve().parents[1] / "acceleration" / "vendor"
HAS_CPU_DEPENDENCIES = all(importlib.util.find_spec(name) is not None
                           for name in ("torch", "numpy", "transformers"))


class LLaDASourceTests(unittest.TestCase):
    def test_pinned_source_and_license_are_exact_git_blobs(self):
        provenance = json.loads((VENDOR / "llada_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["commit"], "f51cb1731f5a40ba35c15e51b6b66b147e689f24")
        self.assertEqual(provenance["license"], "MIT")
        for filename, expected in provenance["files"].items():
            with self.subTest(file=filename):
                data = (VENDOR / filename).read_bytes()
                blob = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
                self.assertEqual(len(data), expected["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected["sha256"])
                self.assertEqual(hashlib.sha1(blob).hexdigest(), expected["git_blob_sha"])
                self.assertEqual(expected["changes"], [])


@unittest.skipUnless(HAS_CPU_DEPENDENCIES, "optional CPU torch/numpy/transformers are unavailable")
class LLaDADecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from acceleration.vendor import llada

        cls.torch = torch
        cls.decoder = llada

    def run_tiny_model(self, *, constant_token=None):
        torch = self.torch
        mask_id = 15
        prompt = torch.tensor([[4, 5, 6]], dtype=torch.long, device="cpu")
        original = prompt.clone()

        class TinyModel:
            device = torch.device("cpu")

            def __init__(self):
                self.inputs = []

            def __call__(self, tokens):
                # Record full-sequence inputs for the decoder schedule check.
                self.inputs.append(tokens.clone())
                logits = torch.zeros((*tokens.shape, 16), dtype=torch.float32, device="cpu")
                for position in range(tokens.shape[1]):
                    token = constant_token if constant_token is not None else 1 + position % 3
                    logits[:, position, token] = 2.0 + position / 100.0
                return type("Prediction", (), {"logits": logits})()

        model = TinyModel()
        output = self.decoder.generate(
            model, prompt, steps=32, gen_length=32, block_length=32,
            temperature=0.0, cfg_scale=0.0, remasking="low_confidence", mask_id=mask_id,
        )
        self.assertTrue(torch.equal(prompt, original))
        self.assertTrue(torch.equal(output[:, :3], original))
        self.assertEqual(tuple(output.shape), (1, 35))
        self.assertEqual(output.dtype, torch.long)
        self.assertEqual(len(model.inputs), 32)
        self.assertEqual([tuple(tokens.shape) for tokens in model.inputs], [(1, 35)] * 32)
        self.assertEqual([int((tokens == mask_id).sum()) for tokens in model.inputs], list(range(32, 0, -1)))
        trajectory = model.inputs + [output]
        self.assertTrue(all(int((after != before).sum()) == 1 for before, after in zip(trajectory, trajectory[1:])))
        self.assertFalse(bool((output == mask_id).any()))
        return output

    def test_full_budget_transfers_one_token_per_full_sequence_forward(self):
        output = self.run_tiny_model()
        self.assertEqual(output[0, 3:].tolist(), [1 + position % 3 for position in range(3, 35)])

    def test_eos_like_predictions_do_not_shorten_the_fixed_canvas_or_budget(self):
        # The caller interprets EOS after generation. The upstream decoder must
        # still fill the entire fixed canvas even if every prediction is EOS-like.
        output = self.run_tiny_model(constant_token=1)
        self.assertEqual(output[0, 3:].tolist(), [1] * 32)


if __name__ == "__main__":
    unittest.main()
