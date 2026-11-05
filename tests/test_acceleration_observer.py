"""Exercise real CPU nn.Module hooks and the uncached adapter, without weights."""

import importlib.util
from types import SimpleNamespace
import unittest

from acceleration.backend import _ForwardObserver, _UncachedModel


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DECODER_DEPENDENCIES = all(importlib.util.find_spec(name) is not None
                               for name in ("torch", "numpy", "transformers"))


@unittest.skipUnless(HAS_TORCH, "optional CPU torch is unavailable")
class ForwardObserverTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("anchor", torch.zeros(1, device="cpu"))
                self.config = SimpleNamespace(use_cache=True)
                self.calls = []

            @property
            def device(self):
                return self.anchor.device

            def forward(self, input_ids, *, use_cache=None, past_key_values=None):
                selected = self.config.use_cache if use_cache is None else use_cache
                self.calls.append({"use_cache": use_cache, "selected": selected,
                                   "past_key_values": past_key_values})
                batch, query = input_ids.shape
                # Dual-cache calls replace positions within the full KV cache.
                key_length = query if past_key_values is None else past_key_values[0][0].shape[-2]
                cache = None
                if selected:
                    key = torch.zeros(batch, 1, key_length, 2, device="cpu")
                    cache = ((key, key.clone()),)
                logits = torch.zeros(batch, query, 16, device="cpu")
                logits[:, :, 1] = 3.0
                return SimpleNamespace(logits=logits, past_key_values=cache)

        self.model = TinyModel()

    def test_mixed_cached_and_uncached_calls_preserve_shared_model_defaults(self):
        torch = self.torch
        observer = _ForwardObserver()
        handle = self.model.register_forward_hook(observer, with_kwargs=True)
        try:
            full = self.model(torch.ones((1, 7), dtype=torch.long), use_cache=True)
            adapter = _UncachedModel(self.model)
            self.assertEqual(adapter.device, torch.device("cpu"))
            uncached = adapter(torch.ones((1, 5), dtype=torch.long))
            self.assertIsNone(uncached.past_key_values)
            self.assertIs(self.model.config.use_cache, True)
            self.model(input_ids=torch.ones((1, 2), dtype=torch.long),
                       use_cache=True, past_key_values=full.past_key_values)
            # Verify that SAFE preserves the shared model's cache default.
            self.assertIsNotNone(self.model(torch.ones((1, 3), dtype=torch.long)).past_key_values)
        finally:
            handle.remove()
        metrics = observer.result(4)
        self.assertEqual(metrics["nfe"], 4)
        self.assertEqual(metrics["query_positions"], 17)
        self.assertEqual(metrics["full_forward_calls"], 3)
        self.assertEqual(metrics["cached_forward_calls"], 1)
        self.assertIn({"batch": 1, "query_length": 2, "key_length": 7,
                       "full_forward": False, "count": 1}, metrics["forward_shape_counts"])
        self.assertEqual([call["use_cache"] for call in self.model.calls], [True, False, True, None])
        self.assertIsNone(self.model.calls[1]["past_key_values"])

    def test_missing_cache_requires_explicit_uncached_call(self):
        observer = _ForwardObserver()
        self.model.config.use_cache = False
        handle = self.model.register_forward_hook(observer, with_kwargs=True)
        try:
            with self.assertRaisesRegex(RuntimeError, "explicitly uncached"):
                self.model(self.torch.ones((1, 4), dtype=self.torch.long))
        finally:
            handle.remove()
        self.assertEqual(observer.result(0)["actual_forward_calls"], 0)

    def test_cached_inputs_cannot_silently_use_uncached_metric_path(self):
        tokens = self.torch.ones((1, 4), dtype=self.torch.long)
        cached = self.model(tokens, use_cache=True).past_key_values
        observer = _ForwardObserver()
        handle = self.model.register_forward_hook(observer, with_kwargs=True)
        try:
            with self.assertRaisesRegex(RuntimeError, "explicitly uncached"):
                self.model(tokens, use_cache=False, past_key_values=cached)
        finally:
            handle.remove()

    def test_reported_nfe_must_match_observed_calls(self):
        observer = _ForwardObserver()
        handle = self.model.register_forward_hook(observer, with_kwargs=True)
        try:
            _UncachedModel(self.model)(self.torch.ones((1, 4), dtype=self.torch.long))
        finally:
            handle.remove()
        with self.assertRaisesRegex(RuntimeError, "NFE"):
            observer.result(2)
        self.assertEqual(observer.result(1)["query_positions"], 4)

    def test_uncached_adapter_rejects_a_model_that_ignores_cache_disable(self):
        original = self.model.forward
        self.model.forward = lambda tokens, **kwargs: original(tokens, use_cache=True)
        with self.assertRaisesRegex(RuntimeError, "unexpectedly created a KV cache"):
            _UncachedModel(self.model)(self.torch.ones((1, 4), dtype=self.torch.long))

    @unittest.skipUnless(HAS_DECODER_DEPENDENCIES, "optional decoder dependencies are unavailable")
    def test_official_decoder_through_uncached_proxy_is_observed_end_to_end(self):
        from acceleration.vendor import llada

        observer = _ForwardObserver()
        handle = self.model.register_forward_hook(observer, with_kwargs=True)
        try:
            prompt = self.torch.tensor([[3, 4, 5]], dtype=self.torch.long)
            output = llada.generate(_UncachedModel(self.model), prompt, steps=32,
                                    gen_length=32, block_length=32, temperature=0.0,
                                    cfg_scale=0.0, remasking="low_confidence", mask_id=15)
        finally:
            handle.remove()
        metrics = observer.result(32)
        self.assertEqual(metrics["nfe"], 32)
        self.assertEqual(metrics["query_positions"], 32 * 35)
        self.assertEqual(metrics["cached_forward_calls"], 0)
        self.assertEqual(metrics["full_forward_calls"], 32)
        self.assertEqual(output[0, :3].tolist(), [3, 4, 5])
        self.assertEqual(output[0, 3:].tolist(), [1] * 32)
        self.assertTrue(all(call["use_cache"] is False and call["past_key_values"] is None
                            for call in self.model.calls))
        self.assertIs(self.model.config.use_cache, True)


if __name__ == "__main__":
    unittest.main()
