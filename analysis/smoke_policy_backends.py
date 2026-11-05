"""Exercise all policy paths with a tiny random model on a local CUDA device.

This checks integration, not LLaDA-8B output quality or checkpoint equivalence.
No checkpoint is downloaded. Run separately from the CPU contract tests.
"""

import argparse
import json
import os
from pathlib import Path


def run_smoke(device="cuda:0"):
    # Use eager PyTorch only; this check needs neither TensorFlow nor Triton.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    import torch
    import transformers
    from acceleration import LLaDABackend
    from acceleration.backend import source_provenance
    from acceleration.vendor.instant import model_matmul_spec
    from acceleration.vendor.model.configuration_llada import LLaDAConfig
    from acceleration.vendor.model.modeling_llada import LLaDAModelLM

    if not torch.cuda.is_available():
        raise RuntimeError("This optional smoke check requires CUDA")
    torch.manual_seed(42)
    config = LLaDAConfig(
        d_model=32, n_heads=4, n_kv_heads=4, n_layers=2, mlp_ratio=2,
        vocab_size=126464, embedding_size=126464, max_sequence_length=128,
        block_type="llama", activation_type="silu", rope=True,
        flash_attention=False, use_cache=True, weight_tying=False,
        attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0,
    )
    model = LLaDAModelLM(config, init_params=True).to(device).eval()

    class FixtureTokenizer:
        eos_token_id = 1
        unk_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return torch.tensor([[2, 3, 4]], dtype=torch.long)

        def convert_tokens_to_ids(self, token):
            return 1 if token == "<|eot_id|>" else 0

        def decode(self, tokens, **kwargs):
            return " ".join(map(str, tokens))

    backend = LLaDABackend("synthetic-tiny-model", device=device, dtype="float32",
                          flash_attention=False)
    backend._model = model
    backend._tokenizer = FixtureTokenizer()
    backend._matmul_spec = model_matmul_spec(model)
    backend._runtime = {"model_load_seconds": 0.0, "synthetic_fixture": True}
    backend._provenance = source_provenance()
    model_identity = id(model)
    first = {}
    checks = []
    try:
        for policy in ("FAST", "NORMAL", "SAFE", "FAST", "NORMAL", "SAFE"):
            result = backend.generate_for_policy("synthetic fixture", policy=policy,
                                                 gen_length=32, seed=42)
            assert id(backend._model) == model_identity
            assert model.config.use_cache is True, "SAFE mutated the shared model defaults"
            assert result["policy"] == policy
            assert len(result["generated_token_ids"]) == 32
            assert result["residual_masks"] == 0
            if policy in first:
                assert result["generated_token_ids"] == first[policy]["generated_token_ids"]
                assert result["nfe"] == first[policy]["nfe"]
            else:
                first[policy] = result
            if policy == "SAFE":
                assert result["nfe"] == result["full_forward_calls"] == 32
                assert result["cached_forward_calls"] == 0
                assert result["query_positions"] == 32 * 35
            else:
                assert result["cached_forward_calls"] > 0
            checks.append({"policy": policy, "mode": result["mode"],
                           "nfe": result["nfe"],
                           "full_forward_calls": result["full_forward_calls"],
                           "cached_forward_calls": result["cached_forward_calls"],
                           "residual_masks": result["residual_masks"], "passed": True})
        short = backend.generate_for_policy("synthetic fixture", policy="SAFE",
                                            gen_length=1, seed=42)
        assert short["nfe"] == 1 and short["query_positions"] == 4
        assert short["cached_forward_calls"] == short["residual_masks"] == 0
        checks.append({"policy": "SAFE", "gen_length": 1, "passed": True})
        assert not model._forward_hooks, "Forward observer was not removed"
    finally:
        backend.close()
    return {
        "status": "PASSED", "scope": "Synthetic integration only; no pretrained checkpoint",
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "transformers": transformers.__version__, "dtype": "float32", "attention": "SDPA",
        "torch_compile": False,
        "model": {"d_model": 32, "n_layers": 2, "random_weights": True},
        "shared_model_and_mode_switch_repeatability": True, "checks": checks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", help="Optional new JSON receipt; existing files are rejected")
    args = parser.parse_args(argv)
    if args.output and Path(args.output).exists():
        raise FileExistsError(args.output)
    result = run_smoke(args.device)
    rendered = json.dumps(result, indent=2)
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
