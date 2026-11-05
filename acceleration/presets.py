"""Two frozen generation presets. Importing this module never imports torch."""

MODES = ("instant", "fastdllm")


def describe_config(mode, gen_length=256, seed=42):
    """Return a new JSON-compatible configuration, validating CPU-only inputs."""
    if mode not in MODES:
        raise ValueError("mode must be one of: " + ", ".join(MODES))
    if type(gen_length) is not int or gen_length < 32 or gen_length % 32:
        raise ValueError("gen_length must be a positive multiple of 32")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    common = {
        "mode": mode,
        "model_family": "LLaDA-8B-Instruct",
        "batch_size": 1,
        "gen_length": gen_length,
        "block_length": 32,
        "temperature": 0.0,
        "remasking": "low_confidence",
        "mask_id": 126336,
        "seed": seed,
        "threshold": 0.9,
        "dual_cache": True,
        "factor": None,
        "eos_early_stop": False,
        "output_policy": "fixed_canvas; decode only the prefix before first EOS/EOT",
        "residual_mask_policy": "retain and report; no repair",
    }
    if mode == "instant":
        common.update(
            generator="contiguous_all8",
            parallel_k=8,
            berm_mu=16,
            max_rounds_per_group=64,
            steps=None,
            group_layout="contiguous nonoverlapping groups; keep shorter final group",
            refresh_interval=0,
            kv_delta_gamma=0.0,
            adaptive_alpha=None,
            stopping_rule="each group ends when unmasked or after 64 forwards including warmup",
        )
    else:
        common.update(
            generator="official_fastdllm_v1_dual_cache",
            parallel_k=1,
            berm_mu=0,
            steps=gen_length,
            max_rounds_per_group=None,
            stopping_rule="native upstream loop: at most 32 forwards per block including warmup",
        )
    return common
