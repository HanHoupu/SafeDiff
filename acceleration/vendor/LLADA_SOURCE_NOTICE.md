# Original LLaDA decoder

`llada.py` is an exact byte copy of `generate.py` from
[ML-GSAI/LLaDA at f51cb1731f5a40ba35c15e51b6b66b147e689f24](https://github.com/ML-GSAI/LLaDA/blob/f51cb1731f5a40ba35c15e51b6b66b147e689f24/generate.py).
No imports, functions, defaults, or demo entry point have been changed.

That pinned repository snapshot includes the MIT license, retained verbatim in
`LICENSE.LLaDA` (copyright 2025 NieShenRuc). This decoder has its own license;
the adjacent Fast-dLLM `LICENSE` is a different source's license. Source blob IDs,
SHA-256 hashes, and byte counts are recorded in `llada_provenance.json`.

The SafeDiff caller selects `steps=gen_length`, `block_length=gen_length`,
`temperature=0`, `cfg_scale=0`, `remasking="low_confidence"`, and batch size 1.
This gives one full-sequence forward and one scheduled token transfer per step.
It uses no confidence threshold for parallel transfer, BERM, skipped model
calls, cache reuse, or early EOS termination. The official function also exposes
other arguments; they are not part of this fixed SafeDiff configuration.

The original function calls `model(x)` without specifying `use_cache`. Therefore
the caller must wrap the shared model so every forward explicitly disables cache
creation and supplies no cached states. The original generator is left unchanged.

The shared model's ordinary full-sequence forward interface is compatible with
this generator. Its Fast-dLLM-derived implementation and optional position/cache
extensions still differ from the original model implementation. Synthetic CPU
tests establish decoder behavior only, not real-checkpoint or GPU numerical
equivalence. The policy name `SAFE` denotes the conservative reference choice;
this code does not classify harmful requests or establish response safety.
