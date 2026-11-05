The Fast-dLLM decoding source and LLaDA model/configuration source derive from
NVlabs/Fast-dLLM (Apache License 2.0); original copyright headers and LICENSE
are retained. The pinned upstream commit is
`a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b`.

`fastdllm.py` is upstream `v1/llada/generate.py`, with only the model import
changed to a relative package import. Decoder function bodies are unchanged.

`instant.py` is Han's frozen contiguous K-block generator from the
2026-10-28 threshold-parallel-sweep60 experiment. The instant preset uses
K=8, threshold=0.9, BERM mu=16, and at most 64 forwards per group including
warmup. It is not a 16-step decoder, an upstream Fast-dLLM implementation,
or the historical scatter/bidirectional ALL8 implementation.

Both modes share the model implementation from that frozen experiment. Its
changes relative to the pinned upstream model thread optional `position_ids`
and `q_position_ids` through attention/RoPE for explicit cached positions.
The ordinary branch used by Fast-dLLM does not pass these arguments. This is
an upstream decoder on a shared extended model, not a claim that the entire
model/decoder package is byte-identical to upstream. Numerical equivalence
must be checked with the same checkpoint on a GPU before making that claim.

Original and packaged hashes are recorded in provenance.json. Historical
experiment locations identify provenance only; they are not runtime
dependencies. No checkpoints or historical benchmark answers are bundled.
