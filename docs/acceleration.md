# LLaDA acceleration backends

This entry point runs **LLaDA-8B-Instruct** with either of two fixed generation
configurations. It accepts a raw user prompt, applies the checkpoint's chat
template, and returns generated text plus measurements and source provenance.
It does not estimate risk, assign FAST/NORMAL/SAFE, or judge response safety.

| Mode | Decoder | Fixed settings |
|---|---|---|
| `instant` | Existing contiguous ALL8 decoder | Up to 8 blocks per group, block length 32, confidence threshold 0.9, BERM mu 16, cap 64 forward rounds per group including warmup, dual cache |
| `fastdllm` | Pinned official Fast-dLLM v1 dual-cache decoder | Block length 32, confidence threshold 0.9, sequential blocks, native upstream stopping loop, no BERM |

Both modes use batch size 1, temperature 0, the same model instance and tokenizer,
and a fixed output canvas. Default generation length is 256 tokens. Length must
be a positive multiple of 32. At length 256 the `instant` decoder has one full
eight-block group; a shorter generation length uses a shorter group. Neither mode is
a claim of equivalent quality or safety. The output reports remaining mask
tokens instead of repairing them silently.

## Inspect without a GPU

Run from the repository root:

```bash
python -m analysis.run_acceleration --mode compare --describe-config
python -m analysis.run_acceleration --help
python -m unittest discover -s tests -p 'test_acceleration_contract.py' -v
```

Importing `acceleration` and describing configurations do not import torch or
load a model. Actual generation requires a CUDA GPU and compatible PyTorch,
Transformers and NumPy packages. Use a dedicated environment for this entry
point; it does not require the repository's older, broad `requirements_no_av.txt`
environment. With a suitable CUDA PyTorch build already installed, install the
small decoder dependency set:

```bash
python -m pip install -r requirements_acceleration.txt
```

Transformers is pinned to 4.49.0 to match the frozen implementation's tested
compatibility target. Attention implementation and package versions are recorded in
every generated result. The existing GPU PyTorch installation should be kept
when it is compatible with the machine's CUDA setup.

## Generate and compare

Use a local **LLaDA-8B-Instruct** checkpoint or an already cached Hugging Face
model. Network downloads are disabled by default.

```bash
python -m analysis.run_acceleration \
  --model-path /path/to/LLaDA-8B-Instruct \
  --mode instant \
  --prompt 'What is 17 + 25? Give a brief answer.' \
  --gen-length 256 --seed 42 \
  --output results/instant_example.json

python -m analysis.run_acceleration \
  --model-path /path/to/LLaDA-8B-Instruct \
  --mode compare \
  --input examples/acceleration_prompts.jsonl \
  --gen-length 256 --seed 42 \
  --output results/paired_example.jsonl
```

Set `--mode fastdllm` to run the official baseline alone. A Hugging Face model
ID defaults to `GSAI-ML/LLaDA-8B-Instruct`; use `--allow-download` only when a
download is intended, and `--revision <commit>` to select a fixed remote model
revision. Local checkpoint paths are also supported. Runtime metadata records
model revision when available and checkpoint metadata hashes; it does not hash
every model-weight file.

`compare` processes each prompt in the order `instant`, then `fastdllm`, with
the same model/tokenizer instance, prompt formatting, system prompt, generation
length and seed. It resets the seed for each call. Shared seed and prompt do
not imply identical answers across different decoders. This is a paired output
runner, not a statistically controlled speed benchmark: order is fixed, and
the first request includes lazy model loading. For a warm-call study, use the
API below, load once in advance, and define warmups/repetitions/order explicitly.

Optional settings include `--device cuda:0`, `--dtype bfloat16`,
`--system-prompt '...'`, and `--no-flash-attention`. The default dtype is
bfloat16. If FlashAttention is absent, the model uses PyTorch SDPA; use
`--no-flash-attention` to select SDPA explicitly, including when an installed
FlashAttention binary is incompatible with the current PyTorch build. Raw text
should not already contain a rendered chat template.

## Input and saved records

Input may be one JSON object, a JSON array of objects, or a `.jsonl` file with
one object per line. Recognized prompt fields are `text`, `prompt`, and
`vanilla prompt`. If more than one is present, their original strings must
match exactly, including whitespace. Prompt text is passed through unchanged;
whitespace is stripped only when checking whether a prompt is empty.
Empty/non-string prompts and conflicting fields are errors. All input records
are validated before model loading.
Other fields, including the source ID and existing labels, are retained as
metadata; they do not control backend selection.

The example file contains two benign smoke prompts. It is not a quality or
safety benchmark.

Each saved object has this structure:

```text
schema: safediff.acceleration.run.v1
run_id, input_index, input_id
input: original input object
prompt, mode, seed, gen_length, system_prompt
status: ok | incomplete | error
generation: backend result (on complete or incomplete generation)
error: exception type and message (on failure)
```

For `compare`, each input yields two records linked by `run_id` and
`input_index`. Each completed backend result is saved before starting the next
call. A `.json` file is an array, rewritten through an atomic temporary-file
replacement after each record. A `.jsonl` file is flushed and synchronized
after every line. Existing output files are rejected; select a new name for
another run. If generation fails, completed records and an error record are
retained and the command exits with code 1. A result with remaining masks is
saved as `status: incomplete`, including its original text, token IDs and mask
counts. Both comparison modes and all remaining inputs still run. A run with
any incomplete result exits with code 2; a fully complete run exits with code
0. A later exception takes precedence and exits with code 1. No incomplete
samples are dropped or silently repaired.

The nested `generation` object includes:

- `response`, raw decoded text, generated/response token IDs, prompt token IDs
  and their hash, first EOS index, and remaining mask counts.
- The exact preset in `config`, plus pinned generator/model source provenance
  and verified packaged-source hashes. Sources are described in
  `acceleration/vendor/provenance.json`.
- Actual observed forward calls (`nfe`), full versus cached calls, query
  positions, and forward-shape counts. These are **not FLOPs**. Any analytic
  main-matmul estimate in instant diagnostics is separately named and is not
  a measured hardware FLOP count.
- CUDA peak allocated memory and the timing definitions below.
- Checkpoint, tokenizer/template identity, package versions, GPU, dtype,
  attention implementation, and model-load duration in `runtime`.

`generation_seconds` measures CUDA-synchronized decoder wall time, including
observers and the decoder's own diagnostics. It excludes model loading,
tokenization, text decoding, evaluation, and caller file I/O.
`request_seconds` also includes lazy loading when needed, tokenization, seed
setup, and decoding. `model_was_loaded_before_request` and
`load_seconds_this_request` distinguish cold and warm requests. Instant retains
its existing diagnostics, so their cost remains included. Safety judges and
controller overhead are absent from these two backend-only timings.

The decoder completes its fixed canvas and the displayed response uses only
the token prefix before the first EOS/EOT. The full canvas and masks remain in
the result for inspection. This entry point does not create unsafe-response
rates, false-refusal rates, quality scores, or assertions that `instant` is
safe for a particular request.

## Call from a controller

The API accepts the backend mode explicitly:

```python
from acceleration import LLaDABackend

backend = LLaDABackend("/path/to/LLaDA-8B-Instruct")
try:
    backend.load()  # Optional: load once before measuring warm requests.
    result = backend.generate(
        "Explain why the sky looks blue.",
        mode="instant",
        gen_length=256,
        seed=42,
    )
    print(result["response"])
finally:
    backend.close()
```

A caller that already has a FAST/NORMAL/SAFE controller must supply its own
explicit mapping. There is no built-in mapping or default fallback:

```python
from acceleration import MODES

def generate_for_policy(backend, prompt, policy, policy_to_mode, *, seed=42):
    if policy not in policy_to_mode:
        raise ValueError(f"No backend mapping configured for policy {policy!r}")
    mode = policy_to_mode[policy]
    if mode not in MODES:
        raise ValueError(f"Unknown backend mode: {mode!r}")
    return backend.generate(prompt, mode=mode, gen_length=256, seed=seed)
```

Evaluate any mapping against paired response-quality and safety measurements
before treating it as a validated policy. Neither the names of the controller
policies nor a larger nominal computation budget supply that validation.

## Verification scope

The standard-library contract tests cover input compatibility and rejection,
one-instance paired invocation, identical shared generation arguments,
incremental persistence, preservation of failures and incomplete outputs,
continuation after remaining masks, exact prompt preservation, and refusal to
overwrite an output. They do not execute or numerically validate either decoder. GPU
smoke outputs and parity checks are reported separately with their actual
checkpoint, environment and source revisions.

The [2026-11-05 validation receipt](validation/acceleration-2026-11-05.json)
records a successful run on a dedicated NVIDIA A40 with PyTorch 2.4.0,
CUDA 12.4, Transformers 4.49.0, bfloat16 and SDPA. The checkpoint revision was
`08b83a6feb34df1a6011b80c3c00c7563e963b07`.

- All 9 contract tests passed locally and on the GPU host.
- For each mode, lengths 64, 256 and 288 exercised a short group, one full
  eight-block group, and a shorter last group. All generated token IDs and NFE
  matched the pre-integration references in all 6 comparisons. The Fast-dLLM
  reference used the original upstream model, while the API used the shared
  extended model.
- Repeated requests after mode switching preserved tokens and NFE, with the
  same loaded model object. A MASK-containing prompt was rejected.
- The command-line paired runner saved all 4 expected records. All 18 actual
  test generations completed without residual MASK tokens.

These checks validate the tested implementation paths. They do not establish
safety, benchmark accuracy, an equal-quality speedup, or equivalence for every
possible prompt, dtype, attention implementation, or generation length.
