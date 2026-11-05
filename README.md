# SafeDiff

## Training-Free Safety-Aware Acceleration for Diffusion Language Models

SafeDiff is a research prototype investigating whether the internal denoising trajectory of Diffusion Language Models (dLLMs) can provide a lightweight, training-free signal for safety detection.

### Acceleration backends

The standalone generation API provides two fixed LLaDA-8B-Instruct modes:
`instant` (contiguous ALL8) and `fastdllm` (the pinned Fast-dLLM v1 dual-cache
decoder). They share a loaded model and return responses, exact configurations,
observed model-call counts, timing, and source provenance. These modes do not
define a safety policy.

See [the acceleration guide](docs/acceleration.md) for the Python API, paired
input runner, dependency setup, and measurement definitions. Inspect the presets
without loading a model:

```bash
python -m analysis.run_acceleration --mode compare --describe-config
```

### Research Question

Can safety-relevant information in the diffusion/denoising process be used during inference while preserving the efficiency benefits of fast dLLM inference?

### Current Approach

The prototype currently includes:

- LLaDA-8B diffusion language model inference
- Adversarial prompt refinement
- Hidden-state comparison between refined and control inputs
- Hidden-state measurements over different tail lengths
- Exploratory safety classification
- Initial investigation of Fast-dLLM cache-based acceleration

### Preliminary Experiment

A 20-example WildJailbreak subset was evaluated:

- 10 adversarial harmful examples
- 10 adversarial benign examples

Hidden-state tail sizes of 32, 64, 128, and 256 tokens were investigated.

| Tail size | AUC |
|---:|---:|
| 32 | 0.45 |
| 64 | 0.42 |
| 128 | **0.61** |
| 256 | 0.32 |

The 128-token condition currently gives the strongest exploratory separation.

An exploratory threshold selected on the same 20 samples produced:

- Sensitivity: 0.60
- Specificity: 0.90
- Accuracy: 0.75

These values are preliminary and are **not independent validation results**, because the threshold was selected using the same samples.

### Intended Contribution

Existing work has studied diffusion-language-model safety and fast diffusion inference separately, as well as representation-based safety signals.

SafeDiff investigates their combination:

> **training-free safety detection from diffusion denoising behavior integrated with efficient dLLM inference.**

The next stage is to determine whether safety detection can be retained while reducing inference cost through fast diffusion mechanisms such as caching and selective computation.

### Repository Structure

```text
models/          LLaDA/JailbreakBench model integration
refine_prompt/   Adversarial prompt refinement
analysis/        Experimental and analysis code
results/         Selected experimental results
datasets/        Experimental input data
