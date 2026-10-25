# reel-history-k8s

A personal project in two parts. Part 1 quantizes Qwen3-8B with llm-compressor (AWQ and GPTQ, W4A16) and llama.cpp (GGUF Q4_K_M), then serves and benchmarks the variants on vLLM. Part 2 runs the Reel History recommender and a "why this film" explainer on Kubernetes.

Work in progress: Phase 1 (quantization) is under way.

## Results
<!-- RESULTS:START -->
Generated from `results/*.json` once the benchmarks exist.
<!-- RESULTS:END -->

## Run it

Needs [uv](https://docs.astral.sh/uv/) and a logged-in Modal CLI (`modal token new`).

```bash
uv sync
uv run pytest

# Phase 1: quantize on Modal (weights land in the Modal Volume `c2-models`)
uv run modal run -m quant.awq --smoke     # quick end-to-end check
uv run modal run -m quant.awq
uv run modal run -m quant.gptq
uv run modal run -m quant.gguf
uv run modal run -m quant.verify          # every variant loads and generates
```

Each run writes its settings, wall-clock time and compute cost to `results/quant/`.
