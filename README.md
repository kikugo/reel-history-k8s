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

The quantized models are public on Hugging Face: [AWQ](https://huggingface.co/karthikcr/Qwen3-8B-W4A16-AWQ), [GPTQ](https://huggingface.co/karthikcr/Qwen3-8B-W4A16-GPTQ), [GGUF](https://huggingface.co/karthikcr/Qwen3-8B-GGUF).

```bash
# Phase 2: serve a variant (bf16, awq or gptq) with vLLM on an L4, scaled to zero when idle.
# Needs a Modal secret with the API key the server checks:
modal secret create vllm-api-key VLLM_API_KEY=$(openssl rand -hex 32)
VARIANT=awq uv run modal run -m serve.smoke   # one keyed chat request, one without the key
```

### Part 2: Reel History in containers (Phase 4)

Three services, each with a multi-stage Dockerfile, a non-root user, `/healthz`, `/readyz` and metrics:

- `services/recommender`: FastAPI with GRU4Rec in onnxruntime (`POST /recommend`), LLaRA neighbours from a FAISS index (`GET /similar/{movieId}`) and title search (`GET /search`). It keeps no session, so any replica can answer. `/readyz` turns green only after the model has loaded and run once.
- `services/explainer`: FastAPI that asks the Phase 2 vLLM server (AWQ by default) why a film fits a user's list. It has a per-call timeout, retries with backoff, an in-process cache, and a template answer when the LLM is down or still waking from scale-to-zero.
- `services/frontend`: a small React page, built with Vite and served by unprivileged nginx, which also proxies `/api/rec/` and `/api/explain/`.

The recommender's artifacts download from the public [Reel History Space](https://huggingface.co/spaces/karthikcr/reel-history) at a pinned commit when the stack starts (`python -m recommender.fetch`, an init container on Kubernetes).

```bash
cp .env.example .env            # optional: set VLLM_BASE_URL and VLLM_API_KEY for LLM answers
docker compose up --build -d    # then open http://localhost:8080
scripts/compose_smoke.sh        # health, non-root, recommend, similar, explain and metrics checks
```

The GRU4Rec and LLaRA models were trained in the Reel History team project at EPITA with my teammates Ganesh and Ana Maria. The service code in this repo is new.
