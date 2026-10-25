"""Generate the 50 film explanations with one variant and time them.

GPU variants run in vLLM on an L4 (16 requests in flight); GGUF runs in llama.cpp on CPU, one
prompt at a time. Cost per 1,000 explanations = price per second x (measured seconds / n) x 1,000.

uv run modal run -m bench.explain --variant awq --smoke
uv run modal run -m bench.explain --variant gguf
"""

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from bench.common import (
    BENCH_CPU,
    BENCH_GPU,
    BENCH_MEM_GIB,
    VOLUMES,
    app,
    load_prompts,
    local_vllm,
    vllm_image,
    write_result,
)
from quant import config
from quant.common import gpu_name, package_versions
from quant.gguf import LLAMA_CPP_DIR, llama_cpp_image
from serve.variants import served_name

MAX_TOKENS = 200
CONCURRENCY = 16
SMOKE_PROMPTS = 2
GGUF_CPU = 8
GGUF_MEM_GIB = 16


def render_qwen3(prompt: str) -> str:
    """Qwen3 chat template output for one user turn with enable_thinking=False."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def per_thousand(cost_usd: float, n: int) -> float:
    return round(cost_usd / n * 1000, 4)


def _chat(base: str, key: str, model: str, prompt: str) -> str:
    import urllib.request

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": config.SEED,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"].strip()


@app.function(image=vllm_image, gpu=BENCH_GPU, cpu=BENCH_CPU, memory=BENCH_MEM_GIB * 1024, volumes=VOLUMES,
              timeout=30 * 60)
def explain_gpu(variant: str, prompts: list[dict]) -> dict:
    with local_vllm(variant) as (base, key, _):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(CONCURRENCY) as pool:
            answers = list(pool.map(lambda p: _chat(base, key, served_name(variant), p["prompt"]), prompts))
        seconds = time.perf_counter() - t0
    cost = config.cost_usd(seconds, BENCH_GPU, BENCH_CPU, BENCH_MEM_GIB)
    return {
        "engine": "vllm", "hardware": BENCH_GPU, "gpu_name": gpu_name(), "concurrency": CONCURRENCY,
        "generation_s": round(seconds, 1), "generation_cost_usd": cost,
        "cost_per_1000_usd": per_thousand(cost, len(prompts)),
        "answers": answers, "versions": package_versions("vllm"),
    }


@app.function(image=llama_cpp_image, cpu=GGUF_CPU, memory=GGUF_MEM_GIB * 1024, volumes=VOLUMES, timeout=60 * 60)
def explain_gguf(prompts: list[dict]) -> dict:
    model = f"{config.MODELS_DIR}/{config.OUTPUT_NAMES['gguf']}/{config.MODEL_NAME}-{config.GGUF_QUANT}.gguf"
    answers = []
    t0 = time.perf_counter()
    for p in prompts:
        proc = subprocess.run(
            [f"{LLAMA_CPP_DIR}/build/bin/llama-completion", "-m", model, "-no-cnv", "-p", render_qwen3(p["prompt"]),
             "-n", str(MAX_TOKENS), "--temp", "0", "--seed", str(config.SEED), "-t", str(GGUF_CPU), "-c", "2048"],
            capture_output=True, text=True, check=True,
        )
        text = proc.stdout.split("</think>\n\n", 1)[-1].replace("[end of text]", "").strip()
        answers.append(text)
    seconds = time.perf_counter() - t0
    cost = config.cost_usd(seconds, None, GGUF_CPU, GGUF_MEM_GIB)
    return {
        "engine": "llama.cpp", "hardware": f"{GGUF_CPU} CPU cores", "concurrency": 1,
        "generation_s": round(seconds, 1), "generation_cost_usd": cost,
        "cost_per_1000_usd": per_thousand(cost, len(prompts)),
        "answers": answers, "llama_cpp_tag": config.LLAMA_CPP_TAG,
    }


@app.local_entrypoint()
def run_explain(variant: str, smoke: bool = False):
    prompts = load_prompts(SMOKE_PROMPTS if smoke else None)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    out = explain_gguf.remote(prompts) if variant == "gguf" else explain_gpu.remote(variant, prompts)
    answers = out.pop("answers")
    record = {
        "variant": variant, "smoke": smoke, "started_utc": started, "n": len(prompts),
        "max_tokens": MAX_TOKENS, **out,
        "items": [{"id": p["id"], "lang": p["lang"], "answer": a} for p, a in zip(prompts, answers, strict=True)],
    }
    write_result(f"explain_{variant}", record, smoke)
