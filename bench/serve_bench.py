"""Throughput and latency of one variant with `vllm bench serve`, on an L4.

vLLM runs inside the same container and the client hits localhost, so the numbers measure
the engine, not Modal's proxy or the internet.

uv run modal run -m bench.serve_bench --variant awq --smoke
uv run modal run -m bench.serve_bench --variant awq
"""

import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from bench.common import (
    BENCH_CPU,
    BENCH_GPU,
    BENCH_MEM_GIB,
    VOLUMES,
    app,
    local_vllm,
    parse_vllm_log,
    vllm_image,
    write_result,
)
from quant import config
from quant.common import gpu_name, package_versions
from serve.variants import model_path, served_name

INPUT_LEN = 512
OUTPUT_LEN = 256
# (concurrency, number of prompts): more prompts at higher concurrency so each level runs a similar time
LEVELS = [(1, 16), (4, 32), (16, 128)]
SMOKE = {"input_len": 128, "output_len": 32, "levels": [(1, 4)]}

KEEP = (
    "completed", "failed", "duration", "request_throughput", "output_throughput", "total_token_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p50_ttft_ms", "p95_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p50_tpot_ms", "p95_tpot_ms",
    "mean_itl_ms", "p50_itl_ms", "p95_itl_ms", "mean_e2el_ms", "p50_e2el_ms", "p95_e2el_ms",
)


def bench_cmd(base_url: str, variant: str, concurrency: int, num_prompts: int, input_len: int,
              output_len: int, out_dir: str) -> list[str]:
    return [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--base-url", base_url,
        "--endpoint", "/v1/completions",
        "--model", served_name(variant),
        "--tokenizer", model_path(variant),
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--ignore-eos",
        "--temperature", "0",
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--seed", str(config.SEED),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95",
        "--save-result",
        "--result-dir", out_dir,
        "--result-filename", f"c{concurrency}.json",
    ]


def summarize(raw: dict) -> dict:
    return {k: raw[k] for k in KEEP if k in raw}


@app.function(image=vllm_image, gpu=BENCH_GPU, cpu=BENCH_CPU, memory=BENCH_MEM_GIB * 1024,
              volumes=VOLUMES, timeout=40 * 60)
def run_bench(variant: str, smoke: bool = False) -> dict:
    input_len = SMOKE["input_len"] if smoke else INPUT_LEN
    output_len = SMOKE["output_len"] if smoke else OUTPUT_LEN
    levels = SMOKE["levels"] if smoke else LEVELS
    out_dir = "/tmp/bench"
    started = datetime.now(UTC).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    results = []
    with local_vllm(variant) as (base, key, log_path):
        startup_s = time.perf_counter() - t0
        for concurrency, n in levels:
            subprocess.run(
                bench_cmd(base, variant, concurrency, n, input_len, output_len, out_dir),
                check=True, env={**os.environ, "OPENAI_API_KEY": key},
            )
            raw = json.loads(Path(out_dir, f"c{concurrency}.json").read_text())
            results.append({"concurrency": concurrency, "num_prompts": n, **summarize(raw)})
        memory = parse_vllm_log(Path(log_path).read_text())
    seconds = time.perf_counter() - t0
    return {
        "variant": variant,
        "smoke": smoke,
        "started_utc": started,
        "input_len": input_len,
        "output_len": output_len,
        "startup_s": round(startup_s, 1),
        "levels": results,
        "memory": memory,
        "gpu": BENCH_GPU,
        "gpu_name": gpu_name(),
        "wall_clock_s": round(seconds, 1),
        "cost_usd": config.cost_usd(seconds, BENCH_GPU, BENCH_CPU, BENCH_MEM_GIB),
        "versions": package_versions("vllm", "torch", "transformers"),
    }


@app.local_entrypoint()
def run_serve_bench(variant: str, smoke: bool = False):
    write_result(f"serve_{variant}", run_bench.remote(variant, smoke), smoke)
