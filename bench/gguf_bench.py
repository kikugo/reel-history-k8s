"""CPU baseline: llama-bench on the GGUF Q4_K_M file, 8 cores.

uv run modal run -m bench.gguf_bench --smoke
uv run modal run -m bench.gguf_bench
"""

import json
import subprocess
import time
from datetime import UTC, datetime

from bench.common import VOLUMES, app, write_result
from quant import config
from quant.gguf import LLAMA_CPP_DIR, llama_cpp_image

CPU = 8
MEM_GIB = 16
PROMPT_TOKENS = 512
GEN_TOKENS = 256


def bench_cmd(model: str, smoke: bool) -> list[str]:
    return [
        f"{LLAMA_CPP_DIR}/build/bin/llama-bench", "-m", model,
        "-p", "64" if smoke else str(PROMPT_TOKENS),
        "-n", "16" if smoke else str(GEN_TOKENS),
        "-t", str(CPU), "-r", "1" if smoke else "3", "-o", "json",
    ]


def summarize(rows: list[dict]) -> dict:
    """llama-bench JSON rows: n_prompt>0 is prompt processing, n_gen>0 is generation."""
    out = {}
    for row in rows:
        kind = "prompt" if row.get("n_prompt", 0) > 0 and row.get("n_gen", 0) == 0 else "generation"
        out[kind] = {"tokens": row["n_prompt"] if kind == "prompt" else row["n_gen"],
                     "avg_tokens_per_s": row["avg_ts"], "stddev_tokens_per_s": row["stddev_ts"]}
    return out


@app.function(image=llama_cpp_image, cpu=CPU, memory=MEM_GIB * 1024, volumes=VOLUMES, timeout=30 * 60)
def run_llama_bench(smoke: bool) -> dict:
    model = f"{config.MODELS_DIR}/{config.OUTPUT_NAMES['gguf']}/{config.MODEL_NAME}-{config.GGUF_QUANT}.gguf"
    t0 = time.perf_counter()
    proc = subprocess.run(bench_cmd(model, smoke), capture_output=True, text=True, check=True)
    seconds = time.perf_counter() - t0
    rows = json.loads(proc.stdout)
    return {
        "variant": "gguf", "quant_type": config.GGUF_QUANT, "llama_cpp_tag": config.LLAMA_CPP_TAG,
        "cpu_cores": CPU, "cpu_info": rows[0].get("cpu_info") if rows else None,
        **summarize(rows),
        "wall_clock_s": round(seconds, 1), "cost_usd": config.cost_usd(seconds, None, CPU, MEM_GIB),
    }


@app.local_entrypoint()
def run_gguf_bench(smoke: bool = False):
    record = {"smoke": smoke, "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
              **run_llama_bench.remote(smoke)}
    write_result("gguf_cpu", record, smoke)
