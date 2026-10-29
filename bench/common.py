"""Shared pieces for the Phase 3 benchmarks: Modal app, images, a local vLLM server, log parsing."""

import json
import re
import time
from contextlib import contextmanager
from pathlib import Path

import modal

from quant import config
from quant.gguf import llama_cpp_image

app = modal.App("c2-bench")
models = modal.Volume.from_name("c2-models")
vllm_cache = modal.Volume.from_name("c2-vllm-cache", create_if_missing=True)
VOLUMES = {config.VOL_MOUNT: models, "/root/.cache/vllm": vllm_cache}

BENCH_GPU = "L4"
BENCH_CPU = 4
BENCH_MEM_GIB = 32

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "bench"
PROMPTS_PATH = Path(__file__).resolve().parent / "data" / "explain_prompts.jsonl"

# Same vLLM build as serve/vllm_app.py, plus the benchmark clients.
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.32.0", "transformers==5.17.0", "huggingface_hub[hf_xet]", "lm-eval==0.4.13")
    # FlashInfer's sampler JIT-compiles with nvcc, which debian_slim lacks; use vLLM's own sampler.
    .env({"HF_HOME": config.HF_HOME, "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("quant", "serve", "bench")
)

# Phase 1 llama.cpp build, plus this repo's packages that the GGUF steps import.
gguf_image = llama_cpp_image.add_local_python_source("serve", "bench")

_WEIGHTS = re.compile(r"Model loading took ([\d.]+) GiB")
_KV_MEM = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
_KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_MAX_CONC = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")


def parse_vllm_log(text: str) -> dict:
    """Weight memory and KV-cache capacity from vLLM's startup log (last value wins)."""
    def last(pattern, group=1):
        found = pattern.findall(text)
        if not found:
            return None
        value = found[-1] if isinstance(found[-1], str) else found[-1][group - 1]
        return value.replace(",", "")

    conc_len = last(_MAX_CONC, 1)
    return {
        "weight_gib": float(v) if (v := last(_WEIGHTS)) else None,
        "kv_cache_gib": float(v) if (v := last(_KV_MEM)) else None,
        "kv_cache_tokens": int(v) if (v := last(_KV_TOKENS)) else None,
        "max_concurrency": float(v) if (v := last(_MAX_CONC, 2)) else None,
        "max_concurrency_at_tokens": int(conc_len) if conc_len else None,
    }


def _get(url: str, timeout: float = 5) -> int:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0


@contextmanager
def local_vllm(variant: str, log_path: str = "/tmp/vllm.log", timeout_s: int = 900):
    """Start `vllm serve` for a variant inside this container; yield (base_url, api_key, log_path)."""
    import secrets
    import subprocess

    from serve.variants import PORT, build_vllm_cmd

    key = secrets.token_hex(16)  # lives only inside this container
    base = f"http://127.0.0.1:{PORT}"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(build_vllm_cmd(variant, key), stdout=log, stderr=subprocess.STDOUT)
        try:
            t0 = time.perf_counter()
            while _get(f"{base}/health") != 200:
                if proc.poll() is not None:
                    raise RuntimeError(f"vllm exited with {proc.returncode}:\n{Path(log_path).read_text()[-3000:]}")
                if time.perf_counter() - t0 > timeout_s:
                    raise TimeoutError("vllm did not become healthy")
                time.sleep(2)
            yield base, key, log_path
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()


def load_prompts(limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines() if line]
    return rows[:limit] if limit else rows


def write_result(name: str, record: dict, smoke: bool) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / (f"{name}_smoke.json" if smoke else f"{name}.json")
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path}")
    return path
