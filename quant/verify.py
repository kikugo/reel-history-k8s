"""Phase 1 "Done when" check: every variant loads and generates.

AWQ, GPTQ and the BF16 base load in vLLM on an L4; GGUF Q4_K_M loads in llama.cpp on CPU.
Each variant gets the same prompt, greedy, 32 new tokens.

uv run modal run -m quant.verify --smoke   # checks the smoke AWQ/GPTQ outputs only
uv run modal run -m quant.verify           # all variants, writes results/quant/load_check.json
"""

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

from quant import config
from quant.common import VOLUMES, app, base_snapshot_path, gpu_name, package_versions, write_result
from quant.gguf import LLAMA_CPP_DIR, llama_cpp_image

VERIFY_GPU = "L4"
VERIFY_CPU = 4
VERIFY_MEM_GIB = 32
MAX_NEW_TOKENS = 32

PROMPT = "In one sentence, why might someone who loved Amélie enjoy Cinema Paradiso?"
# Qwen3 chat template output for PROMPT with enable_thinking=False. The vLLM check
# re-renders it with the real tokenizer and records whether it still matches.
PROMPT_RENDERED = (
    f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.32.0", "transformers==5.17.0", "huggingface_hub[hf_xet]")
    # FlashInfer's sampler JIT-compiles with nvcc, which debian_slim lacks; use vLLM's own sampler.
    .env({"HF_HOME": config.HF_HOME, "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("quant")
)


def variant_paths(smoke: bool) -> dict:
    root = config.SMOKE_DIR if smoke else config.MODELS_DIR
    paths = {m: f"{root}/{config.OUTPUT_NAMES[m]}" for m in ("awq", "gptq")}
    if not smoke:
        paths["bf16"] = None  # resolved to the HF cache snapshot inside the container
    return paths


@app.function(image=vllm_image, gpu=VERIFY_GPU, cpu=VERIFY_CPU, memory=VERIFY_MEM_GIB * 1024,
              volumes=VOLUMES, timeout=20 * 60)
def check_vllm(name: str, path: str | None) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    path = path or base_snapshot_path()
    record = {"variant": name, "path": path, "engine": "vllm", "loaded": False}
    t = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        llm = LLM(model=path, max_model_len=4096, gpu_memory_utilization=0.85, seed=config.SEED)
        out = llm.generate([rendered], SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS))
        qcfg = json.loads(Path(path, "config.json").read_text()).get("quantization_config", {})
        record.update(
            loaded=True,
            output=out[0].outputs[0].text,
            prompt_matches_expected=rendered == PROMPT_RENDERED,
            quant_method=qcfg.get("quant_method"),
            quant_format=qcfg.get("format"),
        )
    except Exception as exc:  # noqa: BLE001 - report the failure instead of stopping other variants
        record["error"] = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - t
    record.update(
        seconds=round(seconds, 1),
        gpu_name=gpu_name(),
        cost_usd=config.cost_usd(seconds, VERIFY_GPU, VERIFY_CPU, VERIFY_MEM_GIB),
        versions=package_versions("vllm", "torch", "transformers"),
    )
    return record


@app.function(image=llama_cpp_image, cpu=VERIFY_CPU, memory=VERIFY_MEM_GIB * 1024, volumes=VOLUMES,
              timeout=20 * 60)
def check_gguf() -> dict:
    out_dir = f"{config.MODELS_DIR}/{config.OUTPUT_NAMES['gguf']}"
    path = f"{out_dir}/{config.MODEL_NAME}-{config.GGUF_QUANT}.gguf"
    record = {"variant": "gguf", "path": path, "engine": "llama.cpp", "loaded": False}
    t = time.perf_counter()
    proc = subprocess.run(
        [f"{LLAMA_CPP_DIR}/build/bin/llama-completion", "-m", path, "-no-cnv", "-p", PROMPT_RENDERED,
         "-n", str(MAX_NEW_TOKENS), "--temp", "0", "--seed", str(config.SEED), "-t", str(VERIFY_CPU)],
        capture_output=True, text=True, check=False,
    )
    seconds = time.perf_counter() - t
    text = proc.stdout.split("</think>\n\n", 1)[-1].strip()
    if proc.returncode == 0 and text:
        record.update(loaded=True, output=text)
    else:
        record["error"] = proc.stderr[-2000:]
    record.update(
        seconds=round(seconds, 1),
        cost_usd=config.cost_usd(seconds, None, VERIFY_CPU, VERIFY_MEM_GIB),
        llama_cpp_tag=config.LLAMA_CPP_TAG,
    )
    return record


@app.local_entrypoint()
def run_verify(smoke: bool = False):
    started = datetime.now(UTC).isoformat(timespec="seconds")
    names, paths = zip(*variant_paths(smoke).items())
    checks = list(check_vllm.map(names, paths))
    if not smoke:
        checks.append(check_gguf.remote())
    record = {
        "smoke": smoke,
        "started_utc": started,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "all_loaded": all(c["loaded"] and c.get("output") for c in checks),
        "checks": checks,
    }
    write_result("load_check", record, smoke)
    for c in checks:
        print(f"{c['variant']:>5}: loaded={c['loaded']} {c.get('output', c.get('error', ''))[:120]!r}")
