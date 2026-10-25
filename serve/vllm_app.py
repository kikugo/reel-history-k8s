"""vLLM's OpenAI-compatible server on Modal, one app per variant.

Follows Modal's vLLM example (app.server + vllm serve in a subprocess). Modal Servers take
no parameters, so the variant comes from the VARIANT environment variable:

VARIANT=awq uv run modal run -m serve.smoke
"""

import os
import subprocess

import modal

from quant import config
from serve.variants import PORT, VARIANTS, app_name, build_vllm_cmd

VARIANT = os.environ.get("VARIANT", "awq")
if VARIANT not in VARIANTS:
    raise ValueError(f"VARIANT must be one of {VARIANTS}, got {VARIANT!r}")

SERVE_GPU = "L4"
SERVE_CPU = 4
SERVE_MEM_GIB = 32
MINUTES = 60

app = modal.App(app_name(VARIANT))
models = modal.Volume.from_name("c2-models")
vllm_cache = modal.Volume.from_name("c2-vllm-cache", create_if_missing=True)
api_key = modal.Secret.from_name("vllm-api-key")

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.32.0", "transformers==5.17.0", "huggingface_hub[hf_xet]")
    # FlashInfer's sampler JIT-compiles with nvcc, which debian_slim lacks; use vLLM's own sampler.
    .env({"HF_HOME": config.HF_HOME, "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("quant", "serve")
)


@app.server(
    image=vllm_image,
    gpu=SERVE_GPU,
    cpu=SERVE_CPU,
    memory=SERVE_MEM_GIB * 1024,
    port=PORT,
    env={"VARIANT": VARIANT},
    secrets=[api_key],
    volumes={config.VOL_MOUNT: models, "/root/.cache/vllm": vllm_cache},
    scaledown_window=5 * MINUTES,  # scale to zero after 5 idle minutes
    startup_timeout=10 * MINUTES,
    target_concurrency=16,
    unauthenticated=True,  # Modal's proxy is open; vLLM checks the API key itself
)
class Server:
    @modal.enter()
    def start(self):
        cmd = build_vllm_cmd(VARIANT, os.environ["VLLM_API_KEY"])
        print("starting:", " ".join(cmd[:-1] + ["<api-key>"]))
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()
