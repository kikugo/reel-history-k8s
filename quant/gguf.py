"""GGUF F16 conversion and Q4_K_M quantization of the base model, on CPU.

uv run modal run -m quant.gguf   # ~20-30 min of 8 CPU cores, ask first
"""

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

from quant import config
from quant.common import VOLUMES, app, base_snapshot_path, download_base, run_record, volume, write_result

GGUF_CPU = 8
GGUF_MEM_GIB = 32
LLAMA_CPP_DIR = "/llama.cpp"

# CPU-only llama.cpp build. GGML_NATIVE=OFF with runtime CPU variants, because the image
# builder and the container that runs it can have different CPUs.
llama_cpp_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "cmake", "build-essential")
    .run_commands(
        f"git clone --depth 1 --branch {config.LLAMA_CPP_TAG} https://github.com/ggml-org/llama.cpp {LLAMA_CPP_DIR}",
        f"cmake -S {LLAMA_CPP_DIR} -B {LLAMA_CPP_DIR}/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF "
        "-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_OPENSSL=OFF",
        f"cmake --build {LLAMA_CPP_DIR}/build -j 8 --target llama-quantize llama-completion llama-bench",
        f"pip install -r {LLAMA_CPP_DIR}/requirements/requirements-convert_hf_to_gguf.txt",
    )
    .env({"HF_HOME": config.HF_HOME, "LLAMA_CPP_DIR": LLAMA_CPP_DIR})
    .uv_pip_install("huggingface_hub")
    .add_local_file(Path(__file__).parent / "gguf.sh", "/opt/gguf.sh")
    .add_local_python_source("quant")
)


@app.function(image=llama_cpp_image, volumes=VOLUMES, cpu=GGUF_CPU, memory=GGUF_MEM_GIB * 1024, timeout=60 * 60)
def convert_gguf() -> dict:
    started = datetime.now(UTC).isoformat(timespec="seconds")
    out_dir = f"{config.MODELS_DIR}/{config.OUTPUT_NAMES['gguf']}"
    t = time.perf_counter()
    subprocess.run(
        ["bash", "/opt/gguf.sh", base_snapshot_path(), out_dir, config.GGUF_QUANT],
        check=True,
        env={"GGUF_NAME": config.MODEL_NAME, "LLAMA_CPP_DIR": LLAMA_CPP_DIR, "PATH": "/usr/local/bin:/usr/bin:/bin"},
    )
    total = time.perf_counter() - t
    volume.commit()
    steps = json.loads(Path(out_dir, "timings.json").read_text())
    timings = {"convert_s": steps["convert_s"], "quantize_s": steps["quantize_s"]}
    timings["other_s"] = max(total - sum(timings.values()), 0.0)
    extra = {
        "quant_type": config.GGUF_QUANT,
        "llama_cpp_tag": config.LLAMA_CPP_TAG,
        "imatrix": False,
        "file_bytes": {"f16": steps["f16_bytes"], "quant": steps["quant_bytes"]},
    }
    return run_record("gguf", False, timings, started, out_dir, extra, None, GGUF_CPU, GGUF_MEM_GIB)


@app.local_entrypoint()
def run_gguf():
    download_base.remote()
    write_result("gguf", convert_gguf.remote(), smoke=False)
