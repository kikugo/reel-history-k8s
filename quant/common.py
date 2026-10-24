"""Modal app, Volume, images and the shared oneshot runner for quant/awq.py and quant/gptq.py."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

from quant import config

app = modal.App("c2-quant")
volume = modal.Volume.from_name("c2-models", create_if_missing=True)
VOLUMES = {config.VOL_MOUNT: volume}

# Resources for the quantization containers. Cost is computed from these requests.
QUANT_GPU = "A100"
QUANT_CPU = 8
QUANT_MEM_GIB = 64

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "quant"

quant_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "llmcompressor==0.14.0.1",
        "compressed-tensors==0.19.0",
        "torch==2.13.0",
        "transformers==5.17.0",
        "datasets",
        "huggingface_hub[hf_xet]",
    )
    .env({"HF_HOME": config.HF_HOME, "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("quant")
)

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_xet]")
    .env({"HF_HOME": config.HF_HOME, "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("quant")
)


@app.function(image=download_image, volumes=VOLUMES, cpu=2, memory=4096, timeout=30 * 60)
def download_base() -> str:
    """Put the pinned base model in the HF cache on the Volume and return its snapshot path."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(config.MODEL_ID, revision=config.MODEL_REVISION)
    volume.commit()
    return path


def base_snapshot_path() -> str:
    """Snapshot path of the base model inside a container that has the Volume mounted."""
    from huggingface_hub import snapshot_download

    return snapshot_download(config.MODEL_ID, revision=config.MODEL_REVISION, local_files_only=True)


def gpu_name() -> str:
    import subprocess

    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
        return out.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return "none"


def package_versions(*names: str) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def run_record(method: str, smoke: bool, timings: dict, started: str, output_dir: str, extra: dict,
               gpu: str | None, cpu: float, mem_gib: float) -> dict:
    total = sum(timings.values())
    return {
        "method": method,
        "smoke": smoke,
        "base_model": config.MODEL_ID,
        "base_revision": config.MODEL_REVISION,
        "output_dir": output_dir,
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
        "wall_clock_s": round(total, 1),
        "hardware": {"requested_gpu": gpu, "gpu_name": gpu_name(), "cpu_cores": cpu, "mem_gib": mem_gib},
        "cost_usd": config.cost_usd(total, gpu, cpu, mem_gib),
        "price_source": {"url": config.PRICE_SOURCE, "date": config.PRICE_DATE},
        **extra,
    }


def run_oneshot(method: str, recipe, smoke: bool) -> dict:
    """Load the base model, calibrate with `recipe`, save compressed weights to the Volume.

    Runs inside the quant container. `timings` covers load, calibration and save; image
    pull and container start are billed too but are a few seconds next to these.
    """
    import torch
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    calib = config.SMOKE_CALIB if smoke else config.CALIB
    out_root = config.SMOKE_DIR if smoke else config.MODELS_DIR
    out_dir = f"{out_root}/{config.OUTPUT_NAMES[method]}"
    started = datetime.now(UTC).isoformat(timespec="seconds")
    timings = {}

    t = time.perf_counter()
    path = base_snapshot_path()
    model = AutoModelForCausalLM.from_pretrained(path, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(path)
    timings["load_s"] = time.perf_counter() - t

    if smoke:
        # Show how the Qwen3 template renders a calibration conversation (thinking tags).
        sample = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        print("Rendered calibration sample:\n" + tokenizer.apply_chat_template(sample, tokenize=False))

    # Calibration shuffling uses torch's global RNG (RandomSampler without a generator).
    torch.manual_seed(calib["seed"])
    t = time.perf_counter()
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calib["dataset"],
        splits=calib["split"],
        recipe=recipe,
        max_seq_length=calib["max_seq_length"],
        num_calibration_samples=calib["num_samples"],
    )
    timings["calibrate_s"] = time.perf_counter() - t

    t = time.perf_counter()
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)
    volume.commit()
    timings["save_s"] = time.perf_counter() - t

    extra = {
        "calibration": calib,
        "recipe": config.RECIPES[method],
        "versions": package_versions("llmcompressor", "compressed-tensors", "torch", "transformers"),
    }
    return run_record(method, smoke, timings, started, out_dir, extra, QUANT_GPU, QUANT_CPU, QUANT_MEM_GIB)


def write_result(name: str, record: dict, smoke: bool) -> Path:
    """Write a run record to results/quant/ on the local machine."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / (f"{name}_smoke.json" if smoke else f"{name}.json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {path}")
    return path
