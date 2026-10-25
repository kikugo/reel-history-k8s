"""Settings shared by every quantization run.

Pure Python on purpose: tests import this without Modal, torch or llm-compressor.
"""

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MODEL_NAME = "Qwen3-8B"

SEED = 42

# One calibration set for both methods, so the algorithm is the only thing that changes.
# Settings from llm-compressor examples/quantization_w4a16 (perfectblend, 512 x 2048).
CALIB = {
    "dataset": "perfectblend",
    "hf_dataset": "mlabonne/open-perfectblend",
    "split": "train[:512]",
    "num_samples": 512,
    "max_seq_length": 2048,
    "seed": SEED,
}

# Cheap end-to-end check before paying for the full runs.
SMOKE_CALIB = {**CALIB, "split": "train[:16]", "num_samples": 16, "max_seq_length": 256}

# Recipes as plain data; quant/awq.py and quant/gptq.py build the modifiers from these.
RECIPES = {
    "awq": {
        "awq": {"duo_scaling": "both"},
        "quant": {"targets": ["Linear"], "scheme": "W4A16_ASYM", "ignore": ["lm_head"]},
    },
    "gptq": {
        "gptq": {"targets": "Linear", "scheme": "W4A16", "ignore": ["lm_head"]},
    },
}

OUTPUT_NAMES = {
    "awq": f"{MODEL_NAME}-W4A16-AWQ",
    "gptq": f"{MODEL_NAME}-W4A16-GPTQ",
    "gguf": f"{MODEL_NAME}-GGUF",
}

HF_USER = "karthikcr"
HF_REPOS = {method: f"{HF_USER}/{name}" for method, name in OUTPUT_NAMES.items()}

GGUF_QUANT = "Q4_K_M"
LLAMA_CPP_TAG = "v0.7.0"

# Volume layout (mounted at /vol inside every container).
VOL_MOUNT = "/vol"
HF_HOME = f"{VOL_MOUNT}/hf"
MODELS_DIR = f"{VOL_MOUNT}/models"
SMOKE_DIR = f"{VOL_MOUNT}/smoke"

# Modal list prices, USD per second. Source and date are written into every result file.
# A "A100" request is billed at the 40 GB price even when Modal upgrades it to 80 GB.
PRICE_SOURCE = "https://modal.com/pricing"
PRICE_DATE = "2026-10-24"
GPU_PRICE_PER_SEC = {
    "A100": 0.000583,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "H100": 0.001097,
}
CPU_PRICE_PER_CORE_SEC = 0.0000131
MEM_PRICE_PER_GIB_SEC = 0.00000222


def cost_usd(seconds: float, gpu: str | None = None, cpu_cores: float = 0.0, mem_gib: float = 0.0) -> float:
    """Billed cost of one container for `seconds`, from the requested resources.

    Modal bills the higher of requested and used CPU and memory, so the requested
    amounts give the bill unless the job used more than it asked for.
    """
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    if gpu is not None and gpu not in GPU_PRICE_PER_SEC:
        raise KeyError(f"no price for GPU {gpu!r}; add it to GPU_PRICE_PER_SEC")
    per_sec = GPU_PRICE_PER_SEC[gpu] if gpu else 0.0
    per_sec += cpu_cores * CPU_PRICE_PER_CORE_SEC + mem_gib * MEM_PRICE_PER_GIB_SEC
    return round(seconds * per_sec, 4)
