"""Which model each served variant points at, and the vllm serve command line.

Pure Python so tests run without Modal or vLLM.
"""

from quant import config

VARIANTS = ("bf16", "awq", "gptq")
PORT = 8000
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.90


def model_path(variant: str) -> str:
    """Path of the variant's weights on the c2-models Volume."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if variant == "bf16":
        repo_dir = "models--" + config.MODEL_ID.replace("/", "--")
        return f"{config.HF_HOME}/hub/{repo_dir}/snapshots/{config.MODEL_REVISION}"
    return f"{config.MODELS_DIR}/{config.OUTPUT_NAMES[variant]}"


def served_name(variant: str) -> str:
    return f"{config.MODEL_NAME.lower()}-{variant}"


def app_name(variant: str) -> str:
    return f"c2-vllm-{variant}"


def build_vllm_cmd(variant: str, api_key: str) -> list[str]:
    return [
        "vllm", "serve", model_path(variant),
        "--served-model-name", served_name(variant),
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--seed", str(config.SEED),
        "--api-key", api_key,
    ]
