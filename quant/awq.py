"""AWQ W4A16 quantization of the base model on an A100.

uv run modal run -m quant.awq --smoke   # 16 samples, a few minutes
uv run modal run -m quant.awq           # full run, ask first (> 15 min of A100)
"""

from quant import config
from quant.common import (
    QUANT_CPU,
    QUANT_GPU,
    QUANT_MEM_GIB,
    VOLUMES,
    app,
    download_base,
    quant_image,
    run_oneshot,
    write_result,
)


@app.function(
    image=quant_image,
    gpu=QUANT_GPU,
    cpu=QUANT_CPU,
    memory=QUANT_MEM_GIB * 1024,
    volumes=VOLUMES,
    timeout=90 * 60,
)
def quantize_awq(smoke: bool = False) -> dict:
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    spec = config.RECIPES["awq"]
    recipe = [AWQModifier(**spec["awq"]), QuantizationModifier(**spec["quant"])]
    return run_oneshot("awq", recipe, smoke)


@app.local_entrypoint()
def run_awq(smoke: bool = False):
    download_base.remote()
    write_result("awq", quantize_awq.remote(smoke), smoke)
