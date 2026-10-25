"""Hugging Face model cards, rendered from results/quant/<method>.json.

No number is typed here: every value in a card comes from the run record.
"""

import json
from pathlib import Path

from quant import config

GITHUB_REPO = "https://github.com/kikugo/reel-history-k8s"

FRONT_MATTER = """---
base_model: {base_model}
license: apache-2.0
pipeline_tag: text-generation
tags:
{tags}
---
"""

RUN_TABLE = """| | |
|---|---|
| Base model | [{base_model}](https://huggingface.co/{base_model}) @ `{base_revision}` |
| Hardware | {gpu_name} (Modal `{requested_gpu}`), {cpu_cores} CPU cores, {mem_gib} GiB RAM requested |
| Wall-clock | {wall_clock_s} s ({timings}) |
| Compute cost | ${cost_usd} at Modal list prices ({price_url}, {price_date}) |
| Finished | {finished_utc} |
"""

LLMC_BODY = """# {name}

{method_upper} 4-bit weight quantization (W4A16) of {base_model}, made with
[llm-compressor](https://github.com/vllm-project/llm-compressor) for a personal benchmarking project:
{github}. Weights are stored in the compressed-tensors format and load directly in vLLM.

## How it was made

| Setting | Value |
|---|---|
| Recipe | `{recipe}` |
| Calibration data | [{hf_dataset}](https://huggingface.co/datasets/{hf_dataset}), split `{split}` |
| Calibration samples | {num_samples} |
| Max sequence length | {max_seq_length} |
| Seed | {seed} |
| Versions | {versions} |

The script that produced this model is `quant/{method}.py` in the GitHub repo.

## Run

{run_table}
## Use with vLLM

```bash
vllm serve {repo}
```

Qwen3 thinks by default. Pass `chat_template_kwargs={{"enable_thinking": false}}` for direct answers.

## Benchmarks

Throughput, latency and quality against the BF16 model are in the GitHub repo's README, generated from
its `results/` folder.
"""

GGUF_BODY = """# {name}

{quant_type} GGUF of {base_model}, converted and quantized with
[llama.cpp](https://github.com/ggml-org/llama.cpp) `{llama_cpp_tag}` for a personal benchmarking
project: {github}. It is the CPU baseline next to the AWQ and GPTQ vLLM variants.

## How it was made

`convert_hf_to_gguf.py --outtype f16`, then `llama-quantize ... {quant_type}`, without an importance
matrix (imatrix: {imatrix}). The script is `quant/gguf.sh` in the GitHub repo.

| File | Bytes |
|---|---|
| `{file}` | {quant_bytes} |

## Run

{run_table}
## Use with llama.cpp

```bash
llama-cli -hf {repo}
```
"""


def _run_table(rec: dict) -> str:
    hw = rec["hardware"]
    timings = ", ".join(f"{k.removesuffix('_s')} {v} s" for k, v in rec["timings_s"].items())
    return RUN_TABLE.format(
        base_model=rec["base_model"],
        base_revision=rec["base_revision"],
        gpu_name=hw["gpu_name"],
        requested_gpu=hw["requested_gpu"] or "CPU only",
        cpu_cores=hw["cpu_cores"],
        mem_gib=hw["mem_gib"],
        wall_clock_s=rec["wall_clock_s"],
        timings=timings,
        cost_usd=rec["cost_usd"],
        price_url=rec["price_source"]["url"],
        price_date=rec["price_source"]["date"],
        finished_utc=rec["finished_utc"],
    )


def render(rec: dict) -> str:
    method = rec["method"]
    repo = config.HF_REPOS[method]
    name = config.OUTPUT_NAMES[method]
    if method == "gguf":
        tags = ["gguf", "llama.cpp", "qwen3"]
        body = GGUF_BODY.format(
            name=name,
            quant_type=rec["quant_type"],
            base_model=rec["base_model"],
            llama_cpp_tag=rec["llama_cpp_tag"],
            github=GITHUB_REPO,
            imatrix=str(rec["imatrix"]).lower(),
            file=gguf_filename(rec),
            quant_bytes=rec["file_bytes"]["quant"],
            run_table=_run_table(rec),
            repo=repo,
        )
    else:
        cal = rec["calibration"]
        tags = ["llmcompressor", "compressed-tensors", "vllm", "w4a16", method, "qwen3"]
        body = LLMC_BODY.format(
            name=name,
            method=method,
            method_upper=method.upper(),
            base_model=rec["base_model"],
            github=GITHUB_REPO,
            recipe=json.dumps(rec["recipe"]),
            hf_dataset=cal["hf_dataset"],
            split=cal["split"],
            num_samples=cal["num_samples"],
            max_seq_length=cal["max_seq_length"],
            seed=cal["seed"],
            versions=", ".join(f"{k} {v}" for k, v in rec["versions"].items()),
            run_table=_run_table(rec),
            repo=repo,
        )
    front = FRONT_MATTER.format(base_model=rec["base_model"], tags="\n".join(f"- {t}" for t in tags))
    return front + "\n" + body


def gguf_filename(rec: dict) -> str:
    return f"{config.MODEL_NAME}-{rec['quant_type']}.gguf"


def load_record(method: str, results_dir: Path | None = None) -> dict:
    results_dir = results_dir or Path(__file__).resolve().parent.parent / "results" / "quant"
    rec = json.loads((results_dir / f"{method}.json").read_text())
    if rec.get("smoke"):
        raise ValueError(f"{method}.json is a smoke run; model cards come from full runs only")
    return rec
