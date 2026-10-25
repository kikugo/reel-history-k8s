"""Perplexity on WikiText-2 and a French Wikipedia slice with lm-evaluation-harness (vLLM backend).

uv run modal run -m bench.ppl --variant awq --smoke
uv run modal run -m bench.ppl --variant awq
"""

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from bench.common import BENCH_CPU, BENCH_GPU, BENCH_MEM_GIB, VOLUMES, app, models, vllm_image, write_result
from quant import config
from quant.common import gpu_name, package_versions
from serve.variants import GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, model_path

TASKS_DIR = "/root/bench_tasks"
ppl_image = vllm_image.add_local_dir(Path(__file__).parent / "tasks", TASKS_DIR)

# French slice: the first articles over MIN_CHARS characters in one pinned parquet shard.
FR_REPO = "wikimedia/wikipedia"
FR_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
FR_SHARD = "20231101.fr/train-00000-of-00017.parquet"
FR_ARTICLES = 100
FR_SMOKE_ARTICLES = 5
MIN_CHARS = 2000
DATA_DIR = f"{config.VOL_MOUNT}/data"


def lm_eval_cmd(variant: str, smoke: bool, out_dir: str) -> list[str]:
    model_args = ",".join([
        f"pretrained={model_path(variant)}",
        f"max_model_len={MAX_MODEL_LEN}",
        f"max_length={MAX_MODEL_LEN}",
        f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION}",
        f"seed={config.SEED}",
    ])
    cmd = [
        "lm-eval", "run",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", "wikitext,frwiki_ppl_smoke" if smoke else "wikitext,frwiki_ppl",
        "--include_path", TASKS_DIR,
        "--batch_size", "auto",
        "--seed", str(config.SEED),
        "--output_path", out_dir,
    ]
    if smoke:
        cmd += ["--limit", "2"]
    return cmd


def pick_metrics(results: dict, task: str) -> dict:
    """lm-eval reports metrics as '<name>,none'."""
    task_res = results["results"][task]
    return {m: task_res[f"{m},none"] for m in ("word_perplexity", "byte_perplexity", "bits_per_byte")}


@app.function(image=ppl_image, volumes=VOLUMES, cpu=2, memory=8192, timeout=30 * 60)
def build_fr_slice(n_articles: int, filename: str) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(FR_REPO, FR_SHARD, repo_type="dataset", revision=FR_REVISION)
    rows = []
    for batch in pq.ParquetFile(path).iter_batches(columns=["id", "title", "text"], batch_size=1000):
        for row in batch.to_pylist():
            if len(row["text"]) > MIN_CHARS:
                rows.append({"id": row["id"], "title": row["title"], "page": row["text"]})
                if len(rows) == n_articles:
                    break
        if len(rows) == n_articles:
            break
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(DATA_DIR, filename)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    models.commit()
    return {"articles": len(rows), "chars": sum(len(r["page"]) for r in rows), "path": str(out)}


@app.function(image=ppl_image, gpu=BENCH_GPU, cpu=BENCH_CPU, memory=BENCH_MEM_GIB * 1024, volumes=VOLUMES,
              timeout=45 * 60)
def run_ppl(variant: str, smoke: bool = False) -> dict:
    out_dir = "/tmp/lmeval"
    started = datetime.now(UTC).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    subprocess.run(lm_eval_cmd(variant, smoke, out_dir), check=True)
    seconds = time.perf_counter() - t0
    results = json.loads(max(Path(out_dir).rglob("results_*.json")).read_text())
    fr_task = "frwiki_ppl_smoke" if smoke else "frwiki_ppl"
    return {
        "variant": variant,
        "smoke": smoke,
        "started_utc": started,
        "wikitext2": pick_metrics(results, "wikitext"),
        "frwiki": pick_metrics(results, fr_task),
        "frwiki_slice": {"repo": FR_REPO, "revision": FR_REVISION, "shard": FR_SHARD,
                         "articles": FR_SMOKE_ARTICLES if smoke else FR_ARTICLES, "min_chars": MIN_CHARS},
        "limit": 2 if smoke else None,
        "gpu": BENCH_GPU,
        "gpu_name": gpu_name(),
        "wall_clock_s": round(seconds, 1),
        "cost_usd": config.cost_usd(seconds, BENCH_GPU, BENCH_CPU, BENCH_MEM_GIB),
        "versions": package_versions("lm-eval", "vllm", "torch", "transformers"),
    }


@app.local_entrypoint()
def run_perplexity(variant: str, smoke: bool = False):
    n, name = (FR_SMOKE_ARTICLES, "frwiki_slice_smoke.jsonl") if smoke else (FR_ARTICLES, "frwiki_slice.jsonl")
    print(build_fr_slice.remote(n, name))
    write_result(f"ppl_{variant}", run_ppl.remote(variant, smoke), smoke)
