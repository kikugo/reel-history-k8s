"""Regenerate the README results block and the throughput chart from results/bench/*.json.

uv run python -m bench.report

Every number in the output comes from a results file; ratios are computed here.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "bench"
README = ROOT / "README.md"
CHART = RESULTS / "throughput_vs_concurrency.png"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

GPU_VARIANTS = ("bf16", "awq", "gptq")
LABELS = {"bf16": "BF16", "awq": "AWQ W4A16", "gptq": "GPTQ W4A16", "gguf": "GGUF Q4_K_M"}


def load(results_dir: Path) -> dict:
    data = {}
    for path in sorted(results_dir.glob("*.json")):
        if not path.stem.endswith("_smoke"):
            data[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _f(value, digits=1) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _level(serve: dict, concurrency: int) -> dict | None:
    return next((lv for lv in serve["levels"] if lv["concurrency"] == concurrency), None)


def headline(data: dict) -> str:
    base = data.get("serve_bf16")
    if not base:
        return ""
    lines = []
    top = max(lv["concurrency"] for lv in base["levels"])
    for v in ("awq", "gptq"):
        s = data.get(f"serve_{v}")
        if not s:
            continue
        ratios = []
        for c in (1, top):
            a, b = _level(s, c), _level(base, c)
            if a and b:
                ratios.append(f"{a['output_throughput'] / b['output_throughput']:.2f}x at concurrency {c}")
        saved = base["memory"]["weight_gib"] - s["memory"]["weight_gib"]
        lines.append(f"- **{LABELS[v]}** vs BF16: output throughput {', '.join(ratios)}; "
                     f"{saved:.2f} GiB less weight memory.")
    return "\n".join(lines)


def throughput_table(data: dict) -> str:
    rows = []
    for v in GPU_VARIANTS:
        s = data.get(f"serve_{v}")
        for lv in (s or {}).get("levels", []):
            rows.append([LABELS[v], str(lv["concurrency"]), _f(lv["output_throughput"]),
                         _f(lv["request_throughput"], 2), _f(lv.get("p50_ttft_ms")), _f(lv.get("p95_ttft_ms")),
                         _f(lv.get("p50_tpot_ms")), _f(lv.get("p95_tpot_ms"))])
    if not rows:
        return ""
    s0 = next(data[f"serve_{v}"] for v in GPU_VARIANTS if f"serve_{v}" in data)
    note = (f"`vllm bench serve` on one {s0['gpu_name']}, random prompts of {s0['input_len']} tokens in "
            f"and exactly {s0['output_len']} out.")
    header = ["Variant", "Concurrency", "Output tok/s", "Req/s", "TTFT p50 ms", "TTFT p95 ms",
              "TPOT p50 ms", "TPOT p95 ms"]
    return note + "\n\n" + _table(header, rows)


def memory_table(data: dict) -> str:
    rows = []
    for v in GPU_VARIANTS:
        s = data.get(f"serve_{v}")
        if s:
            m = s["memory"]
            rows.append([LABELS[v], _f(m["weight_gib"], 2), _f(m["kv_cache_gib"], 2),
                         _f(m["kv_cache_tokens"], 0), _f(m["max_concurrency"], 2)])
    if not rows:
        return ""
    at = next(data[f"serve_{v}"]["memory"]["max_concurrency_at_tokens"] for v in GPU_VARIANTS if f"serve_{v}" in data)
    header = ["Variant", "Weights GiB", "KV cache GiB", "KV cache tokens", f"Max concurrent {at}-token requests"]
    return "From vLLM's startup log, same `--gpu-memory-utilization` for every variant.\n\n" + _table(header, rows)


def quality_table(data: dict) -> str:
    rows = []
    for v in (*GPU_VARIANTS, "gguf"):
        p, j = data.get(f"ppl_{v}"), data.get(f"judge_{v}")
        if not p and not j:
            continue
        wt = _f(p["wikitext2"]["word_perplexity"], 2) if p else "n/a"
        fr = _f(p["frwiki"]["word_perplexity"], 2) if p else "n/a"
        if j:
            o = j["overall"]
            verdict = f"{o['win']} / {o['tie']} / {o['loss']}"
        else:
            verdict = "reference" if v == "bf16" else "n/a"
        rows.append([LABELS[v], wt, fr, verdict])
    if not rows:
        return ""
    judge = next((data[k]["judge"] for k in data if k.startswith("judge_")), None)
    note = "Word perplexity (lower is better) with lm-evaluation-harness."
    if judge:
        n = next(data[k]["n"] for k in data if k.startswith("judge_"))
        note += (f" Judge: {judge['model']} on {judge['provider']}, {n} MovieLens prompts, both answer orders; "
                 "a win or loss needs agreement in both orders.")
    return note + "\n\n" + _table(["Variant", "WikiText-2", "French Wikipedia", "Judge vs BF16 (win / tie / loss)"],
                                  rows)


def cost_table(data: dict) -> str:
    rows = []
    for v in (*GPU_VARIANTS, "gguf"):
        e = data.get(f"explain_{v}")
        if e:
            rows.append([LABELS[v], e["hardware"], str(e["concurrency"]), _f(e["generation_s"]),
                         _f(e["cost_per_1000_usd"], 3)])
    if not rows:
        return ""
    n = next(data[f"explain_{v}"]["n"] for v in (*GPU_VARIANTS, "gguf") if f"explain_{v}" in data)
    header = ["Variant", "Hardware", "Concurrency", f"Seconds for {n} explanations", "USD per 1,000 explanations"]
    return "Modal list price per second x measured seconds.\n\n" + _table(header, rows)


def cpu_table(data: dict) -> str:
    g = data.get("gguf_cpu")
    if not g:
        return ""
    rows = [[LABELS["gguf"], str(g["cpu_cores"]), _f(g["prompt"]["avg_tokens_per_s"]),
             _f(g["generation"]["avg_tokens_per_s"])]]
    note = f"`llama-bench` ({g['llama_cpp_tag']}), {g['prompt']['tokens']} prompt tokens, {g['generation']['tokens']} generated."
    return note + "\n\n" + _table(["Variant", "CPU cores", "Prompt tok/s", "Generation tok/s"], rows)


def render(data: dict) -> str:
    sections = [
        ("Headline", headline(data)),
        ("Throughput and latency", throughput_table(data)),
        ("Memory", memory_table(data)),
        ("Quality", quality_table(data)),
        ("Cost", cost_table(data)),
        ("CPU baseline", cpu_table(data)),
    ]
    body = [f"### {title}\n\n{text}" for title, text in sections if text]
    if any(k.startswith("serve_") for k in data):
        body.append(f"![Output throughput vs concurrency]({CHART.relative_to(ROOT).as_posix()})")
    if not body:
        body = ["Generated from `results/bench/*.json` once the benchmarks exist."]
    return "\n\n".join(body)


def replace_block(readme: str, block: str) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(readme):
        raise ValueError("README has no RESULTS markers")
    return pattern.sub(lambda _: f"{START}\n{block}\n{END}", readme)


def chart(data: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    for v in GPU_VARIANTS:
        s = data.get(f"serve_{v}")
        if s:
            xs = [lv["concurrency"] for lv in s["levels"]]
            ax.plot(xs, [lv["output_throughput"] for lv in s["levels"]], marker="o", label=LABELS[v])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Output tokens / s")
    ax.set_title("vLLM output throughput vs concurrency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def main(results_dir: Path = RESULTS, readme: Path = README) -> None:
    data = load(results_dir)
    readme.write_text(replace_block(readme.read_text(encoding="utf-8"), render(data)), encoding="utf-8")
    if any(k.startswith("serve_") for k in data):
        chart(data, results_dir / CHART.name)
    print(f"updated {readme} from {len(data)} result files")


if __name__ == "__main__":
    main()
