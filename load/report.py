"""Turn results/load/*.json into the README's load-test block and a p95-vs-rate chart.

uv run python -m load.report

Rewrites only the text between <!-- LOAD:START --> and <!-- LOAD:END -->; every number in it
comes from the JSON written by scripts/load_test.py.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "load"
README = ROOT / "README.md"
CHART = RESULTS / "p95_vs_rate.png"
START, END = "<!-- LOAD:START -->", "<!-- LOAD:END -->"
MODES = {"hpa_off": "HPA off (2 replicas)", "hpa_on": "HPA on (2-5 replicas)"}


def load(results: Path = RESULTS) -> dict:
    missing = [f"{m}.json" for m in (*MODES, "pod_kill") if not (results / f"{m}.json").exists()]
    if missing:
        raise SystemExit(f"missing {missing} in {results}; run scripts/load_test.py for each mode first")
    return {m: json.loads((results / f"{m}.json").read_text()) for m in (*MODES, "pod_kill")}


def best_step(run: dict) -> dict | None:
    return next((s for s in run["steps"] if s["offered_rps"] == run["sustained_rps"]), None)


def render(data: dict, chart_rel: str = "results/load/p95_vs_rate.png") -> str:
    off, kill = data["hpa_off"], data["pod_kill"]
    crit = off["criteria"]
    lines = [
        (f"k6 against the recommender through Envoy on the kind cluster. Open-model load at a fixed rate; "
         f"each step warms for {crit['warm_s']} s and is judged on the next {crit['measure_s']} s: "
         f"p95 under {crit['p95_ms_below']:.0f} ms, errors under {crit['error_rate_below']:.0%}, and k6 able to "
         f"send what it offered. Everything shares one laptop ({off['docker_cpus']} cores in Docker Desktop), "
         "so these say what this machine sustains, and the two modes compare fairly with each other."),
        "",
        "| | Sustained req/s at p95 < 200 ms | Achieved req/s | p95 at that rate (ms) | Recommender pods |",
        "|---|---|---|---|---|",
    ]
    for key, label in MODES.items():
        run, best = data[key], best_step(data[key])
        if best is None:
            lines.append(f"| {label} | none passed | | | |")
            continue
        lines.append(f"| {label} | {run['sustained_rps']:,} | {best['achieved_rps']:,.0f} | "
                     f"{best['p95_ms']:.1f} | {best['recommender_replicas']} |")
    lines += ["", "| Offered req/s | HPA off p95 (ms) | HPA off | HPA on p95 (ms) | HPA on pods | HPA on |",
              "|---|---|---|---|---|---|"]
    by_rate = {}
    for key in MODES:
        for s in data[key]["steps"]:
            by_rate.setdefault(s["offered_rps"], {})[key] = s
    for rate in sorted(by_rate):
        o, n = by_rate[rate].get("hpa_off"), by_rate[rate].get("hpa_on")
        cell = (lambda s: (f"{s['p95_ms']:.1f}", "pass" if s["passed"] else "**fail**") if s else ("", ""))
        lines.append(f"| {rate:,} | {cell(o)[0]} | {cell(o)[1]} | {cell(n)[0]} | "
                     f"{n['recommender_replicas'] if n else ''} | {cell(n)[1]} |")
    ov = kill["overall"]
    lines += [
        "",
        (f"**Pod kill:** a steady {kill['rate_rps']:,} req/s for {kill['duration_s']} s with the HPA on; "
         f"`{kill['pod']}` was deleted {kill['kill_after_s']} s in. "
         f"{kill['failed_total']:,} of {ov['requests']:,} requests failed "
         f"({kill['failed_30s_after_kill']:,} in the 30 s after the kill). "
         f"Worst per-second p95: {kill['worst_p95_ms_30s_before']:.1f} ms in the 30 s before, "
         f"{kill['worst_p95_ms_30s_after']:.1f} ms in the 30 s after. "
         f"The Deployment was back to {kill['replicas_at_kill']} ready pods {kill['seconds_to_full']} s later."),
        "",
        f"![p95 latency vs offered rate]({chart_rel})",
        "",
        ("Rerun: `uv run python scripts/load_test.py --mode hpa-off` (then `hpa-on`, `pod-kill`), "
         "then `uv run python -m load.report`."),
    ]
    return "\n".join(lines)


def plot(data: dict, path: Path = CHART) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in MODES.items():
        steps = data[key]["steps"]
        ax.plot([s["offered_rps"] for s in steps], [s["p95_ms"] for s in steps], marker="o", label=label)
    ax.axhline(data["hpa_off"]["criteria"]["p95_ms_below"], color="red", linestyle="--", label="200 ms target")
    ax.set_xlabel("Offered load (req/s)")
    ax.set_ylabel("p95 latency (ms)")
    ax.set_yscale("log")
    ax.set_title("Recommender p95 vs load on kind")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def update_readme(block: str, readme: Path = README, start: str = START, end: str = END) -> None:
    text = readme.read_text()
    if start not in text or end not in text:
        raise SystemExit(f"{readme} has no {start} / {end} markers")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(f"{head}{start}\n{block}\n{end}{tail}")


def main() -> None:
    data = load()
    plot(data)
    update_readme(render(data))
    print(f"updated {README.name} and {CHART.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
