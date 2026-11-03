"""Turn results/kserve/compare.json into the README's KServe block.

uv run python -m load.kserve_report

Rewrites only the text between <!-- KSERVE:START --> and <!-- KSERVE:END -->; every number in it
comes from the JSON written by scripts/kserve_compare.py.
"""

import json
from pathlib import Path

from load.report import README, update_readme

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "kserve" / "compare.json"
START, END = "<!-- KSERVE:START -->", "<!-- KSERVE:END -->"


def load(path: Path = RESULTS) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path}; run `caffeinate -i uv run python scripts/kserve_compare.py` first")
    return json.loads(path.read_text())


def render(r: dict) -> str:
    runs = {(x["target"], x["offered_rps"]): x for x in r["runs"]}
    lines = [
        (f"The same GRU4Rec + FAISS code served two ways, {r['replicas']} pods each, same resources, "
         f"behind the same Envoy Gateway; k6 sends identical requests to both (each warms for "
         f"{r['warm_s']} s, then {r['measure_s']} s are measured; runs interleave and start from idle). "
         + ("Both return identical recommendations for the same input." if r["same_results"] else "")),
        "",
        "| Offered req/s | Plain p50 / p95 / p99 (ms) | KServe p50 / p95 / p99 (ms) | Plain errors | KServe errors |",
        "|---|---|---|---|---|",
    ]
    for rate in r["rates"]:
        p, k = runs[("plain", rate)], runs[("kserve", rate)]
        trio = (lambda x: f"{x['p50_ms']:.1f} / {x['p95_ms']:.1f} / {x['p99_ms']:.1f}"
                + ("" if x.get("valid", True) else " (memory stall)"))
        lines.append(f"| {rate:,} | {trio(p)} | {trio(k)} | {p['error_rate']:.2%} | {k['error_rate']:.2%} |")
    fp, ready, addons = r["footprint"], r["seconds_to_ready"], r["footprint"]["kserve_addons"]
    extra_mem = sum(v["mem_mib"] for k, v in addons.items() if isinstance(v, dict))
    extra_pods = sum(v["pods"] for k, v in addons.items() if isinstance(v, dict))
    lines += [
        "",
        "| | Plain Deployment | KServe InferenceService |",
        "|---|---|---|",
        f"| Image size | {fp['plain']['image_mb']:,} MB | {fp['kserve']['image_mb']:,} MB |",
        f"| Idle memory per pod | {fp['plain']['idle_mem_mib_per_pod']} MiB | {fp['kserve']['idle_mem_mib_per_pod']} MiB |",
        f"| New pod to Ready (download + load) | {ready['plain']} s | {ready['kserve']} s |",
        (f"| Extra cluster components | none | {extra_pods} pods (KServe controller, cert-manager), "
         f"{extra_mem} MiB, {addons['crds']} CRDs |"),
    ]
    return "\n".join(lines)


def main() -> None:
    update_readme(render(load()), README, START, END)
    print(f"updated {README.name}")


if __name__ == "__main__":
    main()
