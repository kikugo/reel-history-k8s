"""Load test the recommender on kind with k6, write results/load/<mode>.json.

uv run python scripts/load_test.py --mode hpa-off     # recommender pinned at 2 replicas
uv run python scripts/load_test.py --mode hpa-on      # HPA, 2-5 replicas at 70% CPU
uv run python scripts/load_test.py --mode pod-kill    # steady load, one pod deleted mid-run

Step modes raise the offered rate until a step's measured window fails: p95 >= 200 ms, errors >= 1%
or k6 dropping more than 1% of the iterations it should have sent. "Sustained" is the highest
passing step. k6 runs in the grafana/k6 image on kind's Docker network, hitting Envoy's NodePort,
so Docker Desktop's port forwarding stays out of the numbers.
"""

import argparse
import csv
import gzip
import json
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "load"
K6_IMAGE = "grafana/k6:2.3.0"
NS = "reel-history"
STEPS = (100, 200, 400, 600, 800, 1000, 1200, 1500, 1800, 2200, 2600, 3000, 3500, 4000)
WARM_S, MEASURE_S = 60, 60
P95_LIMIT_MS = 200.0
MAX_ERROR_RATE = 0.01
MAX_DROPPED = 0.01
KILL_AFTER_S = 90
STEADY_S = 240


def sh(*cmd: str, timeout: int = 900) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout, cwd=ROOT).stdout


def kubectl_json(*args: str, ns: str = NS) -> dict:
    return json.loads(sh("kubectl", "-n", ns, *args, "-o", "json"))


def set_hpa(enabled: bool) -> None:
    sh("helm", "upgrade", "reel-history", "charts/reel-history", "-n", NS, "--reuse-values",
       "--set", f"services.recommender.hpa.enabled={str(enabled).lower()}", "--wait", "--timeout", "5m")
    sh("kubectl", "-n", NS, "rollout", "status", "deployment/recommender", "--timeout=300s")


def wait_replicas(want: int, timeout_s: int = 600) -> None:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        status = kubectl_json("get", "deployment", "recommender")["status"]
        if status.get("readyReplicas") == want == status.get("replicas"):
            return
        time.sleep(3)
    raise TimeoutError(f"recommender never settled at {want} replicas")


def cpu_millicores(selector: str, ns: str = NS) -> dict[str, int]:
    out = sh("kubectl", "-n", ns, "top", "pods", "-l", selector, "--no-headers")
    return {line.split()[0]: int(line.split()[1].rstrip("m")) for line in out.splitlines() if line.strip()}


def check_wall_clock(started: float, planned_s: int) -> None:
    """Refuse a run that took far longer on the wall clock than planned: the laptop slept.

    time.time(), not time.monotonic(): on macOS the monotonic clock stops while asleep.
    """
    took = time.time() - started
    if took > planned_s * 1.5 + 60:
        raise RuntimeError(f"run took {took:.0f} s for a planned {planned_s} s; the machine probably slept, "
                           "so this result would be wrong. Rerun it under `caffeinate -i`.")


def run_k6(env: dict, out_dir: Path, planned_s: int, extra: tuple[str, ...] = ()) -> dict:
    args = ["docker", "run", "--rm", "--network", "kind", "-v", f"{ROOT / 'load'}:/scripts:ro",
            "-v", f"{out_dir}:/out", "-w", "/scripts"]
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    args += [K6_IMAGE, "run", "--quiet", *extra, "recommender.js"]
    started = time.time()
    # exit 99 means a threshold failed, which is a result, not an error
    proc = subprocess.run(args, capture_output=True, text=True, timeout=1800, check=False)
    check_wall_clock(started, planned_s)
    if proc.returncode not in (0, 99):
        raise RuntimeError(f"k6 exited {proc.returncode}: {proc.stderr[-2000:]}")
    return json.loads((out_dir / "summary.json").read_text())["metrics"]


def window_stats(metrics: dict, window: str, seconds: int, rate: int) -> dict:
    dur = metrics[f"http_req_duration{{scenario:{window}}}"]["values"]
    reqs = metrics[f"http_reqs{{scenario:{window}}}"]["values"]["count"]
    failed = metrics[f"http_req_failed{{scenario:{window}}}"]["values"]["rate"]
    dropped = metrics[f"dropped_iterations{{scenario:{window}}}"]["values"]["count"]
    return {
        "offered_rps": rate,
        "achieved_rps": round(reqs / seconds, 1),
        "requests": reqs,
        "p50_ms": round(dur["med"], 2),
        "p95_ms": round(dur["p(95)"], 2),
        "p99_ms": round(dur["p(99)"], 2),
        "max_ms": round(dur["max"], 2),
        "error_rate": round(failed, 5),
        "dropped_iterations": int(dropped),
    }


def step_passes(step: dict) -> bool:
    expected = step["offered_rps"] * MEASURE_S
    return (step["p95_ms"] < P95_LIMIT_MS and step["error_rate"] < MAX_ERROR_RATE
            and step["dropped_iterations"] <= MAX_DROPPED * expected)


def sustained(steps: list[dict]) -> dict | None:
    """The highest passing step before the first failure."""
    best = None
    for step in steps:
        if not step["passed"]:
            break
        best = step
    return best


def cool_down(max_m: int = 50, timeout_s: int = 420) -> None:
    """Wait until every recommender pod is idle again.

    A new HPA reads the last minute of CPU. Enabled straight after a run that saturated the pods, it
    sees the leftover load, jumps to maxReplicas, and its 5-minute scale-down window keeps it there.
    """
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            if all(m <= max_m for m in cpu_millicores("app.kubernetes.io/name=recommender").values()):
                return
        except subprocess.CalledProcessError:
            pass
        time.sleep(10)
    raise TimeoutError("recommender CPU never settled")


def node_cpu_m() -> int:
    return int(sh("kubectl", "top", "node", "--no-headers").split()[1].rstrip("m"))


def step_mode(hpa: bool) -> dict:
    set_hpa(False)  # pins replicas back to 2, so both modes start from the same place
    wait_replicas(2)
    cool_down()
    if hpa:
        set_hpa(True)
    steps = []
    for rate in STEPS:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_k6({"RATE": rate, "WARM": f"{WARM_S}s", "MEASURE": f"{MEASURE_S}s"}, Path(tmp),
                             WARM_S + MEASURE_S)
        step = window_stats(metrics, "measure", MEASURE_S, rate)
        dep = kubectl_json("get", "deployment", "recommender")["status"]
        step["recommender_replicas"] = dep.get("readyReplicas", 0)
        try:
            step["recommender_cpu_m"] = cpu_millicores("app.kubernetes.io/name=recommender")
            step["envoy_cpu_m"] = cpu_millicores("gateway.envoyproxy.io/owning-gateway-name=reel-history",
                                                 ns="envoy-gateway-system")
            step["node_cpu_m"] = node_cpu_m()  # the whole kind node: shows when the laptop is the limit
        except subprocess.CalledProcessError:
            pass  # metrics-server can lag right after a scale-up; the step result stands
        step["passed"] = step_passes(step)
        steps.append(step)
        print(f"{rate:>5} req/s: p95 {step['p95_ms']:>7} ms, errors {step['error_rate']:.4f}, "
              f"dropped {step['dropped_iterations']}, replicas {step['recommender_replicas']} "
              f"-> {'pass' if step['passed'] else 'FAIL'}")
        if not step["passed"]:
            break
    best = sustained(steps)
    return {"steps": steps, "sustained_rps": best["offered_rps"] if best else 0,
            "sustained_achieved_rps": best["achieved_rps"] if best else 0}


def per_second(csv_path: Path, kill_ts: float) -> list[dict]:
    """Per-second request count, failures and p95 from k6's CSV output, relative to the kill."""
    buckets: dict[int, dict] = {}
    opener = gzip.open if csv_path.suffix == ".gz" else open
    with opener(csv_path, "rt") as f:
        for row in csv.DictReader(f):
            if row["metric_name"] not in ("http_req_duration", "http_req_failed"):
                continue
            sec = int(float(row["timestamp"]) - kill_ts)
            b = buckets.setdefault(sec, {"durations": [], "failed": 0})
            if row["metric_name"] == "http_req_duration":
                b["durations"].append(float(row["metric_value"]))
            elif float(row["metric_value"]) > 0:
                b["failed"] += 1
    out = []
    for sec in sorted(buckets):
        d = sorted(buckets[sec]["durations"])
        p95 = d[min(len(d) - 1, int(0.95 * len(d)))] if d else None
        out.append({"t_s": sec, "requests": len(d), "failed": buckets[sec]["failed"],
                    "p95_ms": round(p95, 2) if p95 is not None else None})
    return out


def pod_kill_mode() -> dict:
    on = json.loads((RESULTS / "hpa_on.json").read_text())
    rate = max(100, int(on["sustained_rps"] * 0.7) // 50 * 50)
    set_hpa(False)
    wait_replicas(2)
    set_hpa(True)
    kill: dict = {}

    def killer() -> None:
        time.sleep(KILL_AFTER_S)
        pods = kubectl_json("get", "pods", "-l", "app.kubernetes.io/name=recommender")["items"]
        victim = min(p["metadata"]["name"] for p in pods if not p["metadata"].get("deletionTimestamp"))
        kill["ts"] = time.time()
        sh("kubectl", "-n", NS, "delete", "pod", victim, "--wait=false")
        kill["pod"] = victim
        want = kubectl_json("get", "deployment", "recommender")["spec"]["replicas"]
        while True:
            status = kubectl_json("get", "deployment", "recommender")["status"]
            if status.get("readyReplicas", 0) >= want and status.get("updatedReplicas", 0) >= want:
                kill["seconds_to_full"] = round(time.time() - kill["ts"], 1)
                kill["replicas_at_kill"] = want
                return
            time.sleep(1)

    with tempfile.TemporaryDirectory() as tmp:
        t = threading.Thread(target=killer, daemon=True)
        t.start()
        metrics = run_k6({"RATE": rate, "MODE": "steady", "DURATION": f"{STEADY_S}s"}, Path(tmp), STEADY_S,
                         ("--out", "csv=/out/requests.csv.gz"))
        t.join(timeout=300)
        series = per_second(Path(tmp) / "requests.csv.gz", kill["ts"])
    overall = window_stats(metrics, "steady", STEADY_S, rate)
    after = [s for s in series if 0 <= s["t_s"] < 30]
    before = [s for s in series if -30 <= s["t_s"] < 0]
    return {
        "rate_rps": rate, "duration_s": STEADY_S, "kill_after_s": KILL_AFTER_S, **kill, "overall": overall,
        "failed_total": sum(s["failed"] for s in series),
        "failed_30s_after_kill": sum(s["failed"] for s in after),
        "worst_p95_ms_30s_before": max((s["p95_ms"] or 0) for s in before) if before else None,
        "worst_p95_ms_30s_after": max((s["p95_ms"] or 0) for s in after) if after else None,
        "per_second": series,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("hpa-off", "hpa-on", "pod-kill"), required=True)
    mode = parser.parse_args().mode
    if shutil.which("docker") is None:
        raise SystemExit("docker is needed to run k6")
    record = {"mode": mode, "started_utc": datetime.now(UTC).isoformat(timespec="seconds"), "k6_image": K6_IMAGE,
              "path": "k6 (kind network) -> Envoy NodePort -> recommender",
              "criteria": {"p95_ms_below": P95_LIMIT_MS, "error_rate_below": MAX_ERROR_RATE,
                           "dropped_share_at_most": MAX_DROPPED, "warm_s": WARM_S, "measure_s": MEASURE_S},
              "docker_cpus": int(sh("docker", "info", "--format", "{{.NCPU}}").strip()),
              "docker_mem_gib": round(int(sh("docker", "info", "--format", "{{.MemTotal}}").strip()) / 2**30, 1)}
    try:
        if mode == "pod-kill":
            record.update(pod_kill_mode())
        else:
            record.update(step_mode(hpa=(mode == "hpa-on")))
    finally:
        set_hpa(True)  # leave the chart's default (HPA on) behind, whatever happened
    record["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{mode.replace('-', '_')}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
