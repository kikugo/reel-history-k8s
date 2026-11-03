"""Compare the plain recommender Deployment with the same model behind KServe (Standard mode).

caffeinate -i uv run python scripts/kserve_compare.py

Both run 2 replicas with the same image baseline, resources and model code, behind the same Envoy
Gateway, and k6 hits both through it with the same requests. Writes results/kserve/compare.json:
latency at fixed rates (interleaved, with a cool-down before each run), what each option costs in
image size, idle memory and cluster add-ons, and how long a replacement pod takes to get Ready.
"""

import json
import tempfile
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import load_test as lt  # scripts/load_test.py: k6 runner, window stats, wall-clock guard, HPA control

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "kserve" / "compare.json"
RATES = (200, 800, 1200)
BASE = "http://localhost:8088"
KSERVE_HOST = "recommender-kserve-reel-history.kserve.local"
TARGETS = {
    "plain": {"selector": "app.kubernetes.io/name=recommender", "deployment": "recommender",
              "image": "reel-history/recommender:p6"},
    "kserve": {"selector": "serving.kserve.io/inferenceservice=recommender-kserve",
               "deployment": "recommender-kserve-predictor", "image": "reel-history/kserve-predictor:dev"},
}
SAME_RESULT_LISTS = ([4973, 1172], [318, 356, 296], [2571], [], [593, 260, 2959, 480, 1, 2, 3])


def post(path: str, body: dict, host: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if host:
        headers["Host"] = host
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def same_results() -> bool:
    for liked in SAME_RESULT_LISTS:
        plain = post("/api/rec/recommend", {"liked": liked, "k": 10})["items"]
        kserve = post("/v1/models/recommender:predict", {"instances": [{"liked": liked, "k": 10}]},
                      host=KSERVE_HOST)["predictions"][0]
        if plain != kserve:
            return False
    return True


def idle(max_m: int = 50, timeout_s: int = 420) -> None:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            cpus = [m for t in TARGETS.values() for m in lt.cpu_millicores(t["selector"]).values()]
            if cpus and all(m <= max_m for m in cpus):
                return
        except lt.subprocess.CalledProcessError:
            pass
        time.sleep(10)
    raise TimeoutError("pods never went idle")


def time_to_ready(target: dict) -> float:
    """Delete one pod and time its replacement from creation to Ready (init download included)."""
    pods = lt.kubectl_json("get", "pods", "-l", target["selector"])["items"]
    victim = min(p["metadata"]["name"] for p in pods)
    lt.sh("kubectl", "-n", lt.NS, "delete", "pod", victim, "--wait=false")
    end = time.monotonic() + 600
    while time.monotonic() < end:
        for p in lt.kubectl_json("get", "pods", "-l", target["selector"])["items"]:
            if p["metadata"]["name"] == victim or p["metadata"].get("deletionTimestamp"):
                continue
            ready = next((c for c in p["status"].get("conditions", []) if c["type"] == "Ready"), None)
            if ready and ready["status"] == "True":
                created = datetime.fromisoformat(p["metadata"]["creationTimestamp"])
                became = datetime.fromisoformat(ready["lastTransitionTime"])
                if became > created:
                    return (became - created).total_seconds()
        time.sleep(2)
    raise TimeoutError(f"no replacement for {victim}")


def mem_mib(ns: str, selector: str | None = None) -> dict[str, int]:
    out = lt.sh("kubectl", "-n", ns, "top", "pods", "--no-headers", *(["-l", selector] if selector else []))
    return {line.split()[0]: int(line.split()[2].rstrip("Mi")) for line in out.splitlines() if line.strip()}


def footprint() -> dict:
    out = {}
    for name, t in TARGETS.items():
        size = int(lt.sh("docker", "image", "inspect", t["image"], "--format", "{{.Size}}").strip())
        mems = mem_mib(lt.NS, t["selector"])
        out[name] = {"image_mb": round(size / 1e6), "idle_mem_mib_per_pod": round(sum(mems.values()) / len(mems))}
    out["kserve_addons"] = {}
    for ns in ("kserve", "cert-manager"):  # what KServe needs that the plain Deployment doesn't
        mems = mem_mib(ns)
        out["kserve_addons"][ns] = {"pods": len(mems), "mem_mib": sum(mems.values())}
    crds = lt.sh("kubectl", "get", "crd", "-o", "name").split()
    out["kserve_addons"]["crds"] = sum(1 for c in crds if "serving.kserve.io" in c or "cert-manager.io" in c)
    return out


def vm_memory() -> dict:
    free = lt.sh("docker", "exec", "reel-history-control-plane", "free", "-m").split("\n")
    mem, swap = free[1].split(), free[2].split()
    return {"available_mib": int(mem[6]), "swap_used_mib": int(swap[2])}


MAX_STALL_SHARE = 0.01


def memory_stall_us() -> int:
    """Microseconds all tasks in the VM have been stalled on memory ("full" in PSI), cumulative."""
    psi = lt.sh("docker", "exec", "reel-history-control-plane", "cat", "/proc/pressure/memory")
    full = next(line for line in psi.splitlines() if line.startswith("full"))
    return int(full.rsplit("total=", 1)[1])


def measure(target: str, rate: int) -> dict:
    idle()
    stall0, t0 = memory_stall_us(), time.time()
    with tempfile.TemporaryDirectory() as tmp:
        metrics = lt.run_k6({"RATE": rate, "TARGET": target, "KSERVE_HOST": KSERVE_HOST,
                             "WARM": f"{lt.WARM_S}s", "MEASURE": f"{lt.MEASURE_S}s"},
                            Path(tmp), lt.WARM_S + lt.MEASURE_S)
    stall_share = (memory_stall_us() - stall0) / 1e6 / (time.time() - t0)
    run = {"target": target, **lt.window_stats(metrics, "measure", lt.MEASURE_S, rate)}
    run["cpu_m"] = lt.cpu_millicores(TARGETS[target]["selector"])
    run["memory_stall_share"] = round(stall_share, 5)
    run["passed"] = lt.step_passes(run)
    run["valid"] = stall_share <= MAX_STALL_SHARE
    return run


def remeasure(spec: str) -> None:
    """--only kserve:800: rerun one point that the memory check marked invalid, keep the rest of the JSON."""
    target, rate = spec.split(":")
    record = json.loads(OUT.read_text())
    try:
        lt.set_hpa(False)
        lt.wait_replicas(2)
        for attempt in (1, 2, 3):
            run = measure(target, int(rate))
            run["attempt"] = attempt
            if run["valid"]:
                break
    finally:
        lt.set_hpa(True)
    run["remeasured_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    record["runs"] = [run if (r["target"], r["offered_rps"]) == (target, int(rate)) else r for r in record["runs"]]
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{target} {rate} req/s: p95 {run['p95_ms']} ms, memory stall {run['memory_stall_share']:.2%}, "
          f"valid {run['valid']}")


def main() -> None:
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--only":
        remeasure(sys.argv[2])
        return
    record = {"started_utc": datetime.now(UTC).isoformat(timespec="seconds"), "k6_image": lt.K6_IMAGE,
              "path": "k6 (kind network) -> Envoy NodePort -> 2 pods of each target",
              "replicas": 2, "rates": list(RATES), "warm_s": lt.WARM_S, "measure_s": lt.MEASURE_S,
              "docker_mem_gib": round(int(lt.sh("docker", "info", "--format", "{{.MemTotal}}").strip()) / 2**30, 1)}
    try:
        lt.set_hpa(False)
        lt.wait_replicas(2)
        record["same_results"] = same_results()
        if not record["same_results"]:
            raise SystemExit("plain and KServe disagree on the same input; the comparison would be meaningless")
        record["vm_memory_before"] = vm_memory()
        idle()
        record["footprint"] = footprint()
        runs = []
        for rate in RATES:
            for target in ("plain", "kserve"):  # interleaved, so drift over time hits both
                for attempt in (1, 2):  # one rerun if the VM stalled on memory during the run
                    run = measure(target, rate)
                    run["attempt"] = attempt
                    if run["memory_stall_share"] <= MAX_STALL_SHARE:
                        break
                runs.append(run)
                print(f"{target:>6} {rate:>5} req/s: p50 {run['p50_ms']} ms, p95 {run['p95_ms']} ms, "
                      f"p99 {run['p99_ms']} ms, errors {run['error_rate']}, "
                      f"memory stall {run['memory_stall_share']:.2%}")
        record["runs"] = runs
        record["seconds_to_ready"] = {name: round(time_to_ready(t), 1) for name, t in TARGETS.items()}
        record["vm_memory_after"] = vm_memory()
    finally:
        lt.set_hpa(True)
    record["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
