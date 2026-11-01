"""Phase 5 done-when checks on the kind cluster, written to results/k8s/kind_check.json.

uv run python scripts/kind_check.py            (or: make check)

1. Every pod in the app and gateway namespaces is Ready, and the app answers through Envoy.
2. Self-healing: a deleted recommender pod is replaced and the Deployment is back to full.
3. Zero-downtime rollout: while threads send requests through the gateway, `helm upgrade`
   switches all three Deployments to a new image tag; the check passes only with 0 failures.
"""

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://localhost:8080"
NS = "reel-history"
DEPLOYMENTS = ("recommender", "explainer", "frontend")
LOAD_THREADS = 4
OUT = ROOT / "results" / "k8s" / "kind_check.json"


def sh(*cmd: str, timeout: int = 600) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout, cwd=ROOT).stdout


def kubectl_json(*args: str) -> dict:
    return json.loads(sh("kubectl", "-n", NS, *args, "-o", "json"))


def request(path: str, body: dict | None = None) -> int:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0


def check_ready() -> dict:
    for ns in (NS, "envoy-gateway-system"):
        sh("kubectl", "-n", ns, "wait", "--for=condition=Ready", "pod", "--all", "--timeout=300s")
    paths = {"/": None, "/api/rec/readyz": None, "/api/explain/readyz": None}
    statuses = {p: request(p) for p in paths}
    pods = [p["metadata"]["name"] for p in kubectl_json("get", "pods")["items"]]
    return {"passed": all(s == 200 for s in statuses.values()), "statuses": statuses, "pods": pods}


def check_self_healing() -> dict:
    sel = "app.kubernetes.io/name=recommender"
    before = [p["metadata"]["name"] for p in kubectl_json("get", "pods", "-l", sel)["items"]]
    victim = before[0]
    t0 = time.monotonic()
    sh("kubectl", "-n", NS, "delete", "pod", victim, "--wait=false")
    want = kubectl_json("get", "deployment", "recommender")["spec"]["replicas"]
    deadline = t0 + 300
    while time.monotonic() < deadline:
        pods = [p for p in kubectl_json("get", "pods", "-l", sel)["items"]
                if not p["metadata"].get("deletionTimestamp")]
        ready = [p["metadata"]["name"] for p in pods
                 if any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", []))]
        if victim not in ready and len(ready) >= want:
            return {"passed": True, "deleted": victim, "pods_before": before, "pods_after": ready,
                    "seconds_to_full": round(time.monotonic() - t0, 1)}
        time.sleep(2)
    return {"passed": False, "deleted": victim, "pods_before": before}


def check_rollout(tag: str) -> dict:
    sh("make", "images", f"TAG={tag}", timeout=1200)
    stop = threading.Event()
    counts = {"total": 0, "failed": 0}
    failures: list[dict] = []
    lock = threading.Lock()

    def load(i: int) -> None:
        calls = [("/api/rec/recommend", {"liked": [4973, 1172], "k": 5}), ("/", None),
                 ("/api/explain/readyz", None)]
        n = i
        while not stop.is_set():
            path, body = calls[n % len(calls)]
            status = request(path, body)
            with lock:
                counts["total"] += 1
                if status != 200:
                    counts["failed"] += 1
                    failures.append({"path": path, "status": status, "at": round(time.monotonic() - t0, 2)})
            n += 1

    t0 = time.monotonic()
    threads = [threading.Thread(target=load, args=(i,), daemon=True) for i in range(LOAD_THREADS)]
    for t in threads:
        t.start()
    time.sleep(3)  # traffic is flowing before the upgrade starts
    sh("helm", "upgrade", "reel-history", "charts/reel-history", "-n", NS, "--reuse-values",
       "--set", f"image.tag={tag}", "--wait", "--timeout", "10m", timeout=900)
    for d in DEPLOYMENTS:
        sh("kubectl", "-n", NS, "rollout", "status", f"deployment/{d}", "--timeout=300s")
    time.sleep(10)  # old pods finish draining while load continues
    stop.set()
    for t in threads:
        t.join()
    images = {d: kubectl_json("get", "deployment", d)["spec"]["template"]["spec"]["containers"][0]["image"]
              for d in DEPLOYMENTS}
    rolled = all(img.endswith(f":{tag}") for img in images.values())
    return {"passed": rolled and counts["failed"] == 0 and counts["total"] > 0, "tag": tag, "images": images,
            "requests": counts["total"], "failed": counts["failed"], "failures": failures[:20],
            "load_seconds": round(time.monotonic() - t0, 1), "load_threads": LOAD_THREADS}


def main() -> None:
    tag = f"{sh('git', 'rev-parse', '--short', 'HEAD').strip()}-{int(time.time())}"
    record = {"started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
              "kubernetes": json.loads(sh("kubectl", "version", "-o", "json"))["serverVersion"]["gitVersion"]}
    record["ready"] = check_ready()
    print("ready:", record["ready"]["passed"])
    record["self_healing"] = check_self_healing()
    print("self-healing:", record["self_healing"]["passed"])
    record["rollout"] = check_rollout(tag)
    print("rollout:", record["rollout"]["passed"],
          f"({record['rollout']['failed']} failed of {record['rollout']['requests']})")
    record["passed"] = all(record[k]["passed"] for k in ("ready", "self_healing", "rollout"))
    record["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: passed={record['passed']}")
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
