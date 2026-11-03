"""Smoke-test the AKS deployment and record it in results/aks/smoke.json.

uv run python scripts/aks_smoke.py --image-tag <sha>      # after make aks-deploy (make aks-smoke)
uv run python scripts/aks_smoke.py --record-destroy        # after make aks-destroy

Checks every pod is Ready, the app pods spread over the nodes, the Gateway's public address serves
the frontend, a GRU4Rec recommendation and an explanation, and a deleted recommender pod comes
back. The record keeps cluster facts and timings, never IPs or credentials.
"""

import argparse
import json
import subprocess
import time
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import kind_check  # scripts/kind_check.py: shared kubectl helpers and the self-healing check

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "aks" / "smoke.json"
NS = kind_check.NS
RESOURCE_GROUP = "rg-reel-history-aks"
APP = ("recommender", "explainer", "frontend")


def spread(pods: list[dict]) -> dict[str, int]:
    """Running app pods per node."""
    return dict(Counter(p["spec"].get("nodeName", "?") for p in pods if p["status"].get("phase") == "Running"))


def spread_ok(per_node: dict[str, int], nodes: int) -> bool:
    return len(per_node) >= min(2, nodes)


def gateway_address(timeout_s: int = 600) -> str:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        status = kind_check.kubectl_json("get", "gateway", "reel-history").get("status", {})
        if addrs := status.get("addresses"):
            return addrs[0]["value"]
        time.sleep(5)
    raise TimeoutError("the Gateway never got a public address")


def call(base: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    end = time.monotonic() + 180  # a new Azure load balancer can take a minute to answer
    while True:
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                return resp.status, (json.loads(raw) if raw.startswith(b"{") else {})
        except OSError:
            if time.monotonic() > end:
                return 0, {}
            time.sleep(5)


def cluster_facts() -> dict:
    nodes = json.loads(kind_check.sh("kubectl", "get", "nodes", "-o", "json"))["items"]
    labels = nodes[0]["metadata"]["labels"]
    return {
        "nodes": len(nodes),
        "node_vm_size": labels.get("node.kubernetes.io/instance-type"),
        "region": labels.get("topology.kubernetes.io/region"),
        "kubernetes": nodes[0]["status"]["nodeInfo"]["kubeletVersion"],
        "context": kind_check.sh("kubectl", "config", "current-context").strip(),
    }


def smoke(image_tag: str) -> dict:
    facts = cluster_facts()
    if not facts["context"].startswith("aks-"):
        raise SystemExit(f"kubectl points at {facts['context']!r}, not the AKS cluster; run make aks-credentials")
    record = {"started_utc": datetime.now(UTC).isoformat(timespec="seconds"), **facts}
    t0 = time.monotonic()
    for ns in (NS, "envoy-gateway-system"):
        kind_check.sh("kubectl", "-n", ns, "wait", "--for=condition=Ready", "pod", "--all", "--timeout=600s")
    base = f"http://{gateway_address()}"
    record["seconds_to_ready"] = round(time.monotonic() - t0, 1)

    pods = [p for p in kind_check.kubectl_json("get", "pods")["items"]
            if p["metadata"]["labels"].get("app.kubernetes.io/name") in APP]
    per_node = spread(pods)
    record["pods_per_node"] = sorted(per_node.values(), reverse=True)  # counts only, no node names
    record["spread_ok"] = spread_ok(per_node, facts["nodes"])

    film = {"title": "Amelie", "year": 2001, "genres": ["Comedy", "Romance"]}
    checks = {
        "frontend": call(base, "/")[0] == 200,
        "recommender_ready": call(base, "/api/rec/readyz")[0] == 200,
        "explainer_ready": call(base, "/api/explain/readyz")[0] == 200,
    }
    status, rec = call(base, "/api/rec/recommend", {"liked": [4973, 1172], "k": 5})
    checks["recommend_gru4rec"] = status == 200 and {i["source"] for i in rec.get("items", [])} == {"gru4rec"}
    status, why = call(base, "/api/explain/explain", {"liked": [film], "candidate": {**film, "title": "Chocolat"}})
    checks["explain"] = status == 200 and bool(why.get("text"))
    record["explain_source"] = why.get("source")
    record["checks"] = checks

    record["self_healing"] = kind_check.check_self_healing()
    record["images"] = sorted({c["image"].rsplit("/", 1)[-1] for d in APP
                               for c in kind_check.kubectl_json("get", "deployment", d)
                               ["spec"]["template"]["spec"]["containers"]})
    record["image_tag"] = image_tag
    record["passed"] = all(checks.values()) and record["spread_ok"] and record["self_healing"]["passed"]
    record["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    return record


def record_destroy() -> dict:
    record = json.loads(OUT.read_text())
    exists = subprocess.run(["az", "group", "exists", "-n", RESOURCE_GROUP], check=True,
                            capture_output=True, text=True).stdout.strip()
    record["destroyed_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    record["resource_group_exists_after_destroy"] = exists == "true"
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-tag")
    parser.add_argument("--record-destroy", action="store_true")
    args = parser.parse_args()
    record = record_destroy() if args.record_destroy else smoke(args.image_tag)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in record if k in ("passed", "checks", "resource_group_exists_after_destroy")}))
    if record.get("resource_group_exists_after_destroy"):
        raise SystemExit("rg-reel-history-aks still exists: delete it now (az group delete -n rg-reel-history-aks)")
    if not args.record_destroy and not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
