"""Check the monitoring stack against the live kind cluster.

uv run python scripts/check_dashboard.py

Sends a little traffic through the gateway, then asks Prometheus (through a port-forward) that
every app target is up and that every query in the committed Grafana dashboard returns data.
Exits non-zero on the first gap, so a dashboard that only looks right in JSON fails here.
"""

import json
import subprocess
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "charts" / "reel-history" / "dashboards" / "reel-history.json"
APP = "http://localhost:8088"
PROM_PORT = 19090  # 9090/9091 are often taken by other local stacks
NAMESPACE = "reel-history"
SERVICES = ("recommender", "explainer", "frontend")


def get(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read()
        return json.loads(raw) if raw.startswith(b"{") else {}


@contextmanager
def port_forward():
    try:
        urllib.request.urlopen(f"http://localhost:{PROM_PORT}/", timeout=2)
        raise SystemExit(f"localhost:{PROM_PORT} is already in use; free it before running this check")
    except OSError:
        pass
    proc = subprocess.Popen(
        ["kubectl", "-n", "monitoring", "port-forward", "svc/kps-kube-prometheus-stack-prometheus",
         f"{PROM_PORT}:9090"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(30):
            try:
                get(f"http://localhost:{PROM_PORT}/-/ready")
                break
            except OSError:
                time.sleep(1)
        yield f"http://localhost:{PROM_PORT}"
    finally:
        proc.terminate()


def query(prom: str, expr: str) -> list:
    url = f"{prom}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    return get(url)["data"]["result"]


def traffic(seconds: int = 75) -> None:
    """Enough requests, spread over more than one scrape window, for every rate() to have points."""
    film = {"title": "Amelie", "year": 2001, "genres": ["Comedy", "Romance"]}
    calls = [(f"{APP}/api/rec/recommend", {"liked": [4973, 1172], "k": 5}),
             (f"{APP}/api/explain/explain", {"liked": [film], "candidate": {**film, "title": "Chocolat"}}),
             (f"{APP}/", None)]
    sent = errors = 0
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for url, body in calls:
            sent += 1
            try:
                get(url, body)
            except OSError:  # a dropped request only means fewer samples; the queries decide
                errors += 1
        time.sleep(0.2)
    print(f"sent {sent} requests, {errors} errors")


def main() -> None:
    print("sending traffic for 75 s ...")
    traffic()
    failures = []
    with port_forward() as prom:
        for svc in SERVICES:
            up = query(prom, f'up{{namespace="{NAMESPACE}", service="{svc}"}}')
            ok = up and all(r["value"][1] == "1" for r in up)
            print(f"target {svc}: {len(up)} pods, {'up' if ok else 'DOWN'}")
            if not ok:
                failures.append(f"target {svc}")
        dash = json.loads(DASHBOARD.read_text())
        for panel in dash["panels"]:
            for target in panel.get("targets", []):
                expr = target["expr"].replace("$namespace", NAMESPACE)
                n = len(query(prom, expr))
                print(f"{'ok ' if n else 'EMPTY'} {panel['title']} [{target['legendFormat']}]: {n} series")
                if not n:
                    failures.append(f"{panel['title']} / {target['legendFormat']}")
    if failures:
        raise SystemExit(f"no data for: {failures}")
    print("PASS: every target is up and every dashboard query returns data")


if __name__ == "__main__":
    main()
