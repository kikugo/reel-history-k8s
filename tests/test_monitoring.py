"""Monitoring pieces: the Grafana dashboard JSON, the chart's ServiceMonitors and the nginx exporter."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CHART = ROOT / "charts" / "reel-history"
DASHBOARD = json.loads((CHART / "dashboards" / "reel-history.json").read_text())

# metric names the dashboard may use, and who exports them
KNOWN_METRICS = {
    "http_requests_total", "http_request_duration_seconds_bucket",          # prometheus-fastapi-instrumentator
    "explainer_cache_hits_total", "explainer_cache_misses_total", "explainer_answers_total",  # explainer
    "nginx_http_requests_total", "nginx_connections_active",                 # nginx-prometheus-exporter
    "kube_deployment_status_replicas_available", "kube_horizontalpodautoscaler_status_current_replicas",
    "kube_horizontalpodautoscaler_status_desired_replicas", "kube_pod_container_resource_requests",  # kube-state-metrics
    "container_cpu_usage_seconds_total",                                     # cAdvisor via kubelet
}
PROMQL_WORDS = {"sum", "by", "rate", "histogram_quantile", "max", "or", "le", "service", "handler", "pod",
                "source", "deployment", "namespace"}


def targets():
    return [(p["title"], t) for p in DASHBOARD["panels"] for t in p.get("targets", [])]


def test_dashboard_panels_all_query_prometheus():
    panels = [p for p in DASHBOARD["panels"] if p["type"] != "row"]
    assert len(panels) >= 8
    for p in panels:
        assert p["targets"], p["title"]
        assert p["datasource"]["uid"] == "${datasource}", p["title"]
    assert DASHBOARD["uid"] == "reel-history"


def test_dashboard_uses_only_exported_metrics():
    for title, t in targets():
        names = set(re.findall(r"\b([a-z_]+)\{", t["expr"])) - PROMQL_WORDS
        assert names and names <= KNOWN_METRICS, (title, names - KNOWN_METRICS)
        assert 'namespace="$namespace"' in t["expr"], title


def test_dashboard_covers_what_the_guide_asks_for():
    exprs = " ".join(t["expr"] for _, t in targets())
    for needle in ("http_requests_total", "histogram_quantile(0.95", 'status=~"5.."',
                   "kube_deployment_status_replicas_available", "explainer_cache_hits_total"):
        assert needle in exprs, needle
    p95 = next(p for p in DASHBOARD["panels"] if p["title"].startswith("p95"))
    assert p95["fieldConfig"]["defaults"]["thresholds"]["steps"][-1]["value"] == 0.2


def render(*extra: str) -> list[dict]:
    if shutil.which("helm") is None:
        pytest.skip("helm not installed")
    out = subprocess.run(["helm", "template", "reel-history", str(CHART), "-n", "reel-history", *extra],
                         check=True, capture_output=True, text=True).stdout
    return [d for d in yaml.safe_load_all(out) if d]


def test_servicemonitors_render_with_the_crd():
    docs = render("-f", str(CHART / "values-dev.yaml"), "--api-versions", "monitoring.coreos.com/v1")
    monitors = {d["metadata"]["name"]: d for d in docs if d["kind"] == "ServiceMonitor"}
    assert set(monitors) == {"recommender", "explainer", "frontend"}
    ports = {n: m["spec"]["endpoints"][0]["port"] for n, m in monitors.items()}
    assert ports == {"recommender": "http", "explainer": "http", "frontend": "metrics"}
    assert all(m["spec"]["endpoints"][0]["path"] == "/metrics" for m in monitors.values())
    cm = next(d for d in docs if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "reel-history-dashboard")
    assert cm["metadata"]["labels"]["grafana_dashboard"] == "1"
    assert json.loads(cm["data"]["reel-history.json"])["uid"] == "reel-history"


def test_no_servicemonitors_without_the_crd_or_when_disabled():
    for extra in (["-f", str(CHART / "values-dev.yaml")], ["--api-versions", "monitoring.coreos.com/v1"]):
        kinds = {d["kind"] for d in render(*extra)}
        assert "ServiceMonitor" not in kinds


def test_frontend_exporter_sidecar_is_restricted():
    docs = render("-f", str(CHART / "values-dev.yaml"))
    dep = next(d for d in docs if d["kind"] == "Deployment" and d["metadata"]["name"] == "frontend")
    exporter = next(c for c in dep["spec"]["template"]["spec"]["containers"] if c["name"] == "metrics-exporter")
    assert exporter["image"] == "nginx/nginx-prometheus-exporter:1.5.1"
    assert "--nginx.scrape-uri=http://127.0.0.1:8080/nginx_status" in exporter["args"]
    assert exporter["securityContext"]["readOnlyRootFilesystem"] is True
    svc = next(d for d in docs if d["kind"] == "Service" and d["metadata"]["name"] == "frontend")
    assert {"name": "metrics", "port": 9113, "targetPort": "metrics"} in svc["spec"]["ports"]
