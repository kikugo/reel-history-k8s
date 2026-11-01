"""Invariants for the plain manifests in k8s/ and the Helm chart, checked without a cluster.

The chart tests render with `helm template` and are skipped when helm isn't installed.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CHART = ROOT / "charts" / "reel-history"
SERVICES = {"recommender": 8000, "explainer": 8000, "frontend": 8080}


def load_docs(text: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(text) if d]


def plain_docs() -> list[dict]:
    return [d for p in sorted((ROOT / "k8s").rglob("*.yaml")) if "values" not in p.name
            for d in load_docs(p.read_text())]


def helm_docs(values: str) -> list[dict]:
    if shutil.which("helm") is None:
        pytest.skip("helm not installed")
    out = subprocess.run(
        ["helm", "template", "reel-history", str(CHART), "-n", "reel-history", "-f", str(CHART / values)],
        check=True, capture_output=True, text=True).stdout
    return load_docs(out)


SOURCES = {
    "plain": plain_docs,
    "helm-dev": lambda: helm_docs("values-dev.yaml"),
    "helm-aks": lambda: helm_docs("values-aks.yaml"),
}


@pytest.fixture(params=list(SOURCES))
def docs(request) -> list[dict]:
    return SOURCES[request.param]()


def by_kind(docs: list[dict], kind: str) -> dict[str, dict]:
    return {d["metadata"]["name"]: d for d in docs if d["kind"] == kind}


def test_each_service_has_configmap_deployment_and_service(docs):
    for kind in ("ConfigMap", "Deployment", "Service"):
        assert set(SERVICES) <= set(by_kind(docs, kind)), kind


def test_deployments_follow_the_baseline(docs):
    for name, dep in by_kind(docs, "Deployment").items():
        spec = dep["spec"]
        assert spec["strategy"]["rollingUpdate"]["maxUnavailable"] == 0, name
        pod = spec["template"]["spec"]
        assert pod["securityContext"]["runAsNonRoot"] is True, name
        assert pod["securityContext"]["runAsUser"] != 0, name
        for c in pod["containers"] + pod.get("initContainers", []):
            sc = c["securityContext"]
            assert sc["readOnlyRootFilesystem"] is True and sc["allowPrivilegeEscalation"] is False, c["name"]
            assert sc["capabilities"]["drop"] == ["ALL"], c["name"]
            assert {"cpu", "memory"} <= set(c["resources"]["requests"]) and c["resources"]["limits"], c["name"]
            tag = c["image"].rsplit(":", 1)[1]
            assert tag and tag != "latest", c["image"]
        (app,) = pod["containers"]
        assert app["readinessProbe"]["httpGet"]["path"] == "/readyz", name
        assert app["livenessProbe"]["httpGet"]["path"] == "/healthz", name
        assert app["lifecycle"]["preStop"]["sleep"]["seconds"] > 0, name
        assert app["ports"][0]["containerPort"] == SERVICES[name]


def test_recommender_fetches_artifacts_before_starting(docs):
    pod = by_kind(docs, "Deployment")["recommender"]["spec"]["template"]["spec"]
    (init,) = pod["initContainers"]
    assert init["command"][-1] == "recommender.fetch"
    mounts = {m["name"]: m for m in pod["containers"][0]["volumeMounts"]}
    assert mounts["artifacts"]["readOnly"] is True
    assert "startupProbe" in pod["containers"][0]


def test_recommender_hpa_scales_on_cpu(docs):
    hpa = by_kind(docs, "HorizontalPodAutoscaler")["recommender"]["spec"]
    assert hpa["scaleTargetRef"]["name"] == "recommender"
    assert hpa["minReplicas"] >= 2
    assert hpa["metrics"][0]["resource"]["name"] == "cpu"
    cpu_requested = by_kind(docs, "Deployment")["recommender"]["spec"]["template"]["spec"]["containers"][0]
    assert cpu_requested["resources"]["requests"]["cpu"]


def test_route_sends_prefixes_to_the_right_services(docs):
    (route,) = by_kind(docs, "HTTPRoute").values()
    table = {r["matches"][0]["path"]["value"]: r for r in route["spec"]["rules"]}
    assert table["/api/rec/"]["backendRefs"][0]["name"] == "recommender"
    assert table["/api/explain/"]["backendRefs"][0]["name"] == "explainer"
    assert table["/"]["backendRefs"][0]["name"] == "frontend"
    for prefix in ("/api/rec/", "/api/explain/"):
        rewrite = table[prefix]["filters"][0]["urlRewrite"]["path"]
        assert rewrite == {"type": "ReplacePrefixMatch", "replacePrefixMatch": "/"}
    (gateway,) = by_kind(docs, "Gateway").values()
    assert gateway["spec"]["gatewayClassName"] == "eg"


def test_no_secret_values_are_committed(docs):
    assert by_kind(docs, "Secret") == {}
    env_from = by_kind(docs, "Deployment")["explainer"]["spec"]["template"]["spec"]["containers"][0]["envFrom"]
    assert {"secretRef": {"name": "explainer-llm", "optional": True}} in env_from


def test_envoy_service_type_per_environment():
    proxies = {k: by_kind(SOURCES[k](), "EnvoyProxy") for k in SOURCES}
    svc = {k: next(iter(v.values()))["spec"]["provider"]["kubernetes"]["envoyService"] for k, v in proxies.items()}
    assert svc["plain"]["type"] == svc["helm-dev"]["type"] == "NodePort"
    assert svc["helm-dev"]["patch"]["value"]["spec"]["ports"][0]["nodePort"] == 30080
    assert svc["helm-aks"]["type"] == "LoadBalancer" and "patch" not in svc["helm-aks"]


def test_chart_creates_secret_only_when_a_key_is_passed():
    if shutil.which("helm") is None:
        pytest.skip("helm not installed")
    out = subprocess.run(
        ["helm", "template", "reel-history", str(CHART), "--set", "llmApiKey=test-key", "-s", "templates/secret.yaml"],
        check=True, capture_output=True, text=True).stdout
    (secret,) = load_docs(out)
    assert secret["metadata"]["name"] == "explainer-llm"
    assert secret["stringData"] == {"VLLM_API_KEY": "test-key"}


def test_hpa_owned_deployments_omit_replicas():
    deps = by_kind(helm_docs("values-dev.yaml"), "Deployment")
    assert "replicas" not in deps["recommender"]["spec"]  # helm upgrade must not fight the HPA
    assert deps["explainer"]["spec"]["replicas"] == 2


def test_kind_maps_the_gateway_nodeport():
    cluster = yaml.safe_load((ROOT / "kind" / "cluster.yaml").read_text())
    (mapping,) = cluster["nodes"][0]["extraPortMappings"]
    assert mapping["containerPort"] == 30080 and mapping["listenAddress"] == "127.0.0.1"
