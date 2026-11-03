"""The InferenceService matches the plain recommender, so the comparison differs only in serving."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ISVC = yaml.safe_load((ROOT / "k8s" / "kserve" / "inferenceservice.yaml").read_text())
PLAIN = next(d for d in yaml.safe_load_all((ROOT / "k8s" / "recommender.yaml").read_text())
             if d and d["kind"] == "Deployment")


def test_standard_mode_with_fixed_replicas():
    assert ISVC["kind"] == "InferenceService"
    assert ISVC["metadata"]["annotations"]["serving.kserve.io/deploymentMode"] == "Standard"
    pred = ISVC["spec"]["predictor"]
    assert pred["minReplicas"] == pred["maxReplicas"] == 2
    assert pred["deploymentStrategy"]["rollingUpdate"] == {"maxUnavailable": 0, "maxSurge": 1}


def test_same_resources_init_and_security_as_the_plain_recommender():
    pred = ISVC["spec"]["predictor"]
    plain = PLAIN["spec"]["template"]["spec"]
    (container,) = pred["containers"]
    assert container["resources"] == plain["containers"][0]["resources"]
    assert pred["initContainers"][0]["command"][-1] == "recommender.fetch"
    assert pred["securityContext"]["runAsNonRoot"] is True
    for c in pred["containers"] + pred["initContainers"]:
        assert c["securityContext"]["readOnlyRootFilesystem"] is True
        assert c["securityContext"]["capabilities"]["drop"] == ["ALL"]
    assert container["readinessProbe"]["httpGet"]["path"] == "/v1/models/recommender"


def test_port_is_http1_so_envoy_speaks_http1():
    (port,) = ISVC["spec"]["predictor"]["containers"][0]["ports"]
    assert port["name"] == "http1" and port["containerPort"] == 8080
