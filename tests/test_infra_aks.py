"""infra/aks invariants, checked by parsing the HCL (no Azure, no terraform binary needed)."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import hcl2
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
TF = ROOT / "infra" / "aks"
VCPUS = {"Standard_B2s_v2": 2, "Standard_B4s_v2": 4, "Standard_D2s_v3": 2, "Standard_D4s_v3": 4}
REGIONAL_VCPU_QUOTA = 6  # Azure for Students, every allowed region (az vm list-usage, Phase 8)


def unquote(v):
    return v[1:-1] if isinstance(v, str) and v.startswith('"') and v.endswith('"') else v


def parse(name: str) -> dict:
    with open(TF / name) as f:
        return hcl2.load(f)


def blocks(name: str, kind: str) -> dict:
    out = {}
    for item in parse(name)[kind]:
        for label, body in item.items():
            label = unquote(label)
            if kind == "resource":
                (n, b), = body.items()
                out[(label, unquote(n))] = b
            else:
                out[label] = body
    return out


RES = blocks("main.tf", "resource")
VARS = blocks("variables.tf", "variable")
AKS = RES[("azurerm_kubernetes_cluster", "aks")]


def default(var: str):
    return unquote(VARS[var]["default"])


def test_free_control_plane_and_required_v5_block():
    assert unquote(AKS["sku_tier"]) == "Free"
    assert unquote(AKS["node_provisioning_profile"][0]["mode"]) == "Manual"
    assert unquote(AKS["identity"][0]["type"]) == "SystemAssigned"


def test_one_small_pool_inside_the_quota():
    (pool,) = AKS["default_node_pool"]
    assert pool["vm_size"] == "${var.node_vm_size}" and pool["node_count"] == "${var.node_count}"
    assert VCPUS[default("node_vm_size")] * default("node_count") <= REGIONAL_VCPU_QUOTA
    assert "var.node_count <= 3" in json.dumps(VARS["node_count"])  # 3 x 2 vCPU is the most that fits


def test_registry_has_no_admin_user_and_kubelet_gets_acrpull_on_it_only():
    acr = RES[("azurerm_container_registry", "acr")]
    assert acr["admin_enabled"] is False and unquote(acr["sku"]) == "Basic"
    ra = RES[("azurerm_role_assignment", "kubelet_acr_pull")]
    assert unquote(ra["role_definition_name"]) == "AcrPull"
    assert ra["scope"] == "${azurerm_container_registry.acr.id}"
    assert ra["principal_id"] == "${azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id}"


def test_subscription_is_explicit_and_region_is_policy_checked():
    (provider,) = parse("versions.tf")["provider"]
    assert provider['"azurerm"']["subscription_id"] == "${var.subscription_id}"
    assert "default" not in VARS["subscription_id"]
    assert default("location") == "swedencentral"
    assert "germanywestcentral" in json.dumps(VARS["location"]["validation"])


def test_outputs_for_the_runbook_and_no_secrets():
    outputs = blocks("outputs.tf", "output")
    assert {"get_credentials_command", "image_repository", "acr_name", "resource_group"} <= set(outputs)
    assert "az aks get-credentials" in outputs["get_credentials_command"]["value"]
    text = (TF / "outputs.tf").read_text()
    assert "kube_config" not in text and "sensitive" not in text


def test_state_and_variable_files_are_gitignored():
    ignore = (ROOT / ".gitignore").read_text()
    for pattern in ("infra/**/*.tfstate", "infra/**/*.tfvars", "infra/**/.terraform/"):
        assert pattern in ignore


def test_aks_values_render_a_load_balancer_with_the_acr_repository():
    if shutil.which("helm") is None:
        pytest.skip("helm not installed")
    chart = ROOT / "charts" / "reel-history"
    out = subprocess.run(["helm", "template", "reel-history", str(chart), "-f", str(chart / "values-aks.yaml"),
                          "--set", "image.repository=example.azurecr.io/reel-history", "--set", "image.tag=abc1234"],
                         check=True, capture_output=True, text=True).stdout
    docs = [d for d in yaml.safe_load_all(out) if d]
    proxy = next(d for d in docs if d["kind"] == "EnvoyProxy")["spec"]["provider"]["kubernetes"]["envoyService"]
    assert proxy == {"type": "LoadBalancer"}
    images = {c["image"] for d in docs if d["kind"] == "Deployment" for c in d["spec"]["template"]["spec"]["containers"]
              if c["name"] != "metrics-exporter"}
    assert images == {f"example.azurecr.io/reel-history/{s}:abc1234" for s in ("recommender", "explainer", "frontend")}


def _smoke():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("aks_smoke", ROOT / "scripts" / "aks_smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_spread_helpers():
    smoke = _smoke()
    pods = [{"spec": {"nodeName": n}, "status": {"phase": ph}}
            for n, ph in (("a", "Running"), ("b", "Running"), ("a", "Running"), ("b", "Pending"))]
    assert smoke.spread(pods) == {"a": 2, "b": 1}
    assert smoke.spread_ok({"a": 2, "b": 1}, nodes=2)
    assert not smoke.spread_ok({"a": 6}, nodes=2)
    assert smoke.spread_ok({"a": 6}, nodes=1)


def test_all_digit_git_sha_renders_as_a_string_tag():
    """A short SHA like 4570436 arrives from --set as a number; the image must still be ...:4570436."""
    if shutil.which("helm") is None:
        pytest.skip("helm not installed")
    chart = ROOT / "charts" / "reel-history"
    out = subprocess.run(["helm", "template", "reel-history", str(chart), "-f", str(chart / "values-aks.yaml"),
                          "--set", "image.repository=example.azurecr.io/reel-history", "--set", "image.tag=4570436"],
                         check=True, capture_output=True, text=True).stdout
    assert "%!s(int64" not in out
    assert "image: example.azurecr.io/reel-history/recommender:4570436" in out
    makefile = (ROOT / "Makefile").read_text()
    assert "--set image.tag=" not in makefile and "--set-string image.tag=" in makefile


def test_ci_identity_gets_least_privilege_roles_only_with_the_cluster():
    ci = RES[("azurerm_role_assignment", "ci")]
    assert ci["principal_id"] == "${var.ci_principal_id}"
    text = (TF / "main.tf").read_text()
    for role in ('"Reader"', '"Azure Kubernetes Service Cluster User Role"', '"AcrPush"'):
        assert role in text, role
    assert "Contributor" not in text and "Owner" not in text
    assert 'var.ci_principal_id == "" ? {}' in text  # no identity, no assignments
    assert default("ci_principal_id") == ""
    makefile = (ROOT / "Makefile").read_text()
    assert makefile.count("-var ci_principal_id=$(CI_PRINCIPAL)") == 2  # plan and destroy
