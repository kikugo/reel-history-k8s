"""The GitHub Actions workflows: pinned actions, least privilege, and the checks the guide asks for."""

import re
from pathlib import Path

import pytest
import yaml

WF = Path(__file__).resolve().parent.parent / ".github" / "workflows"
NAMES = ("ci", "release", "deploy-aks")


def load(name: str) -> dict:
    data = yaml.safe_load((WF / f"{name}.yml").read_text())
    data["on"] = data.pop(True, data.get("on"))  # PyYAML reads the key `on:` as True
    return data


def steps(wf: dict) -> list[dict]:
    return [s for job in wf["jobs"].values() for s in job["steps"]]


def runs(wf: dict) -> str:
    return "\n".join(s.get("run", "") for s in steps(wf))


@pytest.mark.parametrize("name", NAMES)
def test_every_action_is_pinned_to_a_commit_sha(name):
    for line in (WF / f"{name}.yml").read_text().splitlines():
        if "uses:" in line:
            assert re.search(r"uses: [\w./-]+@[0-9a-f]{40} # v\d", line), line.strip()


def test_permissions_are_least_privilege():
    assert load("ci")["permissions"] == {"contents": "read"}
    assert load("release")["permissions"] == {"contents": "read", "packages": "write"}
    assert load("deploy-aks")["permissions"] == {"contents": "read", "id-token": "write"}


def test_ci_runs_lint_tests_images_and_a_kind_install():
    wf = load("ci")
    assert set(wf["on"]) == {"pull_request", "push"}
    text = runs(wf)
    for cmd in ("uv run ruff check .", "uv run pytest", "make helm-lint helm-template", "make platform",
                "make images", "make helm-install", "scripts/ci_smoke.sh", "terraform -chdir=infra/aks validate"):
        assert cmd in text, cmd
    kind = next(s for s in steps(wf) if "helm/kind-action" in s.get("uses", ""))
    assert kind["with"]["config"] == "kind/cluster.yaml" and kind["with"]["cluster_name"] == "reel-history"
    matrix = {i["name"] for i in wf["jobs"]["images"]["strategy"]["matrix"]["include"]}
    assert matrix == {"recommender", "explainer", "frontend", "kserve-predictor"}
    assert all(s["with"]["push"] is False for s in steps(wf) if "build-push-action" in s.get("uses", ""))


def test_release_pushes_sha_tagged_images_only_on_version_tags():
    wf = load("release")
    assert wf["on"] == {"push": {"tags": ["v*"]}}
    assert wf["jobs"]["push-images"]["strategy"]["matrix"]["service"] == ["recommender", "explainer", "frontend"]
    text = runs(wf)
    assert "${GITHUB_SHA}" in text and "${GITHUB_SHA::7}" in text
    login = next(s for s in steps(wf) if "docker/login-action" in s.get("uses", ""))
    assert login["with"]["registry"] == "ghcr.io" and login["with"]["password"] == "${{ secrets.GITHUB_TOKEN }}"


def test_deploy_aks_is_manual_oidc_validated_and_checks_the_cluster_first():
    wf = load("deploy-aks")
    assert list(wf["on"]) == ["workflow_dispatch"]
    ordered = steps(wf)
    login = next(s for s in ordered if "azure/login" in s.get("uses", ""))
    ids = {k: v for k, v in login["with"].items() if isinstance(v, str)}
    assert ids and all(v.startswith("${{ secrets.AZURE_") for v in ids.values())
    # the identity has no roles unless a cluster exists, so login must not require a subscription
    assert login["with"]["allow-no-subscriptions"] is True and "subscription-id" not in login["with"]
    check = next(s for s in ordered if s.get("name") == "the cluster must exist")
    assert check["env"] == {"SUBSCRIPTION": "${{ secrets.AZURE_SUBSCRIPTION_ID }}"}
    names = [s.get("name", s.get("uses", "")) for s in ordered]
    assert names.index("validate the image tag") == 0
    assert names.index("the cluster must exist") < names.index("build and push images to the cluster's ACR")
    validate = ordered[0]
    assert validate["env"] == {"INPUT_TAG": "${{ inputs.image_tag }}"}  # input via env, never inlined in run
    assert "^[0-9a-f]{7,40}$" in validate["run"]
    assert "${{ inputs" not in runs(wf)
