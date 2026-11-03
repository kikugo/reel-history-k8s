# Local Kubernetes for Reel History (Phases 5-6). Needs Docker, kind, kubectl and helm.
#   make kind-up platform images secrets k8s-apply   # plain manifests
#   make k8s-delete helm-install                      # same app from the Helm chart
#   make check                                        # the Phase 5 done-when checks
#   make monitoring grafana                           # Prometheus + Grafana (Phase 6)
#   make kserve kserve-deploy                         # KServe comparison (Phase 7)
#   make aks-plan aks-apply ... aks-destroy           # AKS (Phase 8); creates PAID resources, destroy same day

CLUSTER        ?= reel-history
NAMESPACE      ?= reel-history
TAG            ?= dev
CHART          := charts/reel-history
SERVICES       := recommender explainer frontend
ENVOY_GATEWAY  := v1.9.3
METRICS_SERVER := 3.14.0
KPS            := 92.17.0
CERT_MANAGER   := v1.21.3
KSERVE         := v0.21.0
TF             := terraform -chdir=infra/aks
AKS_TAG        ?= $(shell git rev-parse --short HEAD)

.PHONY: kind-up kind-down platform images secrets k8s-validate k8s-apply k8s-delete \
        helm-lint helm-template helm-install check monitoring grafana prometheus kserve kserve-deploy \
        aks-validate aks-plan aks-apply aks-credentials aks-images aks-platform aks-deploy aks-smoke aks-destroy

kind-up:
	kind get clusters | grep -qx $(CLUSTER) || kind create cluster --config kind/cluster.yaml

kind-down:
	kind delete cluster --name $(CLUSTER)

platform:
	helm upgrade --install metrics-server metrics-server --repo https://kubernetes-sigs.github.io/metrics-server/ \
		--version $(METRICS_SERVER) -n kube-system -f k8s/platform/metrics-server-values.yaml --wait
	helm upgrade --install eg oci://docker.io/envoyproxy/gateway-helm --version $(ENVOY_GATEWAY) \
		-n envoy-gateway-system --create-namespace --wait
	kubectl wait --timeout=5m -n envoy-gateway-system deployment/envoy-gateway --for=condition=Available
	kubectl apply -f k8s/platform/gatewayclass.yaml

# build with compose, retag as reel-history/<svc>:$(TAG), copy into the kind node
images:
	docker compose build
	for s in $(SERVICES); do \
		docker tag reel-history/$$s:dev reel-history/$$s:$(TAG); \
		kind load docker-image --name $(CLUSTER) reel-history/$$s:$(TAG); \
	done

# the explainer's LLM key from .env (never committed); skipped when .env has none
secrets:
	kubectl apply -f k8s/namespace.yaml
	@key=$$(grep -s '^VLLM_API_KEY=' .env | cut -d= -f2-); \
	if [ -n "$$key" ]; then \
		kubectl -n $(NAMESPACE) create secret generic explainer-llm --from-literal=VLLM_API_KEY="$$key" \
			--dry-run=client -o yaml | kubectl apply -f -; \
	else echo "no VLLM_API_KEY in .env: the explainer will use its template"; fi

k8s-validate:
	kubectl apply --dry-run=server -f k8s/namespace.yaml
	kubectl apply --dry-run=server -f k8s/

k8s-apply:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/
	kubectl -n $(NAMESPACE) rollout status deployment --timeout=5m

k8s-delete:
	kubectl delete --ignore-not-found -f k8s/recommender.yaml -f k8s/explainer.yaml \
		-f k8s/frontend.yaml -f k8s/gateway.yaml

helm-lint:
	helm lint $(CHART) -f $(CHART)/values-dev.yaml
	helm lint $(CHART) -f $(CHART)/values-aks.yaml

helm-template:
	helm template reel-history $(CHART) -n $(NAMESPACE) -f $(CHART)/values-dev.yaml > /dev/null
	helm template reel-history $(CHART) -n $(NAMESPACE) -f $(CHART)/values-aks.yaml > /dev/null
	@echo "both values files render"

helm-install:
	helm upgrade --install reel-history $(CHART) -n $(NAMESPACE) --create-namespace \
		-f $(CHART)/values-dev.yaml --set image.tag=$(TAG) --wait --timeout 5m

check:
	uv run python scripts/kind_check.py

monitoring:
	helm upgrade --install kps kube-prometheus-stack --repo https://prometheus-community.github.io/helm-charts \
		--version $(KPS) -n monitoring --create-namespace -f k8s/platform/kube-prometheus-stack-values.yaml \
		--wait --timeout 10m

# Grafana on http://localhost:13000 (3000 is often taken; user admin; password: kubectl -n monitoring get secret kps-grafana
# -o jsonpath='{.data.admin-password}' | base64 -d)
grafana:
	kubectl -n monitoring port-forward svc/kps-grafana 13000:80

# Prometheus on http://localhost:19090 (9090 is often taken by other local stacks)
prometheus:
	kubectl -n monitoring port-forward svc/kps-kube-prometheus-stack-prometheus 19090:9090

# cert-manager (KServe's webhook certificates) and KServe in Standard mode
kserve:
	helm upgrade --install cert-manager oci://quay.io/jetstack/charts/cert-manager --version $(CERT_MANAGER) \
		-n cert-manager --create-namespace --set crds.enabled=true --wait --timeout 10m
	helm upgrade --install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version $(KSERVE) -n kserve \
		--create-namespace --wait
	helm upgrade --install kserve oci://ghcr.io/kserve/charts/kserve-resources --version $(KSERVE) -n kserve \
		-f k8s/platform/kserve-values.yaml --wait --timeout 10m

# the KServe predictor image (built from services/ so it reuses the recommender package) + the ISVC
kserve-deploy:
	docker build -t reel-history/kserve-predictor:$(TAG) -f services/kserve-predictor/Dockerfile services
	kind load docker-image --name $(CLUSTER) reel-history/kserve-predictor:$(TAG)
	kubectl apply -f k8s/kserve/inferenceservice.yaml
	kubectl -n $(NAMESPACE) wait --for=condition=Ready inferenceservice/recommender-kserve --timeout=10m

# ---- AKS (Phase 8). aks-apply creates paid Azure resources: run aks-destroy the same day. ----
aks-validate:
	$(TF) fmt -check
	$(TF) init -input=false
	$(TF) validate

# reads Azure, creates nothing; saves the plan that aks-apply applies, so what was reviewed is what runs
aks-plan:
	$(TF) init -input=false
	$(TF) plan -input=false -var subscription_id=$$(az account show --query id -o tsv) -out=aks.tfplan

aks-apply:
	$(TF) apply -input=false aks.tfplan

aks-credentials:
	$$($(TF) output -raw get_credentials_command)
	kubectl config current-context

# built here for linux/amd64 (AKS nodes) and pushed with your Entra login: Azure for Students
# subscriptions can't run ACR Tasks (az acr build), and the registry has no admin user
aks-images:
	az acr login --name $$($(TF) output -raw acr_name)
	for s in $(SERVICES); do \
		docker buildx build --platform linux/amd64 --push \
			-t $$($(TF) output -raw image_repository)/$$s:$(AKS_TAG) services/$$s || exit 1; \
	done

# AKS ships metrics-server; only Envoy Gateway and the GatewayClass are needed
aks-platform:
	helm upgrade --install eg oci://docker.io/envoyproxy/gateway-helm --version $(ENVOY_GATEWAY) \
		-n envoy-gateway-system --create-namespace --wait
	kubectl wait --timeout=5m -n envoy-gateway-system deployment/envoy-gateway --for=condition=Available
	kubectl apply -f k8s/platform/gatewayclass.yaml

aks-deploy:
	helm upgrade --install reel-history $(CHART) -n $(NAMESPACE) --create-namespace \
		-f $(CHART)/values-aks.yaml \
		--set image.repository=$$($(TF) output -raw image_repository) --set image.tag=$(AKS_TAG) \
		--wait --timeout 10m

aks-smoke:
	uv run python scripts/aks_smoke.py --image-tag $(AKS_TAG)

# THE IMPORTANT ONE: removes everything (cluster, nodes, disks, load balancer, IP, registry)
aks-destroy:
	$(TF) destroy -input=false -auto-approve -var subscription_id=$$(az account show --query id -o tsv)
	uv run python scripts/aks_smoke.py --record-destroy
