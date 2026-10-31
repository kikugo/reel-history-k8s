# Local Kubernetes for Reel History (Phase 5). Needs Docker, kind, kubectl and helm.
#   make kind-up platform images secrets k8s-apply   # plain manifests
#   make k8s-delete helm-install                      # same app from the Helm chart
#   make check                                        # the Phase 5 done-when checks

CLUSTER        ?= reel-history
NAMESPACE      ?= reel-history
TAG            ?= dev
CHART          := charts/reel-history
SERVICES       := recommender explainer frontend
ENVOY_GATEWAY  := v1.9.3
METRICS_SERVER := 3.14.0

.PHONY: kind-up kind-down platform images secrets k8s-validate k8s-apply k8s-delete \
        helm-lint helm-template helm-install check

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
