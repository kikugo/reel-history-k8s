#!/usr/bin/env bash
# CI check on the kind cluster: every pod Ready, then /healthz and /readyz of each service through
# the Envoy gateway, one recommendation and one explanation. Usage: scripts/ci_smoke.sh [base_url]
set -euo pipefail
BASE=${1:-http://localhost:8088}

for ns in reel-history envoy-gateway-system; do
  kubectl -n "$ns" wait --for=condition=Ready pod --all --timeout=300s
done

# Envoy can take a few seconds to program the route after the Gateway reports Ready
for i in $(seq 1 30); do curl -fsS "$BASE/healthz" >/dev/null 2>&1 && break; sleep 2; done

for path in /healthz /readyz /api/rec/healthz /api/rec/readyz /api/explain/healthz /api/explain/readyz; do
  curl -fsS --retry 5 --retry-all-errors --retry-delay 2 "$BASE$path" >/dev/null
  echo "ok   $path"
done

source=$(curl -fsS -X POST "$BASE/api/rec/recommend" -H 'Content-Type: application/json' \
  -d '{"liked": [4973, 1172], "k": 3}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["items"][0]["source"])')
[ "$source" = "gru4rec" ] || { echo "FAIL: recommend answered with $source"; exit 1; }
echo "ok   recommend ($source)"

curl -fsS -X POST "$BASE/api/explain/explain" -H 'Content-Type: application/json' \
  -d '{"liked": [{"title": "Amelie", "year": 2001, "genres": ["Comedy"]}], "candidate": {"title": "Chocolat", "year": 2000, "genres": ["Comedy"]}}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["text"]; print("ok   explain (" + d["source"] + ")")'
echo PASS
