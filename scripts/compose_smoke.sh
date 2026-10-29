#!/usr/bin/env bash
# Checks a running `docker compose up` stack end to end, through nginx where the browser goes.
# Usage: scripts/compose_smoke.sh [base_url]   (default http://localhost:8080)
set -euo pipefail
BASE=${1:-http://localhost:8080}
fail() { echo "FAIL: $*" >&2; exit 1; }
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

for url in "$BASE/healthz" "$BASE/readyz" "http://localhost:8001/healthz" "http://localhost:8001/readyz" \
           "http://localhost:8002/healthz" "http://localhost:8002/readyz"; do
  curl -fsS "$url" >/dev/null || fail "$url"
  echo "ok   $url"
done

for svc in recommender explainer frontend; do
  uid=$(docker compose exec -T "$svc" id -u)
  [ "$uid" != "0" ] || fail "$svc runs as root"
  echo "ok   $svc runs as uid $uid"
done

hit=$(curl -fsS "$BASE/api/rec/search?q=amelie&limit=1")
echo "ok   search: $(echo "$hit" | json "d['items'][0]['title']")"
liked=$(curl -fsS "$BASE/api/rec/search?q=cinema%20paradiso&limit=1" | json "d['items'][0]['movieId']")
recs=$(curl -fsS -X POST "$BASE/api/rec/recommend" -H 'Content-Type: application/json' \
  -d "{\"liked\": [$(echo "$hit" | json "d['items'][0]['movieId']"), $liked], \"k\": 3}")
echo "ok   recommend ($(echo "$recs" | json "d['items'][0]['source']")): $(echo "$recs" | json "[i['title'] for i in d['items']]")"
first=$(echo "$recs" | json "d['items'][0]['movieId']")
curl -fsS "$BASE/api/rec/similar/$first?k=3" >/dev/null || fail "similar"
echo "ok   similar/$first"

body=$(echo "$hit" "$recs" | python3 -c "
import json,sys
dec = json.JSONDecoder(); s = sys.stdin.read(); hit, i = dec.raw_decode(s); recs, _ = dec.raw_decode(s[i:].lstrip())
pick = lambda f: {k: f[k] for k in ('title', 'year', 'genres')}
print(json.dumps({'liked': [pick(hit['items'][0])], 'candidate': pick(recs['items'][0])}))")
why=$(curl -fsS -X POST "$BASE/api/explain/explain" -H 'Content-Type: application/json' -d "$body")
echo "ok   explain ($(echo "$why" | json "d['source']")): $(echo "$why" | json "d['text']")"

curl -fsS http://localhost:8001/metrics | grep -q 'http_requests_total{handler="/recommend"' || fail "recommender metrics"
curl -fsS http://localhost:8002/metrics | grep -q 'explainer_answers_total' || fail "explainer metrics"
curl -fsS "$BASE/nginx_status" | grep -q 'Active connections' || fail "nginx_status"
echo "ok   metrics"
echo "PASS"
