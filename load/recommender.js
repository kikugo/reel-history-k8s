// k6 load test for the recommender, through the Envoy gateway.
//
// Open model: requests arrive at RATE per second whatever the latency, the way users do.
//   step mode (default): `warm` at RATE for WARM, then `measure` at RATE for MEASURE. Thresholds on
//     the `measure` scenario make k6 report p95 and the error rate for that window only, after the
//     HPA has had the warm-up to react.
//   steady mode (MODE=steady): one scenario at RATE for DURATION, for the pod-kill run.
// Run by scripts/load_test.py inside the grafana/k6 image on kind's Docker network.
import http from 'k6/http';
import { check } from 'k6';

const BASE = __ENV.BASE_URL || 'http://reel-history-control-plane:30080';
const RATE = parseInt(__ENV.RATE || '10', 10);
const MODE = __ENV.MODE || 'step';
// 200 real catalog movieIds; each request likes a slice of 1 to 20 of them
const POOL = JSON.parse(open('./liked_pool.json'));

const arrival = (extra) => Object.assign({
  executor: 'constant-arrival-rate',
  rate: RATE,
  timeUnit: '1s',
  preAllocatedVUs: Math.max(10, RATE),
  maxVUs: Math.max(50, RATE * 4),
  gracefulStop: '5s',
}, extra);

const scenarios = MODE === 'steady'
  ? { steady: arrival({ duration: __ENV.DURATION || '240s' }) }
  : {
      warm: arrival({ duration: __ENV.WARM || '60s' }),
      measure: arrival({ duration: __ENV.MEASURE || '60s', startTime: __ENV.WARM || '60s' }),
    };

const window = MODE === 'steady' ? 'steady' : 'measure';

export const options = {
  discardResponseBodies: true,
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  scenarios,
  // Thresholds both judge the window and make k6 export these submetrics in the summary.
  thresholds: {
    [`http_req_duration{scenario:${window}}`]: ['p(95)<200'],
    [`http_req_failed{scenario:${window}}`]: ['rate<0.01'],
    [`http_reqs{scenario:${window}}`]: ['count>=0'],
    [`dropped_iterations{scenario:${window}}`]: ['count>=0'],
  },
};

// TARGET=plain (default): the FastAPI recommender behind the gateway.
// TARGET=kserve: the same model as a KServe InferenceService (v1 protocol), routed by Host.
const TARGET = __ENV.TARGET || 'plain';
const KSERVE_HOST = __ENV.KSERVE_HOST || 'recommender-kserve-reel-history.kserve.local';

export default function () {
  const n = 1 + ((__VU * 7 + __ITER) % 20);
  const start = (__VU * 31 + __ITER * 13) % (POOL.length - n);
  const request = { liked: POOL.slice(start, start + n), k: 10 };
  const res = TARGET === 'kserve'
    ? http.post(`${BASE}/v1/models/recommender:predict`, JSON.stringify({ instances: [request] }), {
        headers: { 'Content-Type': 'application/json', Host: KSERVE_HOST },
        timeout: '10s',
      })
    : http.post(`${BASE}/api/rec/recommend`, JSON.stringify(request), {
        headers: { 'Content-Type': 'application/json' },
        timeout: '10s',
      });
  check(res, { 'status 200': (r) => r.status === 200 });
}

export function handleSummary(data) {
  return { '/out/summary.json': JSON.stringify(data) };
}
