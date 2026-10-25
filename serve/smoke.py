"""Send one chat request to a served variant, and one without the key.

VARIANT=awq uv run modal run -m serve.smoke

The request runs in a small Modal container that holds the API-key secret, so the key
never reaches this machine. Writes results/serve/smoke_<variant>.json.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

from serve.variants import served_name
from serve.vllm_app import VARIANT, Server, api_key, app

PROMPT = "In one sentence, why might someone who loved Amélie enjoy Cinema Paradiso?"
HEALTH_TIMEOUT_S = 10 * 60
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "serve"

client_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("quant", "serve")


def _request(url: str, body: dict | None = None, key: str | None = None, timeout: float = 120) -> tuple[int, dict]:
    import urllib.error
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as err:
        return err.code, {}
    except (urllib.error.URLError, TimeoutError):
        return 0, {}


@app.function(image=client_image, secrets=[api_key], timeout=15 * 60)
def smoke(base_url: str, variant: str) -> dict:
    import os

    t0 = time.perf_counter()
    status = 0
    while time.perf_counter() - t0 < HEALTH_TIMEOUT_S:
        status, _ = _request(f"{base_url}/health", timeout=60)
        if status == 200:
            break
        time.sleep(2)  # 503 while the container scales up from zero
    ready_s = time.perf_counter() - t0
    if status != 200:
        return {"variant": variant, "ready": False, "health_status": status, "ready_s": round(ready_s, 1)}

    body = {
        "model": served_name(variant),
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 64,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    no_key_status, _ = _request(f"{base_url}/v1/chat/completions", body)
    t = time.perf_counter()
    status, resp = _request(f"{base_url}/v1/chat/completions", body, key=os.environ["VLLM_API_KEY"])
    latency_s = time.perf_counter() - t
    choice = (resp.get("choices") or [{}])[0]
    return {
        "variant": variant,
        "ready": True,
        "ready_s": round(ready_s, 1),
        "no_key_status": no_key_status,
        "status": status,
        "latency_s": round(latency_s, 2),
        "served_model": resp.get("model"),
        "reply": (choice.get("message") or {}).get("content"),
        "usage": resp.get("usage"),
    }


@app.local_entrypoint()
def run_smoke():
    url = Server.get_url()
    print(f"server for {VARIANT}: {url}")
    record = smoke.remote(url, VARIANT)
    record["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    record["passed"] = record.get("status") == 200 and record.get("no_key_status") == 401 and bool(record.get("reply"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"smoke_{VARIANT}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path}: passed={record['passed']} reply={record.get('reply')!r}")
