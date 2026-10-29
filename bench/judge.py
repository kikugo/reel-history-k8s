"""Pairwise LLM judge: each variant's explanations against BF16, in both orders.

Judge: Groq openai/gpt-oss-120b, temperature 0. The key is in the Modal secret `groq`, and the
calls run in a Modal container, so it never reaches this machine. A variant wins a prompt only
if it wins in both orders and loses only if it loses in both; anything else is a tie.

uv run modal run -m bench.judge::run_judge --variant awq --smoke
uv run modal run --detach -m bench.judge::run_judge --variant gptq
uv run modal run -m bench.judge::fetch_judge --variant gptq   # if the local client disconnected

Each verdict is cached on the c2-models Volume as soon as it arrives, and the finished record
is written there too. A run cut off by a sleeping laptop keeps going with --detach, and a
rerun only asks Groq about pairs it hasn't judged yet.
"""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import modal

from bench.common import RESULTS_DIR, VOLUMES, app, load_prompts, models, write_result
from quant import config

JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_URL = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_PROMPT_PATH = Path(__file__).parent / "judge_prompt.txt"
PARALLEL = 1  # Groq rate limits are per minute; one call at a time with backoff
MAX_ATTEMPTS = 12
JUDGE_DIR = f"{config.VOL_MOUNT}/judge"
POLL_S = 30

judge_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("quant", "serve", "bench")


def judge_message(request: str, answer_a: str, answer_b: str) -> str:
    # Rendered locally: Modal ships only the .py files of this package to the container.
    template = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(request=request, answer_a=answer_a, answer_b=answer_b)


def parse_verdict(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    winner = json.loads(text[start:end + 1]).get("winner", "")
    if winner not in ("A", "B", "tie"):
        raise ValueError(f"bad verdict {winner!r}")
    return winner


def combine(first: str, second: str) -> str:
    """first: candidate is A; second: candidate is B. Returns win, loss or tie for the candidate."""
    if first == "A" and second == "B":
        return "win"
    if first == "B" and second == "A":
        return "loss"
    return "tie"


def tally(outcomes: list[str]) -> dict:
    return {k: outcomes.count(k) for k in ("win", "tie", "loss")}


def pair_key(pair: dict) -> str:
    """Cache key: changes whenever either judge message changes, so stale verdicts are never reused."""
    digest = hashlib.sha256((pair["candidate_first"] + "\0" + pair["reference_first"]).encode())
    return f"{pair['id']}-{digest.hexdigest()[:16]}"


def build_record(variant: str, smoke: bool, started: str, verdicts: list[dict]) -> dict:
    return {
        "variant": variant, "reference": "bf16", "smoke": smoke, "started_utc": started,
        "judge": {"provider": "groq", "model": JUDGE_MODEL, "temperature": 0, "orders": 2},
        "n": len(verdicts),
        "overall": tally([v["outcome"] for v in verdicts]),
        "by_lang": {lang: tally([v["outcome"] for v in verdicts if v["lang"] == lang]) for lang in ("en", "fr")},
        "verdicts": verdicts,
    }


def _run_name(variant: str, smoke: bool) -> str:
    return f"judge_{variant}_smoke" if smoke else f"judge_{variant}"


def _ask(message: str) -> str:
    import os
    import urllib.error
    import urllib.request

    body = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        JUDGE_URL, data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "User-Agent": "reel-history-k8s-bench/0.1",  # Cloudflare rejects urllib's default agent
        },
    )
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as err:
            last = f"HTTP {err.code}: {err.read()[:300]!r}"
            if err.code != 429 and err.code < 500:
                # HTTPError holds a socket and can't cross the Modal boundary; keep status and body
                raise RuntimeError(f"judge {last}") from None
            time.sleep(retry_delay(err.headers.get("retry-after"), attempt))
    raise RuntimeError(f"judge kept failing after {MAX_ATTEMPTS} attempts, last {last}")


def retry_delay(retry_after: str | None, attempt: int) -> float:
    """Seconds to wait: Groq's Retry-After when given, else exponential backoff, capped at 60."""
    try:
        return min(float(retry_after), 60.0) + 0.5
    except (TypeError, ValueError):
        return float(min(2 ** attempt, 60))


@app.function(image=judge_image, secrets=[modal.Secret.from_name("groq")], volumes=VOLUMES, timeout=3 * 60 * 60)
def judge_variant(variant: str, smoke: bool, started: str, pairs: list[dict]) -> dict:
    cache = Path(JUDGE_DIR, _run_name(variant, smoke))
    cache.mkdir(parents=True, exist_ok=True)

    def one(pair):
        path = cache / f"{pair_key(pair)}.json"
        if path.exists():
            return json.loads(path.read_text())
        first = parse_verdict(_ask(pair["candidate_first"]))
        second = parse_verdict(_ask(pair["reference_first"]))
        verdict = {"id": pair["id"], "lang": pair["lang"], "first": first, "second": second,
                   "outcome": combine(first, second)}
        path.write_text(json.dumps(verdict))
        models.commit()
        return verdict

    with ThreadPoolExecutor(PARALLEL) as pool:
        verdicts = list(pool.map(one, pairs))
    record = build_record(variant, smoke, started, verdicts)
    Path(JUDGE_DIR, f"{_run_name(variant, smoke)}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    models.commit()
    return record


@app.function(image=judge_image, volumes=VOLUMES, timeout=5 * 60)
def read_record(variant: str, smoke: bool) -> dict | None:
    path = Path(JUDGE_DIR, f"{_run_name(variant, smoke)}.json")
    return json.loads(path.read_text()) if path.exists() else None


def _answers(variant: str, smoke: bool) -> dict:
    name = f"explain_{variant}_smoke.json" if smoke else f"explain_{variant}.json"
    return {i["id"]: i["answer"] for i in json.loads((RESULTS_DIR / name).read_text())["items"]}


def build_pairs(variant: str, smoke: bool) -> list[dict]:
    reference, candidate = _answers("bf16", smoke), _answers(variant, smoke)
    prompts = [p for p in load_prompts() if p["id"] in candidate]
    return [{
        "id": p["id"], "lang": p["lang"],
        "candidate_first": judge_message(p["prompt"], candidate[p["id"]], reference[p["id"]]),
        "reference_first": judge_message(p["prompt"], reference[p["id"]], candidate[p["id"]]),
    } for p in prompts]


@app.local_entrypoint()
def run_judge(variant: str, smoke: bool = False):
    if variant == "bf16":
        raise ValueError("bf16 is the reference; judge awq, gptq or gguf")
    started = datetime.now(UTC).isoformat(timespec="seconds")
    # spawn + short polls: with --detach the spawned call keeps running on Modal even if this
    # client drops, and no single long-held stream is open for a flaky connection to cut.
    call = judge_variant.spawn(variant, smoke, started, build_pairs(variant, smoke))
    print(f"spawned {call.object_id}; if this client drops, use fetch_judge once it finishes")
    while True:
        try:
            record = call.get(timeout=POLL_S)
            break
        except (TimeoutError, modal.exception.TimeoutError):
            continue
    write_result(f"judge_{variant}", record, smoke)


@app.local_entrypoint()
def fetch_judge(variant: str, smoke: bool = False):
    record = read_record.remote(variant, smoke)
    if record is None:
        raise SystemExit(f"no finished {_run_name(variant, smoke)} record on the Volume yet")
    write_result(f"judge_{variant}", record, smoke)
