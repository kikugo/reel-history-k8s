"""Pairwise LLM judge: each variant's explanations against BF16, in both orders.

Judge: Groq openai/gpt-oss-120b, temperature 0. The key is in the Modal secret `groq`, and the
calls run in a Modal container, so it never reaches this machine. A variant wins a prompt only
if it wins in both orders and loses only if it loses in both; anything else is a tie.

uv run modal run -m bench.judge --variant awq --smoke
uv run modal run -m bench.judge --variant awq
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import modal

from bench.common import RESULTS_DIR, app, load_prompts, write_result

JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_URL = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_PROMPT_PATH = Path(__file__).parent / "judge_prompt.txt"
PARALLEL = 4

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
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as err:
            if err.code != 429 and err.code < 500:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("judge kept failing")


@app.function(image=judge_image, secrets=[modal.Secret.from_name("groq")], timeout=30 * 60)
def judge_pairs(pairs: list[dict]) -> list[dict]:
    def one(pair):
        first = parse_verdict(_ask(pair["candidate_first"]))
        second = parse_verdict(_ask(pair["reference_first"]))
        return {"id": pair["id"], "lang": pair["lang"], "first": first, "second": second,
                "outcome": combine(first, second)}

    with ThreadPoolExecutor(PARALLEL) as pool:
        return list(pool.map(one, pairs))


def _answers(variant: str, smoke: bool) -> dict:
    name = f"explain_{variant}_smoke.json" if smoke else f"explain_{variant}.json"
    return {i["id"]: i["answer"] for i in json.loads((RESULTS_DIR / name).read_text())["items"]}


@app.local_entrypoint()
def run_judge(variant: str, smoke: bool = False):
    if variant == "bf16":
        raise ValueError("bf16 is the reference; judge awq, gptq or gguf")
    reference, candidate = _answers("bf16", smoke), _answers(variant, smoke)
    prompts = [p for p in load_prompts() if p["id"] in candidate]
    pairs = [{
        "id": p["id"], "lang": p["lang"],
        "candidate_first": judge_message(p["prompt"], candidate[p["id"]], reference[p["id"]]),
        "reference_first": judge_message(p["prompt"], reference[p["id"]], candidate[p["id"]]),
    } for p in prompts]
    started = datetime.now(UTC).isoformat(timespec="seconds")
    verdicts = judge_pairs.remote(pairs)
    record = {
        "variant": variant, "reference": "bf16", "smoke": smoke, "started_utc": started,
        "judge": {"provider": "groq", "model": JUDGE_MODEL, "temperature": 0, "orders": 2},
        "n": len(verdicts),
        "overall": tally([v["outcome"] for v in verdicts]),
        "by_lang": {lang: tally([v["outcome"] for v in verdicts if v["lang"] == lang]) for lang in ("en", "fr")},
        "verdicts": verdicts,
    }
    write_result(f"judge_{variant}", record, smoke)
