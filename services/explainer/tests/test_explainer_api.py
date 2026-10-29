import json

import httpx
import pytest
from fastapi.testclient import TestClient

from explainer import template
from explainer.cache import TTLCache
from explainer.main import Settings, create_app
from explainer.prompt import Film, build_prompt

LIKED = [
    {"title": "Forrest Gump", "year": 1994, "genres": ["Comedy", "Drama", "Romance"]},
    {"title": "Contact", "year": 1997, "genres": ["Drama", "Sci-Fi"]},
    {"title": "Heat", "year": 1995, "genres": ["Action", "Crime"]},
]
CANDIDATE = {"title": "A.I. Artificial Intelligence", "year": 2001, "genres": ["Adventure", "Drama", "Sci-Fi"]}
BODY = {"liked": LIKED, "candidate": CANDIDATE}
SETTINGS = Settings(base_url="http://llm.test", api_key="k", timeout_s=1, deadline_s=5)


def chat_reply(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


class FakeLLM:
    """httpx handler that plays back a list of responses (or exceptions) and records requests."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_client(fake: FakeLLM | None, settings: Settings = SETTINGS) -> TestClient:
    transport = httpx.MockTransport(fake) if fake else None
    return TestClient(create_app(settings, transport=transport, backoff_s=0))


def test_health_and_ready_without_llm():
    with make_client(None, Settings()) as c:
        assert c.get("/readyz").json() == {"status": "ready"}
        assert c.get("/healthz").json()["llm_configured"] is False


def test_llm_answer_and_request_shape():
    fake = FakeLLM(chat_reply("<think>\n\n</think>\n\nBecause you loved Contact."))
    with make_client(fake) as c:
        resp = c.post("/explain", json=BODY).json()
    assert resp == {"text": "Because you loved Contact.", "source": "llm"}
    (req,) = fake.requests
    assert str(req.url) == "http://llm.test/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer k"
    body = json.loads(req.content)
    assert body["model"] == "qwen3-8b-awq"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0
    assert "- Contact (1997) (Drama, Sci-Fi)" in body["messages"][0]["content"]
    assert "A.I. Artificial Intelligence (2001)" in body["messages"][0]["content"]


def test_retries_5xx_then_succeeds():
    fake = FakeLLM(httpx.Response(503), httpx.Response(502), chat_reply("ok"))
    with make_client(fake) as c:
        assert c.post("/explain", json=BODY).json()["source"] == "llm"
    assert len(fake.requests) == 3


def test_timeouts_fall_back_to_template():
    fake = FakeLLM(*[httpx.ReadTimeout("slow")] * 3)
    with make_client(fake) as c:
        resp = c.post("/explain", json=BODY).json()
        health = c.get("/healthz").json()
    assert resp["source"] == "template" and resp["fallback_reason"] == "timeout"
    assert resp["text"] == template.explain([Film(**f) for f in LIKED], Film(**CANDIDATE))
    assert len(fake.requests) == 3
    assert health["last_llm_call"]["ok"] is False


def test_connect_error_is_retried():
    fake = FakeLLM(httpx.ConnectError("down"), chat_reply("ok"))
    with make_client(fake) as c:
        assert c.post("/explain", json=BODY).json()["source"] == "llm"


def test_4xx_is_not_retried():
    fake = FakeLLM(httpx.Response(401))
    with make_client(fake) as c:
        resp = c.post("/explain", json=BODY).json()
    assert resp["fallback_reason"] == "http_401"
    assert len(fake.requests) == 1


def test_empty_or_malformed_reply_falls_back():
    for reply in (chat_reply("   "), httpx.Response(200, json={"oops": 1})):
        with make_client(FakeLLM(reply)) as c:
            assert c.post("/explain", json=BODY).json()["source"] == "template"


def test_cache_hit_skips_llm_and_template_is_not_cached():
    fake = FakeLLM(*[httpx.Response(500)] * 3, chat_reply("cached"), httpx.Response(400))
    with make_client(fake) as c:
        assert c.post("/explain", json=BODY).json()["source"] == "template"
        assert c.post("/explain", json=BODY).json() == {"text": "cached", "source": "llm"}
        assert c.post("/explain", json=BODY).json() == {"text": "cached", "source": "cache"}
        # another language is another prompt
        assert c.post("/explain", json={**BODY, "lang": "fr"}).json()["source"] == "template"
        metrics = c.get("/metrics").text
    assert len(fake.requests) == 5
    assert "explainer_cache_hits_total 1.0" in metrics
    assert "explainer_cache_misses_total 3.0" in metrics
    assert 'explainer_fallback_total{reason="http_500"} 1.0' in metrics
    assert 'explainer_fallback_total{reason="disabled"}' not in metrics
    assert 'explainer_answers_total{source="cache"} 1.0' in metrics
    assert "explainer_llm_seconds_count 1.0" in metrics


def test_no_llm_configured_uses_template():
    with make_client(None, Settings()) as c:
        resp = c.post("/explain", json=BODY).json()
        metrics = c.get("/metrics").text
    assert resp["source"] == "template" and resp["fallback_reason"] == "disabled"
    assert 'explainer_fallback_total{reason="disabled"} 1.0' in metrics


@pytest.mark.parametrize("body", [
    {"liked": [], "candidate": CANDIDATE},
    {"liked": LIKED * 4, "candidate": CANDIDATE},
    {"liked": LIKED, "candidate": {"title": ""}},
    {"liked": LIKED, "candidate": CANDIDATE, "lang": "de"},
])
def test_validation(body):
    with make_client(None, Settings()) as c:
        assert c.post("/explain", json=body).status_code == 422


def test_template_names_overlapping_films_in_both_languages():
    liked = [Film(**f) for f in LIKED]
    en = template.explain(liked, Film(**CANDIDATE), "en")
    assert en.startswith("A.I. Artificial Intelligence (2001) shares the Drama and Sci-Fi streak of Contact (1997)")
    assert "Forrest Gump (1994)" in en and "Heat" not in en
    fr = template.explain(liked, Film(**CANDIDATE), "fr")
    assert "drame et science-fiction" in fr and "Contact (1997) et Forrest Gump (1994)" in fr


def test_template_without_overlap_does_not_invent_a_reason():
    text = template.explain([Film(**LIKED[2])], Film(title="Amélie", year=2001, genres=["Romance"]))
    assert text == "Amélie (2001) is a step away from Heat (1995), but people with tastes like yours often watch it next."


def test_prompt_matches_phase3_wording():
    prompt = build_prompt([Film(**LIKED[0])], Film(**CANDIDATE), "en")
    assert prompt == (
        "A user loved these films:\n- Forrest Gump (1994) (Comedy, Drama, Romance)\n\n"
        "In two or three sentences, explain why this user would probably enjoy "
        "A.I. Artificial Intelligence (2001). Refer to specific films from their list."
    )


def test_ttl_cache_expires_and_evicts():
    now = [0.0]
    cache = TTLCache(max_items=2, ttl_s=10, clock=lambda: now[0])
    cache.set("a", "1")
    cache.set("b", "2")
    assert cache.get("a") == "1"  # a is now most recent
    cache.set("c", "3")
    assert cache.get("b") is None and len(cache) == 2
    now[0] = 11
    assert cache.get("a") is None
