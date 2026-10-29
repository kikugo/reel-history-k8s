"""Explainer API: "why this film" from the Phase 2 vLLM server, cached, with a template fallback.

Ready as soon as it starts. The LLM scales to zero on Modal and takes minutes to wake, so
tying readiness to it would pull every explainer pod out of the Service while it sleeps; the
template answers in the meantime and /healthz reports how the last LLM call went.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI
from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from explainer import template
from explainer.cache import TTLCache, cache_key
from explainer.llm import LLMClient, LLMError
from explainer.prompt import Film, build_prompt


class ExplainRequest(BaseModel):
    liked: list[Film] = Field(min_length=1, max_length=10)
    candidate: Film
    lang: Literal["en", "fr"] = "en"


class Settings(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = "qwen3-8b-awq"  # serve.variants.served_name("awq")
    timeout_s: float = 15.0
    deadline_s: float = 30.0
    retries: int = 2
    cache_size: int = 1024
    cache_ttl_s: float = 24 * 3600

    @classmethod
    def from_env(cls) -> "Settings":
        env = {
            "base_url": "VLLM_BASE_URL", "api_key": "VLLM_API_KEY", "model": "LLM_MODEL",
            "timeout_s": "LLM_TIMEOUT_S", "deadline_s": "LLM_DEADLINE_S", "retries": "LLM_RETRIES",
            "cache_size": "CACHE_SIZE", "cache_ttl_s": "CACHE_TTL_S",
        }
        return cls(**{k: os.environ[v] for k, v in env.items() if os.environ.get(v)})


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None,
               backoff_s: float = 0.5) -> FastAPI:
    settings = settings or Settings.from_env()
    registry = CollectorRegistry()  # one per app, so tests can build several in one process
    cache_hits = Counter("explainer_cache_hits", "Explanations served from the cache", registry=registry)
    cache_misses = Counter("explainer_cache_misses", "Explanations not in the cache", registry=registry)
    fallbacks = Counter("explainer_fallback", "Template answers, by why the LLM wasn't used",
                        ["reason"], registry=registry)
    answers = Counter("explainer_answers", "Explanations returned, by source", ["source"], registry=registry)
    llm_seconds = Histogram("explainer_llm_seconds", "Time spent on LLM calls that succeeded",
                            buckets=(0.5, 1, 2, 4, 8, 15, 30), registry=registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(transport=transport) as client:
            app.state.llm = None
            if settings.base_url:
                app.state.llm = LLMClient(
                    client, settings.base_url, settings.api_key, settings.model, timeout_s=settings.timeout_s,
                    deadline_s=settings.deadline_s, retries=settings.retries, backoff_s=backoff_s)
            yield

    app = FastAPI(title="reel-history explainer", lifespan=lifespan)
    app.state.cache = TTLCache(settings.cache_size, settings.cache_ttl_s)
    app.state.last_llm = {"ok": None, "reason": None, "at": None}
    Instrumentator(excluded_handlers=["/metrics", "/healthz", "/readyz"], registry=registry).instrument(
        app).expose(app, include_in_schema=False)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "llm_configured": bool(settings.base_url), "last_llm_call": app.state.last_llm}

    @app.get("/readyz")
    def readyz():
        return {"status": "ready"}

    @app.post("/explain")
    async def explain(req: ExplainRequest):
        prompt = build_prompt(req.liked, req.candidate, req.lang)
        key = cache_key(settings.model, req.lang, prompt)
        if (text := app.state.cache.get(key)) is not None:
            cache_hits.inc()
            answers.labels("cache").inc()
            return {"text": text, "source": "cache"}
        cache_misses.inc()

        llm: LLMClient | None = app.state.llm
        if llm is None:
            reason = "disabled"
        else:
            t0 = time.perf_counter()
            try:
                text = await llm.complete(prompt)
            except LLMError as err:
                reason = err.reason
                app.state.last_llm = {"ok": False, "reason": reason, "at": time.time()}
            else:
                llm_seconds.observe(time.perf_counter() - t0)
                app.state.last_llm = {"ok": True, "reason": None, "at": time.time()}
                app.state.cache.set(key, text)  # only LLM answers are cached; templates are free
                answers.labels("llm").inc()
                return {"text": text, "source": "llm"}

        fallbacks.labels(reason).inc()
        answers.labels("template").inc()
        return {"text": template.explain(req.liked, req.candidate, req.lang), "source": "template",
                "fallback_reason": reason}

    return app


app = create_app()
