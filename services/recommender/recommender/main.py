"""Recommender API: GRU4Rec recommendations, FAISS similar films, title search.

Stateless: every request carries the liked films, so any replica can answer it.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from recommender import fetch
from recommender.model import Recommender

log = logging.getLogger("recommender")



# Edges of the per-handler latency histogram. The library default (0.1, 0.5, 1 s) can't
# resolve a 200 ms p95, which is what the Phase 6 load test is judged on.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2.5, 5, 10, 30)

class RecommendRequest(BaseModel):
    liked: list[int] = Field(default_factory=list, max_length=500, description="movieIds, oldest first")
    k: int = Field(default=10, ge=1, le=50)


def create_app(artifact_dir: Path | None = None) -> FastAPI:
    root = artifact_dir or fetch.ARTIFACT_DIR

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        try:
            model = Recommender.load(root)
            model.warm_up()
            app.state.model = model
            log.info("loaded %d films from %s", len(model.catalog), root)
        except Exception:
            # stay up but unready, so the probe (not a crash loop) says what is wrong
            log.exception("could not load artifacts from %s", root)
        yield

    app = FastAPI(title="reel-history recommender", lifespan=lifespan)
    # one registry per app, so tests can build several apps in one process
    registry = CollectorRegistry()
    Instrumentator(excluded_handlers=["/metrics", "/healthz", "/readyz"], registry=registry).instrument(
        app, latency_lowr_buckets=LATENCY_BUCKETS).expose(app, include_in_schema=False)

    def model() -> Recommender:
        if getattr(app.state, "model", None) is None:
            raise HTTPException(503, "model not loaded")
        return app.state.model

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        if getattr(app.state, "model", None) is None:
            return JSONResponse({"status": "loading"}, status_code=503)
        return {"status": "ready", "films": len(app.state.model.catalog)}

    @app.post("/recommend")
    def recommend(req: RecommendRequest):
        return {"items": model().recommend(req.liked, req.k)}

    @app.get("/similar/{movie_id}")
    def similar(movie_id: int, k: int = Query(10, ge=1, le=50)):
        items = model().similar(movie_id, k)
        if items is None:
            raise HTTPException(404, f"movieId {movie_id} has no LLaRA vector")
        return {"items": items}

    @app.get("/search")
    def search(q: str = Query(..., min_length=1, max_length=100), limit: int = Query(10, ge=1, le=50)):
        return {"items": [f.card() for f in model().catalog.search(q, limit)]}

    return app


app = create_app()
