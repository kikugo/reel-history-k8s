"""The recommender as a KServe custom predictor (v1 protocol).

POST /v1/models/recommender:predict {"instances": [{"liked": [movieId, ...], "k": 10}]}
  -> {"predictions": [[card, ...]]}, one list per instance

The model code is services/recommender's Recommender, unchanged; only the serving layer differs,
which is what the Phase 7 comparison measures.
"""

import os
from pathlib import Path

from kserve import Model
from kserve.errors import InvalidInput

from recommender.fetch import ARTIFACT_DIR
from recommender.model import Recommender

MAX_LIKED = 500
MAX_K = 50


def parse_instance(instance) -> tuple[list[int], int]:
    """Same limits as the FastAPI service's RecommendRequest."""
    if not isinstance(instance, dict):
        raise InvalidInput("each instance must be an object with 'liked' and 'k'")
    liked, k = instance.get("liked", []), instance.get("k", 10)
    if not isinstance(liked, list) or len(liked) > MAX_LIKED or not all(type(m) is int for m in liked):
        raise InvalidInput(f"'liked' must be a list of at most {MAX_LIKED} integer movieIds")
    if type(k) is not int or not 1 <= k <= MAX_K:
        raise InvalidInput(f"'k' must be an integer from 1 to {MAX_K}")
    return liked, k


class RecommenderModel(Model):
    def __init__(self, name: str = "recommender", artifact_dir: Path | None = None):
        super().__init__(name)
        self.artifact_dir = artifact_dir or Path(os.environ.get("ARTIFACT_DIR", ARTIFACT_DIR))
        self.recommender: Recommender | None = None

    def load(self) -> bool:
        model = Recommender.load(self.artifact_dir)
        model.warm_up()  # ready means the model has run once, as in the FastAPI service
        self.recommender = model
        self.ready = True
        return self.ready

    def predict(self, payload: dict, headers: dict | None = None, response_headers: dict | None = None) -> dict:
        instances = payload.get("instances") if isinstance(payload, dict) else None
        if not isinstance(instances, list) or not instances:
            raise InvalidInput("expected {'instances': [...]} with at least one instance")
        parsed = [parse_instance(i) for i in instances]
        return {"predictions": [self.recommender.recommend(liked, k) for liked, k in parsed]}
