"""The KServe predictor serves the same recommendations as the FastAPI recommender.

Reuses the fake artifact set from services/recommender/tests/conftest.py.
"""

import importlib.util
from pathlib import Path

import pytest
from kserve.errors import InvalidInput

from kserve_recommender.model import RecommenderModel
from recommender.model import Recommender

_conftest = Path(__file__).resolve().parents[2] / "recommender" / "tests" / "conftest.py"
_spec = importlib.util.spec_from_file_location("recommender_fixtures", _conftest)
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)
artifact_dir = fixtures.artifact_dir  # the session-scoped fixture, under the same name


@pytest.fixture
def model(artifact_dir):
    m = RecommenderModel(artifact_dir=artifact_dir)
    assert m.ready is False
    m.load()
    return m


def test_ready_only_after_load(model):
    assert model.ready is True


def test_predict_matches_the_fastapi_recommender(model, artifact_dir):
    out = model.predict({"instances": [{"liked": [10, 30], "k": 10}]})
    assert out == {"predictions": [Recommender.load(artifact_dir).recommend([10, 30], 10)]}
    assert [c["movieId"] for c in out["predictions"][0]] == [50, 20, 40]


def test_one_prediction_per_instance(model):
    out = model.predict({"instances": [{"liked": [10], "k": 2}, {"liked": [], "k": 3}]})
    assert [len(p) for p in out["predictions"]] == [2, 3]
    assert out["predictions"][1][0]["source"] == "popularity"


@pytest.mark.parametrize("payload", [
    {}, {"instances": []}, {"instances": ["x"]}, {"instances": [{"liked": "10"}]},
    {"instances": [{"liked": [10], "k": 0}]}, {"instances": [{"liked": [10], "k": 51}]},
    {"instances": [{"liked": list(range(501))}]}, {"instances": [{"liked": [1.5]}]},
])
def test_bad_input_is_rejected(model, payload):
    with pytest.raises(InvalidInput):
        model.predict(payload)
