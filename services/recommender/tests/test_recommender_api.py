from fastapi.testclient import TestClient

from recommender.catalog import split_title
from recommender.main import create_app


def ids(resp) -> list[int]:
    return [item["movieId"] for item in resp.json()["items"]]


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["films"] == 6


def test_not_ready_without_artifacts(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/readyz").status_code == 503
        assert c.post("/recommend", json={"liked": [10]}).status_code == 503


def test_recommend_ranks_by_model_and_skips_padding_and_seen(client):
    resp = client.post("/recommend", json={"liked": [10, 30], "k": 10})
    assert resp.status_code == 200
    # scores 50:0.7 > 20:0.5 > 40:0.3; 10 and 30 were liked; 60 is outside the model
    assert ids(resp) == [50, 20, 40]
    assert {item["source"] for item in resp.json()["items"]} == {"gru4rec"}


def test_recommend_honours_k(client):
    assert ids(client.post("/recommend", json={"liked": [10], "k": 2})) == [30, 50]


def test_recommend_falls_back_to_popularity(client):
    for liked in ([], [60, 999]):
        resp = client.post("/recommend", json={"liked": liked, "k": 3})
        items = resp.json()["items"]
        assert [i["source"] for i in items] == ["popularity"] * 3
        assert all(i["movieId"] not in liked for i in items)
    assert ids(client.post("/recommend", json={"liked": [], "k": 3})) == [10, 40, 50]


def test_recommend_validates(client):
    assert client.post("/recommend", json={"liked": [10], "k": 0}).status_code == 422
    assert client.post("/recommend", json={"liked": "10"}).status_code == 422


def test_similar_uses_faiss_neighbours(client):
    resp = client.get("/similar/10", params={"k": 2})
    assert ids(resp) == [40, 30]
    sims = [i["similarity"] for i in resp.json()["items"]]
    assert sims == sorted(sims, reverse=True)


def test_similar_unknown_film(client):
    assert client.get("/similar/60").status_code == 404


def test_search_prefix_first_and_accent_folded(client):
    assert ids(client.get("/search", params={"q": "ame"})) == [20]
    assert ids(client.get("/search", params={"q": "the matrix"})) == [10]
    resp = client.get("/search", params={"q": "a"})
    first = resp.json()["items"][0]
    assert first["title"] == "Amélie" and first["year"] == 2001 and first["genres"] == ["Comedy", "Romance"]


def test_metrics_exposed(client):
    client.post("/recommend", json={"liked": [10]})
    body = client.get("/metrics").text
    assert "http_requests_total" in body
    assert 'handler="/recommend"' in body
    assert 'handler="/healthz"' not in body


def test_split_title():
    assert split_title("Matrix, The (1999)") == ("The Matrix", 1999)
    assert split_title("Amour (2012)") == ("Amour", 2012)
    assert split_title("Untitled Film") == ("Untitled Film", None)


def test_latency_histogram_resolves_200ms(client):
    client.post("/recommend", json={"liked": [10]})
    body = client.get("/metrics").text
    assert 'http_request_duration_seconds_bucket{handler="/recommend",le="0.2",method="POST"}' in body
