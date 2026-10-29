import zipfile

import pytest

from bench import build_prompts as bp


def _fixture_zip(path, n_users=60, films_per_user=8):
    movies = ["movieId,title,genres"] + [f"{i},Film {i} ({1990 + i % 30}),Drama|Comedy" for i in range(1, 200)] + ["500,Disliked (2000),Horror"]
    ratings = ["userId,movieId,rating,timestamp"]
    for u in range(1, n_users + 1):
        for k in range(films_per_user):
            ratings.append(f"{u},{(u * 7 + k) % 199 + 1},5.0,0")
        ratings.append(f"{u},500,2.0,0")  # a low rating that must never be used
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ml-latest-small/movies.csv", "\n".join(movies) + "\n")
        zf.writestr("ml-latest-small/ratings.csv", "\n".join(ratings) + "\n")
    return path


def test_builds_fifty_balanced_prompts(tmp_path):
    prompts = bp.build(_fixture_zip(tmp_path / "ml.zip"))
    assert len(prompts) == bp.N_PROMPTS
    assert sum(p["lang"] == "fr" for p in prompts) == bp.N_PROMPTS // 2
    assert len({p["id"] for p in prompts}) == bp.N_PROMPTS


def test_candidate_is_not_among_liked_films(tmp_path):
    for p in bp.build(_fixture_zip(tmp_path / "ml.zip")):
        assert len(p["liked_movie_ids"]) == bp.N_LIKED
        assert p["candidate_movie_id"] not in p["liked_movie_ids"]
        assert 500 not in p["liked_movie_ids"] + [p["candidate_movie_id"]]


def test_is_deterministic(tmp_path):
    z = _fixture_zip(tmp_path / "ml.zip")
    assert bp.build(z) == bp.build(z)


def test_french_prompts_use_the_french_template(tmp_path):
    fr = [p for p in bp.build(_fixture_zip(tmp_path / "ml.zip")) if p["lang"] == "fr"]
    assert all(p["prompt"].startswith("Un utilisateur a adoré") for p in fr)


def test_wrong_zip_is_rejected(tmp_path):
    bad = tmp_path / "ml.zip"
    bad.write_bytes(b"not the real file")
    with pytest.raises(ValueError):
        bp.download(bad)
