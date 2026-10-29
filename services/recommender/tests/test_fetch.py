import pytest

from recommender import fetch


def test_missing_lists_absent_and_empty_files(tmp_path):
    assert set(fetch.missing(tmp_path)) == set(fetch.FILES)
    path = fetch.artifact_path("movies.csv", tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("")
    assert "movies.csv" in fetch.missing(tmp_path)
    path.write_text("movieId,title,genres\n")
    assert "movies.csv" not in fetch.missing(tmp_path)


def test_fetch_skips_download_when_complete(artifact_dir, monkeypatch):
    def boom(**_):
        raise AssertionError("should not download")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    assert fetch.fetch(artifact_dir) == artifact_dir


def test_fetch_asks_only_for_pinned_files_and_checks_them(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: calls.append(kw))
    with pytest.raises(FileNotFoundError):
        fetch.fetch(tmp_path)
    (kw,) = calls
    assert kw["repo_type"] == fetch.HF_REPO_TYPE
    assert kw["revision"] == fetch.HF_REVISION and len(kw["revision"]) == 40
    assert sorted(kw["allow_patterns"]) == sorted(fetch.FILES.values())
