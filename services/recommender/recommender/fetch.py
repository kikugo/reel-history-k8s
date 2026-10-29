"""Download the recommender's artifacts from Hugging Face into ARTIFACT_DIR.

python -m recommender.fetch

Runs once before the API starts: a one-shot service in docker-compose, an init container in
Kubernetes. The files come from the public Reel History Space at a pinned commit, so every
start gets the same bytes.
"""

import os
from pathlib import Path

HF_REPO = os.environ.get("HF_REPO", "karthikcr/reel-history")
HF_REPO_TYPE = os.environ.get("HF_REPO_TYPE", "space")
HF_REVISION = os.environ.get("HF_REVISION", "c0509d93deee3296f88cca0c0e10c6789bc2683c")
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "/artifacts"))

# local name -> path inside the repo
FILES = {
    "gru4rec.onnx": "backend/assets/ml32m/gru4rec.onnx",
    "gru4rec_item_ids.npy": "backend/assets/ml32m/gru4rec_item_ids.npy",
    "llara_item.faiss": "backend/assets/ml32m/llara_item.faiss",
    "llara_item_ids.npy": "backend/assets/ml32m/llara_item_ids.npy",
    "movies.csv": "data/ml-32m/movies.csv",
    "movie_stats.csv": "data/ml-32m/movie_stats.csv",
}


def artifact_path(name: str, root: Path = ARTIFACT_DIR) -> Path:
    return root / FILES[name]


def missing(root: Path = ARTIFACT_DIR) -> list[str]:
    """Names of artifacts that are absent or empty under root."""
    return [n for n in FILES if not (p := artifact_path(n, root)).is_file() or p.stat().st_size == 0]


def fetch(root: Path = ARTIFACT_DIR) -> Path:
    if not missing(root):
        print(f"artifacts already in {root}")
        return root
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
        revision=HF_REVISION,
        allow_patterns=list(FILES.values()),
        local_dir=root,
    )
    if gone := missing(root):
        raise FileNotFoundError(f"{HF_REPO}@{HF_REVISION} did not provide {gone}")
    print(f"fetched {len(FILES)} artifacts from {HF_REPO}@{HF_REVISION[:8]} into {root}")
    return root


if __name__ == "__main__":
    fetch()  # an exception exits non-zero, so compose / the init container reports the failure
