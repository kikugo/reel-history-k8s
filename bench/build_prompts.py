"""Build the 50 "why would this user like this film" prompts from MovieLens ml-latest-small.

uv run python -m bench.build_prompts

Downloads the zip once into data/raw/ (gitignored; `curl -o` it there if Python lacks CA
certificates), checks its sha256, and writes
bench/data/explain_prompts.jsonl. Only film titles and genres go into the output, never
ratings. MovieLens terms and citation are in bench/data/MOVIELENS_NOTICE.txt.
"""

import csv
import hashlib
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_SHA256 = "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"
ZIP_PATH = ROOT / "data" / "raw" / "ml-latest-small.zip"
OUT_PATH = ROOT / "bench" / "data" / "explain_prompts.jsonl"

SEED = 42
N_PROMPTS = 50
N_LIKED = 5
MIN_RATING = 4.5

TEMPLATES = {
    "en": (
        "A user loved these films:\n{liked}\n\n"
        "In two or three sentences, explain why this user would probably enjoy {candidate}. "
        "Refer to specific films from their list."
    ),
    "fr": (
        "Un utilisateur a adoré ces films :\n{liked}\n\n"
        "En deux ou trois phrases, explique pourquoi cet utilisateur aimerait probablement {candidate}. "
        "Appuie-toi sur des films précis de sa liste."
    ),
}


def download(path: Path = ZIP_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(ZIP_URL, timeout=60) as resp:
            path.write_bytes(resp.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ZIP_SHA256:
        raise ValueError(f"{path} sha256 {digest} does not match the pinned {ZIP_SHA256}")
    return path


def _read_csv(zf: zipfile.ZipFile, name: str) -> list[dict]:
    with zf.open(f"ml-latest-small/{name}") as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))


def _film(movie: dict) -> str:
    genres = movie["genres"].replace("|", ", ")
    return f"{movie['title']} ({genres})" if genres != "(no genres listed)" else movie["title"]


def build(zip_path: Path) -> list[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        movies = {m["movieId"]: m for m in _read_csv(zf, "movies.csv")}
        ratings = _read_csv(zf, "ratings.csv")

    loved: dict[int, list[str]] = {}
    for r in ratings:
        if float(r["rating"]) >= MIN_RATING:
            loved.setdefault(int(r["userId"]), []).append(r["movieId"])
    eligible = sorted(u for u, films in loved.items() if len(films) >= N_LIKED + 1)

    rng = random.Random(SEED)
    users = sorted(rng.sample(eligible, N_PROMPTS))
    prompts = []
    for i, user in enumerate(users):
        films = sorted(loved[user], key=int)
        candidate, *rest = rng.sample(films, N_LIKED + 1)
        liked = sorted(rest, key=int)
        lang = "en" if i % 2 == 0 else "fr"
        text = TEMPLATES[lang].format(
            liked="\n".join(f"- {_film(movies[m])}" for m in liked),
            candidate=movies[candidate]["title"],
        )
        prompts.append({
            "id": f"ml-{user}",
            "lang": lang,
            "liked_movie_ids": [int(m) for m in liked],
            "candidate_movie_id": int(candidate),
            "prompt": text,
        })
    return prompts


def main():
    prompts = build(download())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(prompts)} prompts to {OUT_PATH}")


if __name__ == "__main__":
    main()
