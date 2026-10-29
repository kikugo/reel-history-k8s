"""MovieLens-32M catalog: titles, years, genres and a popularity percentile per film."""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")
# MovieLens stores "Matrix, The"; show "The Matrix"
_ARTICLE_RE = re.compile(r"^(.*), (The|A|An|Les|La|Le|L'|Il|Das|Der|Die|El)$")


def split_title(raw: str) -> tuple[str, int | None]:
    title, year = raw.strip(), None
    if m := _YEAR_RE.search(title):
        title, year = title[: m.start()].strip(), int(m.group(1))
    if m := _ARTICLE_RE.match(title):
        sep = "" if m.group(2).endswith("'") else " "
        title = f"{m.group(2)}{sep}{m.group(1)}"
    return title, year


def fold(text: str) -> str:
    """Lowercase without accents, for title search."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).lower()


@dataclass(frozen=True)
class Film:
    movie_id: int
    title: str
    year: int | None
    genres: tuple[str, ...]
    pop_pct: float

    def card(self, **extra) -> dict:
        return {"movieId": self.movie_id, "title": self.title, "year": self.year,
                "genres": list(self.genres), "pop_pct": round(self.pop_pct, 4), **extra}


class Catalog:
    def __init__(self, films: list[Film]):
        self.by_id = {f.movie_id: f for f in films}
        # most popular first, so popularity fallback and search ranking are a scan
        self.by_popularity = sorted(films, key=lambda f: -f.pop_pct)
        self._folded = [(fold(f.title), f) for f in self.by_popularity]

    @classmethod
    def load(cls, movies_csv: Path, stats_csv: Path) -> "Catalog":
        movies = pd.read_csv(movies_csv)
        stats = pd.read_csv(stats_csv, usecols=["movieId", "pop_pct"])
        df = movies.merge(stats, on="movieId", how="left").fillna({"pop_pct": 0.0})
        films = []
        for row in df.itertuples(index=False):
            title, year = split_title(row.title)
            genres = () if row.genres == "(no genres listed)" else tuple(row.genres.split("|"))
            films.append(Film(int(row.movieId), title, year, genres, float(row.pop_pct)))
        return cls(films)

    def __len__(self) -> int:
        return len(self.by_id)

    def get(self, movie_id: int) -> Film | None:
        return self.by_id.get(movie_id)

    def popular(self, k: int, exclude: set[int]) -> list[Film]:
        out = []
        for f in self.by_popularity:
            if f.movie_id not in exclude:
                out.append(f)
                if len(out) == k:
                    break
        return out

    def search(self, query: str, limit: int) -> list[Film]:
        q = fold(query.strip())
        if not q:
            return []
        prefix = [f for t, f in self._folded if t.startswith(q)]
        inner = [f for t, f in self._folded if q in t and not t.startswith(q)]
        return (prefix + inner)[:limit]
