"""The "why this film" prompt, the same wording Phase 3 benchmarked (bench/build_prompts.py)."""

from pydantic import BaseModel, Field

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


class Film(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    year: int | None = Field(default=None, ge=1870, le=2100)
    genres: list[str] = Field(default_factory=list, max_length=10)

    def label(self) -> str:
        return f"{self.title} ({self.year})" if self.year else self.title

    def line(self) -> str:
        genres = ", ".join(self.genres)
        return f"{self.label()} ({genres})" if genres else self.label()


def build_prompt(liked: list[Film], candidate: Film, lang: str) -> str:
    films = "\n".join(f"- {f.line()}" for f in liked)
    return TEMPLATES[lang].format(liked=films, candidate=candidate.label())
