"""Deterministic explanation used when the LLM is down, cold or slow.

It only states what the data shows: the genres the candidate shares with the liked films, and
which liked films overlap most. With no overlap it says so, and points at how the recommender
works (people with similar taste watch it next) instead of inventing a reason.
"""

from explainer.prompt import Film

GENRES_FR = {
    "Action": "action", "Adventure": "aventure", "Animation": "animation", "Children": "jeunesse",
    "Comedy": "comédie", "Crime": "policier", "Documentary": "documentaire", "Drama": "drame",
    "Fantasy": "fantastique", "Film-Noir": "film noir", "Horror": "horreur", "IMAX": "IMAX",
    "Musical": "comédie musicale", "Mystery": "mystère", "Romance": "romance",
    "Sci-Fi": "science-fiction", "Thriller": "thriller", "War": "guerre", "Western": "western",
}


def _join(items: list[str], lang: str) -> str:
    conj = " et " if lang == "fr" else " and "
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + conj + items[-1]


def _genre(g: str, lang: str) -> str:
    return GENRES_FR.get(g, g) if lang == "fr" else g


def explain(liked: list[Film], candidate: Film, lang: str = "en") -> str:
    cand = set(candidate.genres)
    # liked films ranked by genre overlap with the candidate; ties keep the user's order
    ranked = sorted(enumerate(liked), key=lambda p: (-len(cand & set(p[1].genres)), p[0]))
    anchors = [f for _, f in ranked[:2] if cand & set(f.genres)]
    if anchors:
        shared = [g for g in candidate.genres if any(g in a.genres for a in anchors)]
        titles = _join([a.label() for a in anchors], lang)
        genres = _join([_genre(g, lang) for g in shared], lang)
        if lang == "fr":
            return (f"{candidate.label()} partage le registre {genres} de {titles}, "
                    "que vous avez adorés. C'est le point commun le plus net avec votre liste.")
        return (f"{candidate.label()} shares the {genres} streak of {titles}, which you loved. "
                "That's the clearest thread running through your list.")
    titles = _join([f.label() for f in liked[:2]], lang)
    if lang == "fr":
        return (f"{candidate.label()} s'éloigne un peu de {titles}, mais les spectateurs aux goûts "
                "proches des vôtres le regardent souvent ensuite.")
    return (f"{candidate.label()} is a step away from {titles}, but people with tastes like yours "
            "often watch it next.")
