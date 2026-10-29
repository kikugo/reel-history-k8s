"""GRU4Rec (ONNX) for recommendations and the LLaRA FAISS index for similar films.

The ONNX graph takes the user's last 50 films as item tokens, in watch order, and scores every
token. Token 0 is padding; gru4rec_item_ids.npy maps token -> movieId. The FAISS index is an
inner-product index over L2-normalized LLaRA vectors, row i is llara_item_ids[i].
"""

from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort

from recommender.catalog import Catalog, Film
from recommender.fetch import artifact_path

MAX_SEQ_LEN = 50  # GRU4Rec's training MAX_ITEM_LIST_LENGTH


class Recommender:
    def __init__(self, catalog: Catalog, session: ort.InferenceSession, item_ids: np.ndarray,
                 index: faiss.Index, llara_ids: np.ndarray):
        self.catalog = catalog
        self.session = session
        self.item_ids = item_ids
        self.token_of = {int(m): t for t, m in enumerate(item_ids) if t != 0}
        self.index = index
        self.llara_ids = llara_ids
        self.llara_row = {int(m): i for i, m in enumerate(llara_ids)}

    @classmethod
    def load(cls, root: Path) -> "Recommender":
        catalog = Catalog.load(artifact_path("movies.csv", root), artifact_path("movie_stats.csv", root))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # one request, one thread; replicas scale instead
        session = ort.InferenceSession(str(artifact_path("gru4rec.onnx", root)), sess_options=opts,
                                       providers=["CPUExecutionProvider"])
        item_ids = np.load(artifact_path("gru4rec_item_ids.npy", root))
        index = faiss.read_index(str(artifact_path("llara_item.faiss", root)))
        llara_ids = np.load(artifact_path("llara_item_ids.npy", root))
        if index.ntotal != len(llara_ids):
            raise ValueError(f"FAISS index has {index.ntotal} rows but {len(llara_ids)} ids")
        return cls(catalog, session, item_ids, index, llara_ids)

    def warm_up(self) -> None:
        """One real inference and one search, so readiness means the model actually runs."""
        self._scores([next(iter(self.token_of.values()))])
        self.index.search(self.index.reconstruct(0).reshape(1, -1), 2)

    def _scores(self, tokens: list[int]) -> np.ndarray:
        tokens = tokens[-MAX_SEQ_LEN:]
        seq = np.zeros((1, MAX_SEQ_LEN), dtype=np.int64)
        seq[0, : len(tokens)] = tokens
        lens = np.array([len(tokens)], dtype=np.int64)
        return self.session.run(["scores"], {"item_seq": seq, "item_seq_len": lens})[0][0]

    def recommend(self, liked: list[int], k: int) -> list[dict]:
        seen = set(liked)
        tokens = [self.token_of[m] for m in liked if m in self.token_of]
        if not tokens:
            return [f.card(source="popularity") for f in self.catalog.popular(k, seen)]
        scores = self._scores(tokens)
        out: list[dict] = []
        for tok in np.argsort(-scores):
            if tok == 0:
                continue
            mid = int(self.item_ids[tok])
            film = self.catalog.get(mid)
            if mid in seen or film is None:
                continue
            out.append(film.card(source="gru4rec"))
            if len(out) == k:
                break
        return out

    def similar(self, movie_id: int, k: int) -> list[dict] | None:
        row = self.llara_row.get(movie_id)
        if row is None:
            return None
        sims, rows = self.index.search(self.index.reconstruct(row).reshape(1, -1), k + 1)
        out = []
        for sim, r in zip(sims[0], rows[0], strict=True):
            if r < 0 or r == row:
                continue
            film: Film | None = self.catalog.get(int(self.llara_ids[r]))
            if film is not None:
                out.append(film.card(similarity=round(float(sim), 4)))
        return out[:k]
