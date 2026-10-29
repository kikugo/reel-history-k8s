"""A tiny fake artifact set with the same file layout and tensor names as the real one.

Model tokens 1..5 are movieIds 10..50; film 60 is in the catalog but not the model. The fake
GRU4Rec ignores the sequence and returns fixed scores, so the ranking is known in advance.
"""

import faiss
import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper

from recommender.fetch import artifact_path
from recommender.main import create_app

ITEM_IDS = [0, 10, 20, 30, 40, 50]  # token 0 is padding
SCORES = [9.0, 0.1, 0.5, 0.9, 0.3, 0.7]  # padding scores highest, so skipping it is tested
MOVIES = """movieId,title,genres
10,"Matrix, The (1999)",Action|Sci-Fi
20,Amélie (2001),Comedy|Romance
30,Cinema Paradiso (1988),Drama
40,Heat (1995),Action|Crime|Thriller
50,Spirited Away (2001),Animation|Fantasy
60,Untitled Film,(no genres listed)
"""
STATS = """movieId,pop_count,pop_pct,rating_mean
10,900,0.99,4.1
20,500,0.80,4.0
30,300,0.60,4.2
40,700,0.90,3.9
50,600,0.85,4.3
60,1,0.01,
"""
# 2-d vectors: 10 and 40 point the same way, 20 and 50 the other
VECTORS = {10: [1.0, 0.0], 20: [0.0, 1.0], 30: [0.7, 0.7], 40: [0.95, 0.1], 50: [0.1, 0.95]}


def _fake_gru4rec() -> onnx.ModelProto:
    n = len(ITEM_IDS)
    seq = helper.make_tensor_value_info("item_seq", TensorProto.INT64, ["batch", 50])
    seq_len = helper.make_tensor_value_info("item_seq_len", TensorProto.INT64, ["batch"])
    out = helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["batch", n])
    const = helper.make_tensor("fixed", TensorProto.FLOAT, [1, n], SCORES)
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    nodes = [
        # batch-shaped zeros from item_seq, plus the fixed scores: [batch, 1] * 0 + [1, n]
        helper.make_node("Cast", ["item_seq"], ["seq_f"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["seq_f"], ["summed"], keepdims=1),
        helper.make_node("Mul", ["summed", "zero"], ["zeros"]),
        helper.make_node("Add", ["zeros", "fixed"], ["scores"]),
    ]
    graph = helper.make_graph(nodes, "fake_gru4rec", [seq, seq_len], [out], initializer=[const, zero])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    for name in ("gru4rec.onnx", "movies.csv"):
        artifact_path(name, root).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(_fake_gru4rec(), artifact_path("gru4rec.onnx", root))
    np.save(artifact_path("gru4rec_item_ids.npy", root), np.array(ITEM_IDS, dtype=np.int64))
    ids = np.array(list(VECTORS), dtype=np.int64)
    vecs = np.array(list(VECTORS.values()), dtype=np.float32)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(artifact_path("llara_item.faiss", root)))
    np.save(artifact_path("llara_item_ids.npy", root), ids)
    artifact_path("movies.csv", root).write_text(MOVIES)
    artifact_path("movie_stats.csv", root).write_text(STATS)
    return root


@pytest.fixture
def client(artifact_dir):
    with TestClient(create_app(artifact_dir)) as c:
        yield c
