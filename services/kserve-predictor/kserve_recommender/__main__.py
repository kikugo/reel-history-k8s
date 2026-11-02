"""python -m kserve_recommender: serve the recommender with KServe's ModelServer on :8080."""

from kserve import ModelServer

from kserve_recommender.model import RecommenderModel

if __name__ == "__main__":
    model = RecommenderModel()
    model.load()
    ModelServer().start([model])
