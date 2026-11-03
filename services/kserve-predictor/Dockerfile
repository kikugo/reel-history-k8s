# syntax=docker/dockerfile:1
# Build context: services/ (so the image reuses the recommender package unchanged).
#   docker build -f services/kserve-predictor/Dockerfile services
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1 UV_PYTHON_DOWNLOADS=0
# the KServe SDK pins psutil < 6, which has no linux/arm64 wheel for 3.12: compile it here only
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=kserve-predictor/uv.lock,target=uv.lock \
    --mount=type=bind,source=kserve-predictor/pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
COPY recommender/recommender ./recommender
COPY kserve-predictor/kserve_recommender ./kserve_recommender

FROM python:3.12-slim-trixie
RUN groupadd --system --gid 10001 nonroot \
 && useradd --system --gid 10001 --uid 10001 --no-create-home nonroot \
 && mkdir /artifacts && chown 10001:10001 /artifacts
COPY --from=builder --chown=10001:10001 /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    ARTIFACT_DIR=/artifacts HF_HOME=/tmp/hf \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
USER 10001
WORKDIR /app
EXPOSE 8080
CMD ["python", "-m", "kserve_recommender"]
