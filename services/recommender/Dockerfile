# syntax=docker/dockerfile:1
# Multi-stage build after astral-sh/uv-docker-example's multistage.Dockerfile: uv installs the
# locked dependencies in the builder, and the runtime image gets only the venv and the code.

FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1 UV_PYTHON_DOWNLOADS=0
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
COPY recommender ./recommender

# the runtime image must match the builder's Python path
FROM python:3.12-slim-trixie
# numeric uid, so Kubernetes runAsNonRoot can check it
RUN groupadd --system --gid 10001 nonroot \
 && useradd --system --gid 10001 --uid 10001 --no-create-home nonroot \
 && mkdir /artifacts && chown 10001:10001 /artifacts
COPY --from=builder --chown=10001:10001 /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
# the init step downloads here; one math thread per request, replicas scale instead
ENV ARTIFACT_DIR=/artifacts HF_HOME=/tmp/hf \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
USER 10001
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "recommender.main:app", "--host", "0.0.0.0", "--port", "8000"]
