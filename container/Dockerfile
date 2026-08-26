# SPDX-License-Identifier: Apache-2.0

FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

LABEL org.opencontainers.image.source="https://github.com/mi-mogu/ossp-2026-llm-router-challenge" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="OSSP 2026 Prompt Router"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    TMPDIR=/tmp \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    LOKY_MAX_CPU_COUNT=2

WORKDIR /opt/router

COPY requirements-router.txt /opt/router/requirements.txt
RUN python3 -m pip install --no-cache-dir --only-binary=:all: --no-deps \
        --requirement /opt/router/requirements.txt \
    && python3 -m pip uninstall --yes pip setuptools wheel

COPY --chown=65532:65532 main.py /opt/router/main.py
COPY --chown=65532:65532 coordinator/chunk_processor.py /opt/router/coordinator/chunk_processor.py
COPY --chown=65532:65532 ["n-gram/ngram_classifier_v2_float32.joblib", "n-gram/ngram_features_v2.py", "n-gram/ngram_predict_optimized.py", "n-gram/predict_ngram_v2.py", "/opt/router/n-gram/"]
COPY --chown=65532:65532 ["라우터 분류/router.py", "라우터 분류/router_core.py", "/opt/router/라우터 분류/"]
COPY --chown=65532:65532 ["라우터 분류/artifacts/router_model.joblib", "/opt/router/라우터 분류/artifacts/router_model.joblib"]
COPY container/router-run /usr/local/bin/router-run
RUN chmod 0755 /usr/local/bin/router-run

USER 65532:65532
ENTRYPOINT ["router-run"]
