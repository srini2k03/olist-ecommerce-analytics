# H&M recommender API (CPU only).
#   docker build -f docker/recsys.Dockerfile -t recsys-api .
#   docker run -p 8000:8000 -v "$PWD/data/hm/artifacts/serving:/artifacts:ro" recsys-api
# The serving bundle (python -m recsys export) is mounted, never baked in: it is
# derived from data whose licence forbids redistribution.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_WAIT_POLICY=PASSIVE \
    PYTHONPATH=/app/src \
    RECSYS_ARTIFACTS=/artifacts

# libgomp: OpenMP runtime for LightGBM and FAISS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-torch-cpu.txt requirements-serve.txt ./
RUN pip install -r requirements-torch-cpu.txt \
    && pip install -r requirements-serve.txt \
    && python -c "import torch; assert not torch.cuda.is_available()"

COPY src/recsys ./src/recsys

RUN useradd --create-home --uid 10001 recsys
USER recsys

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"

CMD ["uvicorn", "recsys.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
