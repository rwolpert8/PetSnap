FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/opt/torch-cache \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000 \
    PETSNAP_MODEL_PATH=/app/model-cache/best_model.pth \
    PETSNAP_CLASSES_PATH=/app/models/classes.json \
    AWS_EC2_METADATA_DISABLED=true \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
COPY requirements-inference.txt ./
COPY deploy/requirements.txt ./deploy/requirements.txt
# CPU wheels avoid shipping the CUDA runtime to a CPU-only host.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 torchvision==0.23.0 \
    && python -m pip install -r deploy/requirements.txt \
    && python -m pip check

RUN groupadd --gid 10001 petsnap \
    && useradd --uid 10001 --gid petsnap --no-create-home petsnap \
    && mkdir /app/model-cache \
    && chown petsnap:petsnap /app/model-cache

COPY api/ ./api/
RUN python -m api.dog_detector && chmod -R a+rX /opt/torch-cache
COPY frontend/ ./frontend/
COPY models/classes.json ./models/classes.json
COPY deploy/download_model.py deploy/start.py deploy/model-manifest.json ./deploy/

USER petsnap
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '10000') + '/health', timeout=4)"

# No credentials or model downloads at build time. Secrets are runtime-only.
CMD ["python", "deploy/start.py"]
