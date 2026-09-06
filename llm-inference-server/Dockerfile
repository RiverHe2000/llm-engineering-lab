# CPU inference image. Multi-stage keeps build tools out of the runtime layer.
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    LLMSERVE_DEVICE=cpu \
    LLMSERVE_MODEL_NAME=distilbert/distilgpt2 \
    OMP_NUM_THREADS=4
COPY --from=builder /opt/venv /opt/venv
RUN useradd --create-home --uid 10001 app && mkdir -p /models && chown app /models
USER app
WORKDIR /home/app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3).status == 200 else 1)"
CMD ["llmserve", "serve", "--host", "0.0.0.0", "--port", "8000"]
