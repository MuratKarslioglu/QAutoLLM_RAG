# Build from the repo root (this Dockerfile COPYs requirements.txt,
# pyproject.toml, and the source tree, all of which live there):
#   docker build -f docker/Dockerfile -t qautollm .
#   docker run --rm qautollm
#
# Brief P8 gate: the default CMD is the one command that reproduces the
# toy-set main table — no GPU, no model downloads (Brief D1), since the
# toy dataset path uses fake generator/reranker/faithfulness components
# specifically so it never needs either. For real (non-toy) GPU-backed
# experiments, run on bare metal per the README instead — this image
# does not install a CUDA-enabled torch build or request GPU passthrough.
FROM python:3.11-slim

WORKDIR /app

# faiss-cpu/scipy/etc. mostly ship manylinux wheels, but keep build tools
# around in case a source build is ever needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "scripts.run_ablation"]
