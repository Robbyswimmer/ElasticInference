FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY common/ common/
COPY workers/ workers/
COPY config.yaml .

# Compile proto stubs
RUN python3 -m grpc_tools.protoc \
    -I common/proto \
    --python_out=common/proto \
    --grpc_python_out=common/proto \
    common/proto/inference.proto

# Download model at build time to bake into image
ARG MODEL_NAME=facebook/opt-1.3b
RUN python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('${MODEL_NAME}'); \
    AutoModelForCausalLM.from_pretrained('${MODEL_NAME}')"

ENV PYTHONUNBUFFERED=1
# Override CMD at deploy time: python3 -m workers.prefill OR python3 -m workers.decode
CMD ["python3", "-m", "workers.prefill"]
