FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL -o /usr/local/bin/kubectl \
    https://dl.k8s.io/release/v1.30.6/bin/linux/amd64/kubectl && \
    chmod +x /usr/local/bin/kubectl

COPY requirements.txt .
RUN pip install --no-cache-dir grpcio pyyaml prometheus-client

COPY common/ common/
COPY scaling/ scaling/
COPY config.yaml .

# Compile proto stubs
RUN pip install --no-cache-dir grpcio-tools && \
    python -m grpc_tools.protoc \
    -I common/proto \
    --python_out=common/proto \
    --grpc_python_out=common/proto \
    common/proto/inference.proto

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "scaling.controller"]
