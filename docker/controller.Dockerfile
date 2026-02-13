FROM python:3.10-slim

WORKDIR /app

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
