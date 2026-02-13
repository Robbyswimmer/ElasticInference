FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ common/
COPY gateway/ gateway/
COPY config.yaml .

# Compile proto stubs
RUN python -m grpc_tools.protoc \
    -I common/proto \
    --python_out=common/proto \
    --grpc_python_out=common/proto \
    common/proto/inference.proto

EXPOSE 50051 9090

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "gateway.server"]
