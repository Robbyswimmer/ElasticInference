#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NAMESPACE="llm-inference"
REGISTRY="${REGISTRY:-}"  # Set to "myregistry.com/" for remote, empty for local

echo "=== Building Docker images ==="
docker build -t ${REGISTRY}llm-inference/gateway:latest   -f "$PROJECT_DIR/docker/gateway.Dockerfile" "$PROJECT_DIR"
docker build -t ${REGISTRY}llm-inference/worker:latest    -f "$PROJECT_DIR/docker/worker.Dockerfile" "$PROJECT_DIR"
docker build -t ${REGISTRY}llm-inference/controller:latest -f "$PROJECT_DIR/docker/controller.Dockerfile" "$PROJECT_DIR"

if [ -n "$REGISTRY" ]; then
    echo "=== Pushing images ==="
    docker push ${REGISTRY}llm-inference/gateway:latest
    docker push ${REGISTRY}llm-inference/worker:latest
    docker push ${REGISTRY}llm-inference/controller:latest
fi

echo "=== Creating namespace ==="
kubectl apply -f "$PROJECT_DIR/k8s/namespace.yaml"

echo "=== Deploying Redis ==="
kubectl apply -f "$PROJECT_DIR/k8s/redis.yaml"
kubectl -n $NAMESPACE rollout status deployment/redis --timeout=60s

echo "=== Deploying Prefill workers ==="
kubectl apply -f "$PROJECT_DIR/k8s/prefill-deployment.yaml"
kubectl apply -f "$PROJECT_DIR/k8s/prefill-service.yaml"

echo "=== Deploying Decode workers ==="
kubectl apply -f "$PROJECT_DIR/k8s/decode-deployment.yaml"
kubectl apply -f "$PROJECT_DIR/k8s/decode-service.yaml"

echo "=== Deploying Gateway ==="
kubectl apply -f "$PROJECT_DIR/k8s/gateway-deployment.yaml"
kubectl apply -f "$PROJECT_DIR/k8s/gateway-service.yaml"

echo "=== Deploying Scaling Controller ==="
kubectl apply -f "$PROJECT_DIR/k8s/metrics-server.yaml"

echo "=== Waiting for rollouts ==="
kubectl -n $NAMESPACE rollout status deployment/prefill --timeout=300s
kubectl -n $NAMESPACE rollout status deployment/decode --timeout=300s
kubectl -n $NAMESPACE rollout status deployment/gateway --timeout=120s

echo "=== Applying HPAs ==="
kubectl apply -f "$PROJECT_DIR/k8s/prefill-hpa.yaml"
kubectl apply -f "$PROJECT_DIR/k8s/decode-hpa.yaml"

echo ""
echo "=== Deployment complete ==="
echo "Gateway NodePort: $(kubectl -n $NAMESPACE get svc gateway -o jsonpath='{.spec.ports[0].nodePort}')"
kubectl -n $NAMESPACE get pods
