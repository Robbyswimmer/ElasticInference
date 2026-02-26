#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NAMESPACE="llm-inference"
REGISTRY="${REGISTRY:-}"  # Set to "myregistry.com/" for remote, empty for local
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d%H%M%S)}"
ENABLE_PROMETHEUS="${ENABLE_PROMETHEUS:-1}"
ENABLE_HPA="${ENABLE_HPA:-0}"

if [ -n "$REGISTRY" ] && [[ "$REGISTRY" != */ ]]; then
    REGISTRY="${REGISTRY}/"
fi

GATEWAY_IMAGE="${REGISTRY}llm-inference/gateway:${IMAGE_TAG}"
WORKER_IMAGE="${REGISTRY}llm-inference/worker:${IMAGE_TAG}"
CONTROLLER_IMAGE="${REGISTRY}llm-inference/controller:${IMAGE_TAG}"

NODE_COUNT="$(kubectl get nodes --no-headers | wc -l | tr -d ' ')"
if [ "$NODE_COUNT" -gt 1 ] && [ -z "$REGISTRY" ]; then
    echo "ERROR: Detected multi-node cluster ($NODE_COUNT nodes) but REGISTRY is empty."
    echo "Set REGISTRY (for example: REGISTRY=ghcr.io/<user>/) so all nodes can pull images."
    exit 1
fi

echo "=== Building Docker images (tag: $IMAGE_TAG) ==="
docker build -t "$GATEWAY_IMAGE" -f "$PROJECT_DIR/docker/gateway.Dockerfile" "$PROJECT_DIR"
docker build -t "$WORKER_IMAGE" -f "$PROJECT_DIR/docker/worker.Dockerfile" "$PROJECT_DIR"
docker build -t "$CONTROLLER_IMAGE" -f "$PROJECT_DIR/docker/controller.Dockerfile" "$PROJECT_DIR"

if [ -n "$REGISTRY" ]; then
    echo "=== Pushing images ==="
    docker push "$GATEWAY_IMAGE"
    docker push "$WORKER_IMAGE"
    docker push "$CONTROLLER_IMAGE"
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

if [ "$ENABLE_PROMETHEUS" = "1" ]; then
    echo "=== Deploying Prometheus ==="
    kubectl apply -f "$PROJECT_DIR/k8s/prometheus.yaml"
fi

echo "=== Updating deployment images ==="
kubectl -n "$NAMESPACE" set image deployment/gateway gateway="$GATEWAY_IMAGE"
kubectl -n "$NAMESPACE" set image deployment/prefill prefill="$WORKER_IMAGE"
kubectl -n "$NAMESPACE" set image deployment/decode decode="$WORKER_IMAGE"
kubectl -n "$NAMESPACE" set image deployment/scaling-controller controller="$CONTROLLER_IMAGE"

if [ -n "$REGISTRY" ]; then
    echo "=== Using remote registry images (imagePullPolicy=Always) ==="
    kubectl -n "$NAMESPACE" patch deployment gateway --type='merge' -p '{"spec":{"template":{"spec":{"containers":[{"name":"gateway","imagePullPolicy":"Always"}]}}}}'
    kubectl -n "$NAMESPACE" patch deployment prefill --type='merge' -p '{"spec":{"template":{"spec":{"containers":[{"name":"prefill","imagePullPolicy":"Always"}]}}}}'
    kubectl -n "$NAMESPACE" patch deployment decode --type='merge' -p '{"spec":{"template":{"spec":{"containers":[{"name":"decode","imagePullPolicy":"Always"}]}}}}'
    kubectl -n "$NAMESPACE" patch deployment scaling-controller --type='merge' -p '{"spec":{"template":{"spec":{"containers":[{"name":"controller","imagePullPolicy":"Always"}]}}}}'
fi

echo "=== Waiting for rollouts ==="
kubectl -n "$NAMESPACE" rollout status deployment/prefill --timeout=300s
kubectl -n "$NAMESPACE" rollout status deployment/decode --timeout=300s
kubectl -n "$NAMESPACE" rollout status deployment/gateway --timeout=120s
kubectl -n "$NAMESPACE" rollout status deployment/scaling-controller --timeout=120s

if [ "$ENABLE_HPA" = "1" ]; then
    echo "=== Applying HPAs (requires external metrics adapter) ==="
    kubectl apply -f "$PROJECT_DIR/k8s/prefill-hpa.yaml"
    kubectl apply -f "$PROJECT_DIR/k8s/decode-hpa.yaml"
else
    echo "=== Skipping HPAs (set ENABLE_HPA=1 to enable) ==="
fi

echo ""
echo "=== Deployment complete ==="
echo "Gateway NodePort: $(kubectl -n "$NAMESPACE" get svc gateway -o jsonpath='{.spec.ports[0].nodePort}')"
kubectl -n "$NAMESPACE" get pods
