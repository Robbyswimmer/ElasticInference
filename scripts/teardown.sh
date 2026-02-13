#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="llm-inference"

echo "=== Tearing down llm-inference namespace ==="
kubectl delete namespace $NAMESPACE --ignore-not-found

echo "=== Cleanup complete ==="
