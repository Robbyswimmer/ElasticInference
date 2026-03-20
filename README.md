# Elastic & Auto-Tuned LLM Inference on Kubernetes

A prototype LLM inference system that decomposes inference into independently scalable stages (prefill, decode, gateway, MoE experts) with elastic scaling and serving-time autotuning on Kubernetes.

## Architecture

```
Client → Gateway/Router (CPU) → Prefill Workers (GPU) → Redis KV Cache → Decode Workers (GPU)
                                                                          ↳ MoE Expert Routing
```

- **Gateway** — gRPC request routing, queue management, tokenization
- **Prefill Workers** — prompt prefill, KV cache generation (throughput-oriented)
- **Decode Workers** — autoregressive decoding with MoE simulation (latency-sensitive)
- **Scaling Controller** — custom elastic scaling with EMA smoothing, anti-oscillation, fast scale-up/slow scale-down
- **Autotune** — knee profiling + online configuration selection across GPU slices

## Project Goals

| Goal | Description |
|------|-------------|
| **G1** | Runnable LLM inference prototype on Kubernetes |
| **G2** | Elastic scaling controller for prefill/decode/expert components |
| **G3** | Serving-time autotuning across GPU slices and runtime parameters |
| **G4** | Reproducible experiments with constrained GPU resources |

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU workers)
- Redis server
- Docker (for K8s deployment)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Local Development (Single Machine)

```bash
# Start Redis
redis-server &

# Start workers and gateway
python -m workers.prefill &
python -m workers.decode &
python -m gateway.server &

# Run evaluation
python -m eval.runner
```

### Kubernetes Deployment

```bash
# Build Docker images
docker build -f docker/gateway.Dockerfile -t llm-inference/gateway:latest .
docker build -f docker/worker.Dockerfile -t llm-inference/worker:latest .
docker build -f docker/controller.Dockerfile -t llm-inference/controller:latest .

# Deploy to cluster
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/
```

### Run Multi-Model Experiment (SLURM)

```bash
sbatch scripts/slurm_autotune.sh
```

This runs the full experiment across 4 models (OPT-1.3B, Qwen2-1.5B, Phi-2, OPT-6.7B) with profiling, knee analysis, autotune comparison, and ablation studies.

## Project Structure

```
├── gateway/          # Gateway/router gRPC service
├── workers/          # Prefill and decode GPU workers
├── scaling/          # Elastic scaling controller
├── autotune/         # Serving-time autotuning (knee profiling, config selection)
├── common/           # Shared utilities, proto definitions, config loading
├── eval/             # Evaluation runner and benchmarking
├── docker/           # Dockerfiles for gateway, worker, controller
├── k8s/              # Kubernetes manifests (deployments, services, HPA, KEDA)
├── monitoring/       # Prometheus monitoring setup
├── scripts/          # Experiment orchestration and figure generation
├── tests/            # Unit tests
├── results/          # Experiment results and figures
└── config.yaml       # Central configuration
```

## Experiment Design

The multi-model experiment (`scripts/run_multimodel_experiment.py`) runs 5 phases per model:

1. **Phase A — GPU% Profiling**: Measure latency/throughput at concurrency 1/2/4 (simulating 100%/50%/25% GPU)
2. **Phase B — Knee Analysis**: Find optimal GPU% operating point using Kneedle algorithm
3. **Phase C — Autotune Comparison**: Compare default vs optimized batch sizes (services restarted with different configs)
4. **Phase D — Ablation Study**: 4 variants with genuinely different configs:
   - `full_system` — optimal batches + MoE enabled
   - `no_autotune` — default batches + MoE enabled
   - `no_moe` — optimal batches + MoE disabled
   - `static_baseline` — default batches + MoE disabled
5. **Phase E — Figure Generation**: Per-model visualization

## Key Results

Tested across 4 models (1.3B–6.7B parameters):

- **Autotune** improves p95 latency up to 70% (Phi-2) and throughput up to 18% (OPT-1.3B/6.7B)
- **MoE simulation** creates measurable overhead; disabling reduces p95 by up to 69% (Qwen2)
- **Full system** (autotune + MoE) achieves highest throughput: up to 50.5 TPS (Qwen2)
- **Latency scales with model size** as expected: OPT-1.3B (412ms) → OPT-6.7B (996ms) at concurrency 1

## Configuration

All system parameters are in `config.yaml`:

- **Model**: name, dtype, max sequence length
- **Gateway**: port, queue size, request timeout
- **Workers**: batch sizes, max concurrency, MoE expert count
- **Scaling**: EMA alpha, cooldown, thresholds, anti-oscillation
- **Autotune**: GPU percentages, config search space

## Metrics

- **SLA**: p50/p95/p99 latency (per-stage and end-to-end)
- **Throughput**: tokens/s and requests/s
- **Efficiency**: GPU utilization, GPU-seconds per request
- **Scaling**: event count, oscillation frequency, convergence time

## Tests

```bash
pytest tests/
```
