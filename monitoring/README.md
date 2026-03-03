# Prometheus Observability for ElasticInference

This folder contains a production-style observability bundle for the PD-separation autoscaling stack:

- `grafana_dashboard_llm_inference.json`: Grafana dashboard template.
- `prometheus_alert_rules.yaml`: Alert rule set used by Prometheus.
- `analyze_decode_from_prometheus.py`: Offline analysis script (Prometheus API -> CSV/JSON/plot).

## 1) Deploy Prometheus + controller metrics

Controller metrics are exported on port `9100` and scraped by Prometheus.

Apply/update:

```bash
kubectl apply -f k8s/metrics-server.yaml
kubectl apply -f k8s/prometheus.yaml
kubectl -n llm-inference rollout restart deploy/scaling-controller deploy/prometheus
kubectl -n llm-inference rollout status deploy/scaling-controller --timeout=300s
kubectl -n llm-inference rollout status deploy/prometheus --timeout=300s
```

## 2) Dashboard

Import `monitoring/grafana_dashboard_llm_inference.json` in Grafana.

Recommended first view:
- Time range: last 30 minutes
- Refresh: 5s

Key panels:
- QPS / error rate / token throughput
- E2E / prefill / decode / TTFT p95
- Queue depth + active requests
- Decode/prefill target replicas
- Decode EMA load and cooldown

## 3) Alerts

Prometheus alert rules are embedded in `k8s/prometheus.yaml` and mirrored in
`monitoring/prometheus_alert_rules.yaml`.

Implemented alerts:
- `HighErrorRate`
- `E2ELatencyP95High`
- `DecodeQueueBacklog`
- `PrefillQueueBacklog`
- `NoTrafficButHighActive`
- `ScalingOscillation`

## 4) Offline analysis (decode workload)

Quick end-to-end run:

```bash
bash tests/run_decode_prometheus_observability.sh
```

Manual run:

```bash
kubectl -n llm-inference port-forward svc/prometheus 19090:9090
python3 monitoring/analyze_decode_from_prometheus.py \
  --prom-url http://127.0.0.1:19090 \
  --lookback-minutes 30 \
  --step-seconds 10 \
  --out-dir eval_results/prometheus_decode
```

Outputs:
- `decode_observability_timeseries.csv`
- `decode_observability_summary.json`
- `decode_observability_plot.png`
