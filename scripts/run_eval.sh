#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GATEWAY="${GATEWAY_TARGET:-localhost:50051}"
NUM_REQUESTS="${NUM_REQUESTS:-100}"
ARRIVAL_RATE="${ARRIVAL_RATE:-10}"
MAX_TOKENS="${MAX_TOKENS:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/eval_results}"

mkdir -p "$OUTPUT_DIR"

echo "=== Running evaluation workload ==="
echo "  Gateway:      $GATEWAY"
echo "  Requests:     $NUM_REQUESTS"
echo "  Arrival rate: $ARRIVAL_RATE rps"
echo "  Max tokens:   $MAX_TOKENS"
echo "  Output:       $OUTPUT_DIR"
echo ""

cd "$PROJECT_DIR"
python -c "
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

from eval.workloads import WorkloadGenerator
from eval.plots import plot_latency_cdf, plot_latency_breakdown, plot_throughput_over_time, save_results_json

gen = WorkloadGenerator(
    gateway_target='${GATEWAY}',
    num_requests=${NUM_REQUESTS},
    arrival_rate=${ARRIVAL_RATE},
    max_tokens=${MAX_TOKENS},
)
summary = gen.run()

records = gen.collector.records
output_dir = '${OUTPUT_DIR}'

save_results_json(summary, records, f'{output_dir}/results.json')
plot_latency_cdf(records, f'{output_dir}/latency_cdf.png')
plot_latency_breakdown(summary, f'{output_dir}/latency_breakdown.png')
plot_throughput_over_time(records, output_path=f'{output_dir}/throughput.png')

print()
print('=== Summary ===')
for k, v in sorted(summary.items()):
    print(f'  {k}: {v:.2f}' if isinstance(v, float) else f'  {k}: {v}')
"
