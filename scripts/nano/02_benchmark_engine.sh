#!/usr/bin/env bash
# Benchmark engine bang trtexec (chi detector, batch 1). Ghi dieu kien chay vao log.
# Cach dung: bash scripts/nano/02_benchmark_engine.sh <engine> [iterations=200] [warmup_ms=1000]
set -eu
ENGINE=$1; ITER=${2:-200}; WARM=${3:-1000}
TRTEXEC=/usr/src/tensorrt/bin/trtexec
OUT="results/phase2/profile/nano_$(basename "$ENGINE" .engine)_trtexec.log"
mkdir -p "$(dirname "$OUT")"
{
  echo "== date: $(date -Is)"; echo "== nvpmodel: $(sudo nvpmodel -q 2>/dev/null | tr '\n' ' ')"
  echo "== jetson_clocks: $(sudo jetson_clocks --show 2>/dev/null | head -3 | tr '\n' ' ')"
  echo "== tegrastats: $(timeout 2 tegrastats 2>/dev/null | head -1)"
  $TRTEXEC --loadEngine="$ENGINE" --iterations="$ITER" --warmUp="$WARM" --avgRuns=10 --useCudaGraph --noDataTransfers=false
} 2>&1 | tee "$OUT"
echo "-> $OUT (xem 'GPU Compute Time' mean/median/percentile, 'Throughput', 'H2D/D2H')"
