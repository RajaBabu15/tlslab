#!/usr/bin/env bash
# Experiments A (spsc-pair), B (batched), C (spsc-shard), D (queue-only retry counts).
# 1 warmup + 5 trials × 3s per cell.

set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./baseline
OUT=results/experiments.csv
DUR=3.0

[[ -x $BIN ]] || { echo "build first: make" >&2; exit 1; }
mkdir -p results
echo "NP,NC,mode,trial,total_ops,duration_ns,ops_per_sec,ns_per_op,min_thread_rate,max_thread_rate,sum_prod_rate,total_iters,iters_per_op" > "$OUT"

run_cell() {
    local NP=$1 NC=$2 mode=$3
    local label="NP=${NP} NC=${NC} mode=${mode}"
    printf '[warmup ] %s\n' "$label" >&2
    "$BIN" "$NP" "$NC" "$mode" "$DUR" > /dev/null
    for trial in 1 2 3 4 5; do
        local row
        row=$("$BIN" "$NP" "$NC" "$mode" "$DUR")
        echo "$row" | sed "s/,/,${trial},/3" >> "$OUT"
        printf '[trial %d] %s  ops/sec=%s iters/op=%s\n' \
            "$trial" "$label" \
            "$(awk -F, '{print $6}' <<< "$row")" \
            "$(awk -F, '{print $12}' <<< "$row")" >&2
    done
}

for N in 1 2 4 8 16; do run_cell "$N" "$N" queue-only; done
for N in 1 2 4 8 16; do run_cell "$N" "$N" spsc-pair;  done
for N in 1 2 4 8 16; do run_cell "$N" "$N" spsc-shard; done
for N in 2 4 8 16; do
    run_cell "$N" "$N" batch8
    run_cell "$N" "$N" batch16
    run_cell "$N" "$N" batch32
done

echo >&2
echo "=== Summary ===" >&2
python3 - "$OUT" <<'PYEOF'
import csv, sys
from collections import defaultdict

rows = defaultdict(list)
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        key = (int(row['NP']), int(row['NC']), row['mode'])
        rows[key].append((float(row['ops_per_sec']),
                          float(row['min_thread_rate']),
                          float(row['max_thread_rate']),
                          float(row['iters_per_op'])))

def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]

def show(key):
    ops, pt_min, pt_max, ipo = zip(*rows[key])
    return med(ops), med(pt_min), med(pt_max), med(ipo)

hdr = f"{'NP':>3} {'NC':>3} {'mode':>12}  {'med_ops/sec':>15}  {'pt_min':>13}  {'pt_max':>13}  {'iters/op':>9}"

print()
print("=== D (queue-only iters), A (spsc-pair), C (spsc-shard) ===")
print(hdr)
print('-' * len(hdr))
for mode in ('queue-only', 'spsc-pair', 'spsc-shard'):
    for N in (1, 2, 4, 8, 16):
        key = (N, N, mode)
        if key not in rows: continue
        m, lo, hi, ipo = show(key)
        print(f"{N:>3} {N:>3} {mode:>12}  {m:>15,.0f}  {lo:>13,.0f}  {hi:>13,.0f}  {ipo:>9.3f}")
    print()

print("=== B (batched dequeue) ===")
print(hdr)
print('-' * len(hdr))
for N in (2, 4, 8, 16):
    for mode in ('queue-only', 'batch8', 'batch16', 'batch32'):
        key = (N, N, mode)
        if key not in rows: continue
        m, lo, hi, ipo = show(key)
        print(f"{N:>3} {N:>3} {mode:>12}  {m:>15,.0f}  {lo:>13,.0f}  {hi:>13,.0f}  {ipo:>9.3f}")
    print()
PYEOF
