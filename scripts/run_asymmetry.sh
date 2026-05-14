#!/usr/bin/env bash
# Three queue-only configs to isolate head vs tail contention:
#   (1,1) baseline, (2,1) tail contended, (1,2) head contended.
# 1 warmup + 5 trials × 3s per config.

set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./baseline
OUT=results/asymmetry.csv
DUR=3.0
MODE=queue-only

[[ -x $BIN ]] || { echo "build first: make" >&2; exit 1; }
mkdir -p results
echo "NP,NC,mode,trial,total_ops,duration_ns,ops_per_sec,ns_per_op,min_thread_rate,max_thread_rate,sum_prod_rate,total_iters,iters_per_op" > "$OUT"

run_cell() {
    local NP=$1 NC=$2 label=$3
    printf '[warmup ] %s\n' "$label" >&2
    "$BIN" "$NP" "$NC" "$MODE" "$DUR" > /dev/null
    for trial in 1 2 3 4 5; do
        local row
        row=$("$BIN" "$NP" "$NC" "$MODE" "$DUR")
        echo "$row" | sed "s/,/,${trial},/3" >> "$OUT"
        printf '[trial %d] %s  ops/sec=%s\n' \
            "$trial" "$label" "$(awk -F, '{print $6}' <<< "$row")" >&2
    done
}

run_cell 1 1 "1P 1C (baseline)"
run_cell 2 1 "2P 1C (tail contended)"
run_cell 1 2 "1P 2C (head contended)"

echo >&2
echo "=== Median over 5 trials per (NP, NC) ===" >&2
python3 - "$OUT" <<'PYEOF'
import csv, sys
from collections import defaultdict

rows = defaultdict(list)
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        rows[(int(row['NP']), int(row['NC']))].append(float(row['ops_per_sec']))

def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]

base = med(rows[(1, 1)]) if (1, 1) in rows else None
hdr = f"{'NP':>3} {'NC':>3}  {'med_ops/sec':>15}  {'Δ vs (1,1)':>12}"
print(hdr)
print('-' * len(hdr))
for NP, NC in ((1, 1), (2, 1), (1, 2)):
    if (NP, NC) not in rows: continue
    m = med(rows[(NP, NC)])
    if base and (NP, NC) != (1, 1):
        delta = f"{(m / base - 1) * 100:+.1f}%"
    else:
        delta = "—"
    print(f"{NP:>3} {NC:>3}  {m:>15,.0f}  {delta:>12}")
PYEOF
