#!/usr/bin/env bash
# Sweep `full` and `queue-only` over N ∈ {1,2,4,8,16}, 1 warmup + 5 trials × 3s.
# Output: results/baseline.csv plus median summary on stderr.

set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./baseline
OUT=results/baseline.csv
DUR=3.0

[[ -x $BIN ]] || { echo "build first: make" >&2; exit 1; }
mkdir -p results
echo "NP,NC,mode,trial,total_ops,duration_ns,ops_per_sec,ns_per_op,min_thread_rate,max_thread_rate,sum_prod_rate,total_iters,iters_per_op" > "$OUT"

for mode in full queue-only; do
    for N in 1 2 4 8 16; do
        printf '[warmup ] N=%2d mode=%-10s\n' "$N" "$mode" >&2
        "$BIN" "$N" "$N" "$mode" "$DUR" > /dev/null
        for trial in 1 2 3 4 5; do
            row=$("$BIN" "$N" "$N" "$mode" "$DUR")
            echo "$row" | sed "s/,/,${trial},/3" >> "$OUT"
            printf '[trial %d] N=%2d mode=%-10s  ops/sec=%s\n' \
                "$trial" "$N" "$mode" "$(awk -F, '{print $6}' <<< "$row")" >&2
        done
    done
done

echo >&2
echo "=== Median over 5 trials per (N, mode) ===" >&2
python3 - "$OUT" <<'PYEOF'
import csv, sys
from collections import defaultdict

rows = defaultdict(list)
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        rows[(int(row['NP']), row['mode'])].append(float(row['ops_per_sec']))

def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]

base = {m: med(rows[(1, m)]) for m in ('full', 'queue-only') if (1, m) in rows}
hdr = f"{'N':>3} {'mode':>10} {'med_ops/sec':>15} {'eff_vs_N=1':>12}"
print(hdr)
print('-' * len(hdr))
for mode in ('full', 'queue-only'):
    for N in (1, 2, 4, 8, 16):
        if (N, mode) not in rows: continue
        m = med(rows[(N, mode)])
        eff = (m / N) / base[mode] if mode in base else float('nan')
        print(f"{N:>3} {mode:>10} {m:>15,.0f} {eff:>11.1%}")
    print()
PYEOF
