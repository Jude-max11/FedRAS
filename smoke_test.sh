#!/usr/bin/env bash
# One-minute smoke test: no GPU, no dataset download.
set -e
cd "$(dirname "$0")"
OUT=smoke_out
rm -rf "$OUT"; mkdir -p "$OUT/figs"

echo "[1/4] scheduler simulator (NumPy only)"
python3 scripts/simulator.py --rounds 3 --clients 60 --seeds 0 --out "$OUT/sim"

echo "[2/4] aggregate runs -> Table 3"
python3 scripts/collect_v6.py --root outputs/real_v6 > "$OUT/collect.txt"
python3 scripts/emit_tab3_zh.py > "$OUT/table3.tex"

echo "[3/4] Figures 2 and 3"
python3 scripts/make_v6_figs.py "$OUT/figs"

echo "[4/4] Figures 4 and 5"
python3 scripts/make_v6_fig4.py "$OUT/figs"
python3 scripts/make_fig5a.py "$OUT/figs"
python3 scripts/make_fig5b.py "$OUT/figs"

echo
echo "OK. Results in $OUT/"
ls -1 "$OUT" "$OUT/figs"
