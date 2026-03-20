#!/bin/bash
# Generate paper plots for P(k) and Cls comparisons.
# Run from the scripts/ directory or repo root.

set -e
cd "$(dirname "$0")"

echo "=== P(k) plots ==="
python plot_pk.py --f_ax 0.3 --m_ax 1e-24 --z 0.0 2.0 --show_naive --layout column
python plot_pk.py --f_ax 0.3 --m_ax 1e-25 --z 0.0 2.0 --show_naive --layout column

echo "=== Cls plots ==="
python plot_cls.py --f_ax 0.3 --m_ax 1e-24 --layout column --Alens 1.05
python plot_cls.py --f_ax 0.3 --m_ax 1e-25 --layout column --Alens 1.05

echo "=== Done ==="
ls figures/pk_*.pdf figures/cls_*.pdf 2>/dev/null
