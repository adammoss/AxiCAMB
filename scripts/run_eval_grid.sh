#!/bin/bash
# Generate chi2 grids for different nonlinear models.
# Run from the scripts/ directory.

set -e
cd "$(dirname "$0")"
mkdir -p output

M_RANGE="-25 -23 10"
F_RANGE="0.01 0.9 10"

echo "=== Naive HMCode ==="
python eval_likelihood.py --m_range $M_RANGE --f_range $F_RANGE \
    --output output/chi2_naive.npz

echo "=== axionHMcode basic ==="
python eval_likelihood.py --m_range $M_RANGE --f_range $F_RANGE \
    --nl_model basic --output output/chi2_basic.npz

echo "=== axionHMcode DOME ==="
python eval_likelihood.py --m_range $M_RANGE --f_range $F_RANGE \
    --nl_model dome --output output/chi2_dome.npz

echo "=== Plotting (with lensing) ==="
python eval_likelihood.py --plot output/chi2_naive.npz output/chi2_basic.npz output/chi2_dome.npz

echo "=== Plotting (without lensing) ==="
python eval_likelihood.py --plot output/chi2_naive.npz output/chi2_basic.npz output/chi2_dome.npz --no_lensing

echo "=== Done ==="
