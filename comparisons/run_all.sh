#!/bin/bash
# Generate all comparison figures between AxiCAMB and AxiECAMB.
# Run from the comparisons/ directory.

set -e
cd "$(dirname "$0")"
rm -f figures/*.pdf

echo "=== P(k) ==="
python compare_pk.py --f_ax 0.3 --m_ax 1e-24 --z 0.0
python compare_pk.py --f_ax 0.3 --m_ax 1e-25 --z 0.0
python compare_pk.py --f_ax 0.3 --m_ax 1e-26 --z 0.0
python compare_pk.py --f_ax 0.001 --m_ax 1e-24 --z 0.0

echo "=== Cls ==="
python compare_cls.py --f_ax 0.3 --m_ax 1e-24 --lmax 2500
python compare_cls.py --f_ax 0.3 --m_ax 1e-24 --lmax 2500 --no_lensing
python compare_cls.py --f_ax 0.3 --m_ax 1e-24 --lmax 2500 --movH_switch 200
python compare_cls.py --f_ax 0.3 --m_ax 1e-25 --lmax 2500
python compare_cls.py --f_ax 0.3 --m_ax 1e-26 --lmax 2500
python compare_cls.py --f_ax 0.3 --m_ax 1e-27 --lmax 2500
python compare_cls.py --f_ax 0.3 --m_ax 1e-28 --lmax 2500
python compare_cls.py --f_ax 0.001 --m_ax 1e-24 --lmax 2500

echo "=== Background ==="
python compare_background.py --f_ax 0.3 --m_ax 1e-24 --movH_switch 50
python compare_background.py --f_ax 0.3 --m_ax 1e-24 --movH_switch 500

echo "=== Perturbations ==="
python compare_perturbations.py --f_ax 0.3 --m_ax 1e-24 --k 0.1 0.5 1.0
python compare_perturbations.py --f_ax 0.3 --m_ax 1e-25 --k 0.1 0.5 1.0
python compare_perturbations.py --f_ax 0.3 --m_ax 1e-26 --k 0.1 0.5 1.0
python compare_perturbations.py --f_ax 0.001 --m_ax 1e-24 --k 0.1 0.5 1.0

echo "=== Done ==="
echo "Generated $(ls figures/*.pdf | wc -l) figures in figures/"
