#!/bin/bash
# Generate all comparison figures between AxiCAMB and AxiECAMB.
# Run from the scripts/ directory.

set -e
cd "$(dirname "$0")"
rm -f figures/*.pdf

echo "=== P(k) ==="
python compare_pk.py --f_ax 0.3 --m_ax 1e-23 --z 0.0
python compare_pk.py --f_ax 0.3 --m_ax 1e-24 --z 0.0
python compare_pk.py --f_ax 0.3 --m_ax 1e-25 --z 0.0
python compare_pk.py --f_ax 0.3 --m_ax 1e-26 --z 0.0
python compare_pk.py --f_ax 0.3 --m_ax 1e-23 --z 2.0
python compare_pk.py --f_ax 0.3 --m_ax 1e-24 --z 2.0
python compare_pk.py --f_ax 0.3 --m_ax 1e-25 --z 2.0
python compare_pk.py --f_ax 0.001 --m_ax 1e-24 --z 0.0

echo "=== Cls ==="
python compare_cls.py --f_ax 0.3 --m_ax 1e-23 --lmax 2500
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
python compare_perturbations.py --f_ax 0.3 --m_ax 1e-23 --k 0.1 0.5 1.0
python compare_perturbations.py --f_ax 0.3 --m_ax 1e-24 --k 0.1 0.5 1.0
python compare_perturbations.py --f_ax 0.3 --m_ax 1e-25 --k 0.1 0.5 1.0
python compare_perturbations.py --f_ax 0.3 --m_ax 1e-26 --k 0.1 0.5 1.0
python compare_perturbations.py --f_ax 0.001 --m_ax 1e-24 --k 0.1 0.5 1.0

echo "=== P(k) plots ==="
python plot_pk.py --f_ax 0.3 --m_ax 1e-23 --z 0.0 2.0 --show_naive --layout column
python plot_pk.py --f_ax 0.3 --m_ax 1e-24 --z 0.0 2.0 --show_naive --layout column
python plot_pk.py --f_ax 0.3 --m_ax 1e-25 --z 0.0 2.0 --show_naive --layout column
python plot_pk.py --m_ax 1e-23 1e-24 1e-25 --z 0.0 2.0 --show_naive 

echo "=== Cls plots ==="
python plot_cls.py --f_ax 0.3 --m_ax 1e-23 --layout column --Alens 1.05
python plot_cls.py --f_ax 0.3 --m_ax 1e-24 --layout column --Alens 1.05
python plot_cls.py --f_ax 0.3 --m_ax 1e-25 --layout column --Alens 1.05
python plot_cls.py --m_ax 1e-23 1e-24 1e-25 --f_ax 0.3 --Alens 1.05

echo "=== Done ==="
echo "Generated $(ls figures/*.pdf | wc -l) figures in figures/"
