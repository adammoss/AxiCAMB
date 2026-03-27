"""
Compute the effective A_lens implied by each nonlinear model for axion cosmologies.

For each nonlinear model, finds the A_lens value applied to LCDM that best
matches the model's lensed TT power spectrum.

Usage:
    python effective_alens.py --m_ax 1e-24
    python effective_alens.py --m_ax 1e-23 1e-24 1e-25
    python effective_alens.py --m_ax 1e-24 --lmax 4000
"""
import numpy as np
import sys, os
import argparse
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import axicamb_runner
import cosmo_params as cp


def compute_lcdm_cls(lcdm_kw, alens, lmax, halofit_version):
    """Compute LCDM lensed Cls at a given A_lens value."""
    result = axicamb_runner.get_lcdm(
        nonlinear=True, halofit_version=halofit_version,
        get_cls=True, do_lensing=True, lmax=lmax,
        Alens=alens, **lcdm_kw)
    return result['cls']


def fit_effective_alens(target_tt, lcdm_kw, lmax, halofit_version,
                        ell_min=100, ell_max=None):
    """Find A_lens that minimises chi2 between target TT and LCDM(A_lens) TT."""
    if ell_max is None:
        ell_max = lmax
    sl = slice(ell_min, ell_max + 1)

    ref_cls = compute_lcdm_cls(lcdm_kw, 1.0, lmax, halofit_version)
    ref_tt = ref_cls['tt'][sl]

    def chi2(alens):
        cls = compute_lcdm_cls(lcdm_kw, alens, lmax, halofit_version)
        diff = (cls['tt'][sl] - target_tt[sl]) / ref_tt
        return np.sum(diff**2)

    result = minimize_scalar(chi2, bounds=(0.8, 1.3), method='bounded')
    return result.x, result.fun


def main():
    parser = argparse.ArgumentParser(
        description='Compute effective A_lens for axion nonlinear models')
    cp.add_cli_args(parser)
    parser.add_argument('--lmax', type=int, default=3000)
    parser.add_argument('--ell_min', type=int, default=100)
    parser.add_argument('--ell_max', type=int, default=None)
    parser.add_argument('--halofit_version', type=str, default='mead2020')
    args = parser.parse_args()

    cosmo, axion = cp.from_args(args)
    m_ax_list = axion['m_ax_list']
    f_ax = axion['f_ax']
    lcdm_kw = cp.get_lcdm_kwargs(cosmo)

    print(f'f_ax = {f_ax}, lmax = {args.lmax}, '
          f'ell_range = [{args.ell_min}, {args.ell_max or args.lmax}]')
    print()

    model_names = ['Axion linear', 'Naive HMCode', 'axionHMcode basic', 'axionHMcode DOME']
    print(f'{"mass":>12s}', end='')
    for name in model_names:
        print(f'  {name:>20s}', end='')
    print()

    for m_ax in m_ax_list:
        axion['m_ax'] = m_ax
        ax_kw = cp.get_axicamb_kwargs(cosmo, axion)

        # Compute all nonlinear models
        ax_lin = axicamb_runner.run(
            get_cls=True, do_lensing=True, lmax=args.lmax, **ax_kw)
        ax_naive = axicamb_runner.run(
            nonlinear=True, halofit_version=args.halofit_version,
            get_cls=True, do_lensing=True, lmax=args.lmax, **ax_kw)
        ax_basic = axicamb_runner.run_with_axionhmcode(
            dome_calibrated=False, lmax=args.lmax, **ax_kw)
        ax_dome = axicamb_runner.run_with_axionhmcode(
            dome_calibrated=True, lmax=args.lmax, **ax_kw)

        models = [
            (model_names[0], ax_lin['cls']),
            (model_names[1], ax_naive['cls']),
            (model_names[2], ax_basic['cls']),
            (model_names[3], ax_dome['cls']),
        ]

        exp = int(np.log10(m_ax))
        print(f'  10^{exp:d}', end='')

        for name, cls in models:
            alens_eff, chi2 = fit_effective_alens(
                cls['tt'], lcdm_kw, args.lmax, args.halofit_version,
                ell_min=args.ell_min, ell_max=args.ell_max)
            print(f'  {alens_eff:20.4f}', end='')

        print()


if __name__ == '__main__':
    main()
