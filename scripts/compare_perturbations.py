"""
Compare perturbation evolution between AxiCAMB and AxiECAMB.

Plots delta_cdm(a) and delta_axion(a) for specified k modes.

Usage:
    python compare_perturbations.py --f_ax 0.3 --m_ax 1e-24 --k 0.1 0.5 1.0
    python compare_perturbations.py --f_ax 0.001 --m_ax 1e-24 --k 0.5
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import argparse
import os

import axicamb_runner
import axiecamb_runner
import cosmo_params

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description='Compare perturbation evolution')
    cosmo_params.add_cli_args(parser)
    parser.add_argument('--k', type=float, nargs='+', default=[0.1, 0.5, 1.0],
                        help='k modes in h/Mpc (default: 0.1 0.5 1.0)')
    args = parser.parse_args()

    cosmo, axion = cosmo_params.from_args(args)
    k_modes = args.k

    print(f'Parameters: m_ax={axion["m_ax"]:.0e}, f_ax={axion["f_ax"]}, '
          f'movH={axion["movH_switch"]}, k={k_modes}')

    nk = len(k_modes)
    fig, axes = plt.subplots(3, nk, figsize=(5 * nk, 12),
                              squeeze=False)

    ax_kw = cosmo_params.get_axicamb_kwargs(cosmo, axion)
    ae_kw = cosmo_params.get_axiecamb_kwargs(cosmo, axion)

    for ik, k_hMpc in enumerate(k_modes):
        print(f'\nk = {k_hMpc} h/Mpc:')

        # AxiCAMB perturbation evolution
        print('  AxiCAMB...')
        ax_pert = axicamb_runner.get_perturbation_evolution(
            k_hMpc=k_hMpc, **ax_kw)

        # AxiECAMB perturbation evolution
        print('  AxiECAMB...')
        ae_result = axiecamb_runner.run(
            z_arr=[0.0], pert_output_kh=k_hMpc, verbose=False, **ae_kw)
        ae_pert = ae_result['pert_evolution']

        if ae_pert is None:
            print('  WARNING: No perturbation evolution from AxiECAMB')
            continue

        # Remove duplicate points from AxiECAMB
        mask = np.diff(ae_pert['a'], prepend=-1) > 0
        a_ae = ae_pert['a'][mask]
        cdm_ae = ae_pert['delta_cdm'][mask]
        ax_ae = ae_pert['delta_axion'][mask]

        a_ax = ax_pert['a']
        cdm_ax = ax_pert['delta_cdm']
        ax_ax = ax_pert['delta_axion']

        # Top row: delta_cdm (raw)
        a = axes[0, ik]
        a.plot(a_ae, np.abs(cdm_ae), 'g-', lw=1.5, label='AxiECAMB')
        a.plot(a_ax, np.abs(cdm_ax), 'r--', lw=1.5, label='AxiCAMB')
        a.set_xscale('log'); a.set_yscale('log')
        a.set_ylabel(r'$|\delta_{cdm}|$')
        a.set_title(f'k = {k_hMpc} h/Mpc')
        a.grid(alpha=0.3); a.set_xlim(1e-5, 1)
        if ik == 0:
            a.legend(fontsize=8)

        # Middle row: delta_axion (raw)
        a = axes[1, ik]
        m_ae = np.abs(ax_ae) > 0
        a.plot(a_ae[m_ae], np.abs(ax_ae[m_ae]), 'g-', lw=1.5, label='AxiECAMB')
        a.plot(a_ax, np.abs(ax_ax), 'r--', lw=1.5, label='AxiCAMB')
        a.set_xscale('log'); a.set_yscale('log')
        a.set_ylabel(r'$|\delta_{ax}|$')
        a.grid(alpha=0.3); a.set_xlim(1e-5, 1)
        if ik == 0:
            a.legend(fontsize=8)

        # Bottom row: raw ratios
        a = axes[2, ik]
        f_ae_cdm = interp1d(np.log(a_ae), cdm_ae,
                             bounds_error=False, fill_value=np.nan)
        ratio_cdm = cdm_ax / f_ae_cdm(np.log(a_ax))
        valid = np.isfinite(ratio_cdm) & (a_ax > 1e-5) & (a_ax < 0.999)
        a.plot(a_ax[valid], ratio_cdm[valid], 'b-', lw=1.5, label=r'$\delta_{cdm}$')

        if np.sum(m_ae) > 2:
            f_ae_ax = interp1d(np.log(a_ae[m_ae]), ax_ae[m_ae],
                                bounds_error=False, fill_value=np.nan)
            ratio_ax = ax_ax / f_ae_ax(np.log(a_ax))
            valid_ax = np.isfinite(ratio_ax) & (a_ax > 1e-4) & (a_ax < 0.999)
            a.plot(a_ax[valid_ax], ratio_ax[valid_ax], 'orange', lw=1.5,
                   ls='--', label=r'$\delta_{ax}$')

        a.axhline(1.0, color='gray', ls=':', alpha=0.5)
        a.set_xscale('log')
        a.set_xlabel('a')
        a.set_ylabel('AxiCAMB / AxiECAMB')
        a.grid(alpha=0.3); a.set_xlim(1e-5, 1)
        a.set_ylim(0.9, 1.1)
        if ik == 0:
            a.legend(fontsize=8)

        # Print z=0 values
        print(f'  delta_cdm(z=0): AxiCAMB={cdm_ax[-1]:.2f}, AxiECAMB={cdm_ae[-1]:.2f}')
        print(f'  delta_ax(z=0):  AxiCAMB={ax_ax[-1]:.2f}, AxiECAMB={ax_ae[-1]:.2f}')

    fig.suptitle(
        f'Perturbation evolution: m={axion["m_ax"]:.0e} eV, f={axion["f_ax"]}, '
        f'movH={axion["movH_switch"]:.0f}',
        fontsize=13)
    plt.tight_layout()

    tag = f'm{axion["m_ax"]:.0e}_f{axion["f_ax"]}_movH{axion["movH_switch"]:.0f}'.replace('.', 'p')
    outpath = os.path.join(FIGDIR, f'perturbations_{tag}.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'\nSaved {outpath}')


if __name__ == '__main__':
    main()
