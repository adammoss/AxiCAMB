"""
Compare linear P(k) between AxiCAMB and AxiECAMB.

Usage:
    python compare_pk.py --f_ax 0.3 --m_ax 1e-24 --z 0.0 1.0 2.0 3.0
    python compare_pk.py --f_ax 0.001 --m_ax 1e-24
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
    parser = argparse.ArgumentParser(description='Compare AxiCAMB vs AxiECAMB P(k)')
    cosmo_params.add_cli_args(parser)
    parser.add_argument('--z', type=float, nargs='+', default=[0.0, 1.0, 2.0, 3.0])
    args = parser.parse_args()

    cosmo, axion = cosmo_params.from_args(args)
    z_arr = sorted(args.z)

    print(f'Parameters: m_ax={axion["m_ax"]:.0e}, f_ax={axion["f_ax"]}, '
          f'z={z_arr}, movH={axion["movH_switch"]}')
    print()

    # Run AxiCAMB
    print('Running AxiCAMB...')
    ax_kw = cosmo_params.get_axicamb_kwargs(cosmo, axion)
    ax = axicamb_runner.run(z_arr=z_arr, **ax_kw)

    # Run AxiECAMB
    print('Running AxiECAMB...')
    ae_kw = cosmo_params.get_axiecamb_kwargs(cosmo, axion)
    ae = axiecamb_runner.run(z_arr=z_arr, **ae_kw)

    # LCDM reference
    print('Running LCDM...')
    lcdm_kw = cosmo_params.get_lcdm_kwargs(cosmo)
    k_lcdm, z_lcdm, pk_lcdm = axicamb_runner.get_lcdm_pk(z_arr=z_arr, **lcdm_kw)

    # Plot
    nz = len(z_arr)
    fig, axes = plt.subplots(2, nz, figsize=(5 * nz, 8),
                              gridspec_kw={'height_ratios': [2, 1]},
                              sharex='col')
    if nz == 1:
        axes = axes.reshape(2, 1)

    k_compare = np.geomspace(0.01, min(ax['k'].max(), ae['k'].max(), 10), 200)

    for iz, z in enumerate(z_arr):
        iz_ax = np.argmin(np.abs(ax['z'] - z))
        iz_ae = np.argmin(np.abs(ae['z'] - z))
        iz_lcdm = np.argmin(np.abs(z_lcdm - z))

        # Top: P(k)
        a = axes[0, iz]
        a.loglog(ax['k'], ax['pk'][iz_ax], 'r-', lw=1.5, label='AxiCAMB')
        a.loglog(ae['k'], ae['pk'][iz_ae], 'g--', lw=1.5, label='AxiECAMB')
        a.loglog(k_lcdm, pk_lcdm[iz_lcdm], 'k:', lw=1, alpha=0.5, label='LCDM')
        a.set_title(f'z = {z:.1f}')
        a.set_xlim(0.01, 10)
        a.grid(alpha=0.3)
        if iz == 0:
            a.set_ylabel(r'$P(k)$ [$(\mathrm{Mpc}/h)^3$]')
            a.legend(fontsize=8)

        # Bottom: ratio
        a2 = axes[1, iz]
        f_ax_interp = interp1d(np.log(ax['k']), np.log(ax['pk'][iz_ax]),
                                bounds_error=False, fill_value=np.nan)
        f_ae_interp = interp1d(np.log(ae['k']), np.log(ae['pk'][iz_ae]),
                                bounds_error=False, fill_value=np.nan)
        ratio = np.exp(f_ax_interp(np.log(k_compare))) / \
                np.exp(f_ae_interp(np.log(k_compare)))
        valid = np.isfinite(ratio)
        a2.semilogx(k_compare[valid], ratio[valid], 'k-', lw=1.5)
        a2.axhline(1.0, color='gray', ls=':', alpha=0.5)
        a2.set_xlabel(r'$k$ [$h$/Mpc]')
        a2.set_xlim(0.01, 10)
        a2.set_ylim(0.85, 1.15)
        a2.grid(alpha=0.3)
        if iz == 0:
            a2.set_ylabel('AxiCAMB / AxiECAMB')

    fig.suptitle(
        f'$m_a = {axion["m_ax"]:.0e}$ eV, $f_{{ax}} = {axion["f_ax"]}$, '
        f'movH={axion["movH_switch"]:.0f}',
        fontsize=13)
    plt.tight_layout()

    tag = f'm{axion["m_ax"]:.0e}_f{axion["f_ax"]}_movH{axion["movH_switch"]:.0f}'.replace('.', 'p')
    outpath = os.path.join(FIGDIR, f'pk_comparison_{tag}.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'\nSaved {outpath}')

    # Summary
    ae_s8 = [s for s in ae['sigma8'] if not np.isnan(s)]
    print(f'\nsigma8(z=0): AxiCAMB={ax["sigma8"]:.4f}, AxiECAMB={ae_s8[-1]:.4f}')

    # Use last z for summary ratios
    iz_ax = np.argmin(np.abs(ax['z'] - z_arr[0]))
    iz_ae = np.argmin(np.abs(ae['z'] - z_arr[0]))
    f_ax_interp = interp1d(np.log(ax['k']), np.log(ax['pk'][iz_ax]),
                            bounds_error=False, fill_value=np.nan)
    f_ae_interp = interp1d(np.log(ae['k']), np.log(ae['pk'][iz_ae]),
                            bounds_error=False, fill_value=np.nan)
    print(f'\nP(k) ratio AxiCAMB/AxiECAMB at z={z_arr[0]}:')
    for kval in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        r = np.exp(f_ax_interp(np.log(kval))) / np.exp(f_ae_interp(np.log(kval)))
        if np.isfinite(r):
            print(f'  k={kval}: {r:.4f}')


if __name__ == '__main__':
    main()
