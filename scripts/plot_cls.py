"""
Plot lensed Cls ratios for different axion nonlinear models vs LCDM.

Shows C_l^axion / C_l^LCDM for:
  - Linear (no nonlinear corrections)
  - Naive HMCode (CAMB built-in, wrong for axions)
  - axionHMcode basic
  - axionHMcode DOME

Usage:
    python plot_cls.py --f_ax 0.3 --m_ax 1e-24
    python plot_cls.py --f_ax 0.3 --m_ax 1e-25 --layout column
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import axicamb_runner
import cosmo_params as cp

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Plot lensed Cls ratios')
    cp.add_cli_args(parser)
    parser.add_argument('--ax_fraction', type=float, default=0.3)
    parser.add_argument('--lmax', type=int, default=3000)
    parser.add_argument('--halofit_version', type=str, default='mead2020')
    parser.add_argument('--layout', type=str, default='row',
                        choices=['row', 'column'])
    parser.add_argument('--Alens', type=float, nargs='*', default=[1.05, 1.1],
                        help='A_lens values to show for LCDM (default: 1.05 1.1)')
    args = parser.parse_args()

    cosmo, axion = cp.from_args(args)
    axion['f_ax'] = args.ax_fraction
    m_ax = axion['m_ax']
    f_ax = args.ax_fraction

    lcdm_kw = cp.get_lcdm_kwargs(cosmo)
    ax_kw = cp.get_axicamb_kwargs(cosmo, axion)

    # 1. LCDM nonlinear (reference)
    print('Computing LCDM HMCode...')
    lcdm = axicamb_runner.get_lcdm(
        nonlinear=True, halofit_version=args.halofit_version,
        get_cls=True, do_lensing=True, lmax=args.lmax, **lcdm_kw)

    # 2. Axion linear
    print('Computing axion linear...')
    ax_lin = axicamb_runner.run(
        get_cls=True, do_lensing=True, lmax=args.lmax, **ax_kw)

    # 3. Axion naive HMCode
    print('Computing axion naive HMCode...')
    ax_naive = axicamb_runner.run(
        nonlinear=True, halofit_version=args.halofit_version,
        get_cls=True, do_lensing=True, lmax=args.lmax, **ax_kw)

    # 4. axionHMcode basic
    print('Computing axionHMcode basic...')
    ax_basic = axicamb_runner.run_with_axionhmcode(
        dome_calibrated=False, lmax=args.lmax, **ax_kw)

    # 5. axionHMcode DOME
    print('Computing axionHMcode DOME...')
    ax_dome = axicamb_runner.run_with_axionhmcode(
        dome_calibrated=True, lmax=args.lmax, **ax_kw)

    # 6. LCDM with A_lens
    alens_results = {}
    if args.Alens:
        for alens_val in args.Alens:
            print(f'Computing LCDM A_lens={alens_val}...')
            alens_results[alens_val] = axicamb_runner.get_lcdm(
                nonlinear=True, halofit_version=args.halofit_version,
                get_cls=True, do_lensing=True, lmax=args.lmax,
                Alens=alens_val, **lcdm_kw)

    # Plot
    ell = np.arange(args.lmax + 1)

    models = [
        ('Linear', ax_lin['cls'], 'C0', '--'),
        ('Naive HMCode', ax_naive['cls'], 'C2', '-'),
        ('axionHMcode basic', ax_basic['cls'], 'C0', '-'),
        ('axionHMcode DOME', ax_dome['cls'], 'C3', '-'),
    ]
    # Add A_lens models
    for alens_val in sorted(alens_results.keys()):
        models.append(
            (f'$\\Lambda$CDM $A_\\mathrm{{lens}}={alens_val}$',
             alens_results[alens_val]['cls'], 'gray', ':')
        )

    if args.layout == 'column':
        fig, axes = plt.subplots(2, 1, figsize=(4.5, 6), sharex=True,
                                  gridspec_kw={'hspace': 0.05})
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (ax, label, key) in enumerate(
            zip(axes, ['TT', 'EE'], ['tt', 'ee'])):
        ref = lcdm['cls'][key]
        valid = np.abs(ref) > 1e-10 * np.max(np.abs(ref))

        for name, cls, color, ls in models:
            ratio = np.ones_like(ref)
            ratio[valid] = cls[key][valid] / ref[valid]
            ax.plot(ell[2:], (ratio[2:] - 1) * 100, color=color, ls=ls,
                    lw=1.2, label=name)

        ax.axhline(0, color='k', ls=':', alpha=0.5, lw=0.8)
        ax.axhspan(-0.1, 0.1, color='gray', alpha=0.1)

        if args.layout == 'column':
            if idx == 1:
                ax.set_xlabel(r'$\ell$')
            ax.set_ylabel(f'{label}: ' + r'$C_\ell^\mathrm{axion}/C_\ell^{\Lambda\mathrm{CDM}} - 1$ [%]')
        else:
            ax.set_xlabel(r'$\ell$')
            ax.set_ylabel(r'$C_\ell^\mathrm{axion}/C_\ell^{\Lambda\mathrm{CDM}} - 1$ [%]')
            ax.set_title(label)

        ax.set_xlim(2, args.lmax)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2)

    if args.layout != 'column':
        from plot_pk import format_mass_label
        fig.suptitle(
            f'$m_a = {format_mass_label(m_ax)}$ eV, '
            f'$f_\\mathrm{{ax}} = {f_ax}$', fontsize=12)

    plt.tight_layout()
    tag = f'cls_m{m_ax:.0e}_f{f_ax}'.replace('.', 'p')
    outpath = os.path.join(FIGDIR, f'{tag}.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'\nSaved {outpath}')
    plt.close()


if __name__ == '__main__':
    main()
