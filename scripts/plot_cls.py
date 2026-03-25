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
    python plot_cls.py --f_ax 0.3 --m_ax 1e-23 1e-24 1e-25
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


def format_mass_label(m):
    exp = int(np.log10(m))
    return f'10^{{{exp}}}'


def compute_models(ax_kw, lcdm, args):
    """Compute all nonlinear models for a given axion parameter set."""
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
        ('Axion linear', ax_lin['cls'], 'C0', '--'),
        ('Naive HMCode', ax_naive['cls'], 'C2', '-'),
        ('axionHMcode basic', ax_basic['cls'], 'C0', '-'),
        ('axionHMcode DOME', ax_dome['cls'], 'C3', '-'),
    ]
    return models


def main():
    parser = argparse.ArgumentParser(description='Plot lensed Cls ratios')
    cp.add_cli_args(parser)
    parser.add_argument('--lmax', type=int, default=3000)
    parser.add_argument('--halofit_version', type=str, default='mead2020')
    parser.add_argument('--layout', type=str, default='row',
                        choices=['row', 'column'])
    parser.add_argument('--Alens', type=float, nargs='*', default=[1.05],
                        help='A_lens values to show for LCDM (default: 1.05)')
    args = parser.parse_args()

    cosmo, axion = cp.from_args(args)
    m_ax_list = axion['m_ax_list']
    f_ax = axion['f_ax']
    nm = len(m_ax_list)

    lcdm_kw = cp.get_lcdm_kwargs(cosmo)

    # LCDM nonlinear (reference, shared across masses)
    print('Computing LCDM HMCode...')
    lcdm = axicamb_runner.get_lcdm(
        nonlinear=True, halofit_version=args.halofit_version,
        get_cls=True, do_lensing=True, lmax=args.lmax, **lcdm_kw)

    # A_lens models (shared)
    alens_results = {}
    if args.Alens:
        for alens_val in args.Alens:
            print(f'Computing LCDM A_lens={alens_val}...')
            alens_results[alens_val] = axicamb_runner.get_lcdm(
                nonlinear=True, halofit_version=args.halofit_version,
                get_cls=True, do_lensing=True, lmax=args.lmax,
                Alens=alens_val, **lcdm_kw)

    # Compute models for each mass
    all_models = {}
    for m_ax in m_ax_list:
        print(f'\nComputing m_ax = {m_ax:.0e}...')
        axion['m_ax'] = m_ax
        ax_kw = cp.get_axicamb_kwargs(cosmo, axion)
        models = compute_models(ax_kw, lcdm, args)
        # Add A_lens
        for alens_val in sorted(alens_results.keys()):
            models.append(
                (f'$\\Lambda$CDM $A_\\mathrm{{lens}}={alens_val}$',
                 alens_results[alens_val]['cls'], 'gray', ':'))
        all_models[m_ax] = models

    ell = np.arange(args.lmax + 1)

    if nm == 1:
        # Single mass: TT and EE side by side or stacked
        m_ax = m_ax_list[0]
        models = all_models[m_ax]

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
            fig.suptitle(
                f'$m_a = {format_mass_label(m_ax)}$ eV, '
                f'$f_\\mathrm{{ax}} = {f_ax}$', fontsize=12)

        plt.tight_layout()
        tag = f'cls_m{m_ax:.0e}_f{f_ax}'.replace('.', 'p')

    else:
        # Multi-mass: columns = masses, rows = TT, EE
        fig, axes = plt.subplots(2, nm, figsize=(5 * nm, 6), sharex=True, sharey='row',
                                  gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
        if nm == 1:
            axes = axes.reshape(2, 1)

        for j, m_ax in enumerate(m_ax_list):
            models = all_models[m_ax]

            for i, (label, key) in enumerate(zip(['TT', 'EE'], ['tt', 'ee'])):
                ax = axes[i, j]
                ref = lcdm['cls'][key]
                valid = np.abs(ref) > 1e-10 * np.max(np.abs(ref))

                for name, cls, color, ls in models:
                    ratio = np.ones_like(ref)
                    ratio[valid] = cls[key][valid] / ref[valid]
                    show_label = (j == 0)  # legend only on first column
                    ax.plot(ell[2:], (ratio[2:] - 1) * 100, color=color, ls=ls,
                            lw=1.2, label=name if show_label else None)

                ax.axhline(0, color='k', ls=':', alpha=0.5, lw=0.8)

                if i == 0:
                    ax.set_title(f'$m_a = {format_mass_label(m_ax)}$ eV')
                if i == 1:
                    ax.set_xlabel(r'$\ell$')
                if j == 0:
                    ax.set_ylabel(f'{label}: ' + r'$C_\ell^\mathrm{axion}/C_\ell^{\Lambda\mathrm{CDM}} - 1$ [%]')
                else:
                    plt.setp(ax.get_yticklabels(), visible=False)
                ax.set_xlim(2, args.lmax)
                ax.grid(True, alpha=0.2)

                if j == 0 and i == 0:
                    ax.legend(fontsize=7, loc='upper left')

        plt.tight_layout()
        masses_str = '_'.join([f'm{m:.0e}' for m in m_ax_list])
        tag = f'cls_{masses_str}_f{f_ax}'.replace('.', 'p')

    outpath = os.path.join(FIGDIR, f'{tag}.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'\nSaved {outpath}')
    plt.close()

    # Lensing potential plot (single mass only)
    if nm == 1 and 'pp' in lcdm['cls']:
        m_ax = m_ax_list[0]
        models = all_models[m_ax]
        import matplotlib.gridspec as gridspec
        fig_pp = plt.figure(figsize=(4.5, 5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0)
        ax_top = fig_pp.add_subplot(gs[0])
        ax_bot = fig_pp.add_subplot(gs[1], sharex=ax_top)
        plt.setp(ax_top.get_xticklabels(), visible=False)

        ref_pp = lcdm['cls']['pp']
        valid_pp = np.abs(ref_pp) > 1e-10 * np.max(np.abs(ref_pp))

        ax_top.loglog(ell[2:], ref_pp[2:], color='k', ls='-', lw=1.2,
                      label=r'$\Lambda$CDM HMCode')
        for name, cls, color, ls in models:
            if 'pp' not in cls:
                continue
            ax_top.loglog(ell[2:], cls['pp'][2:], color=color, ls=ls,
                          lw=1.2, label=name)
        ax_top.set_ylabel(r'$[L(L+1)]^2 C_L^{\phi\phi} / 2\pi$')
        ax_top.set_xlim(2, args.lmax)
        ax_top.legend(fontsize=6.5, loc='lower left')
        ax_top.grid(True, alpha=0.2)

        for name, cls, color, ls in models:
            if 'pp' not in cls:
                continue
            ratio = np.ones_like(ref_pp)
            ratio[valid_pp] = cls['pp'][valid_pp] / ref_pp[valid_pp]
            ax_bot.semilogx(ell[2:], (ratio[2:] - 1) * 100, color=color, ls=ls,
                            lw=1.2)
        ax_bot.axhline(0, color='k', ls=':', alpha=0.5, lw=0.8)
        ax_bot.set_xlabel(r'$L$')
        ax_bot.set_ylabel(r'$C_L^{\phi\phi} / C_L^{\phi\phi,\,\Lambda\mathrm{CDM}} - 1$ [%]')
        ax_bot.set_xlim(2, args.lmax)
        ax_bot.grid(True, alpha=0.2)

        plt.tight_layout()
        tag_single = f'cls_m{m_ax:.0e}_f{f_ax}'.replace('.', 'p')
        outpath_pp = os.path.join(FIGDIR, f'clpp_{tag_single.replace("cls_", "")}.pdf')
        plt.savefig(outpath_pp, dpi=150, bbox_inches='tight')
        print(f'Saved {outpath_pp}')
        plt.close()


if __name__ == '__main__':
    main()
