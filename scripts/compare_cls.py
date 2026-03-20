"""
Compare lensed temperature Cls between AxiCAMB and AxiECAMB (no nonlinear).

Usage:
    python compare_cls.py --f_ax 0.3 --m_ax 1e-24
    python compare_cls.py --f_ax 0.001 --m_ax 1e-24
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

import axicamb_runner
import axiecamb_runner
import cosmo_params

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Compare Cls')
    cosmo_params.add_cli_args(parser)
    parser.add_argument('--lmax', type=int, default=3000)
    parser.add_argument('--no_lensing', action='store_true',
                        help='Compare unlensed (primary) Cls')
    args = parser.parse_args()

    cosmo, axion = cosmo_params.from_args(args)
    do_lensing = not args.no_lensing
    lens_label = 'lensed' if do_lensing else 'unlensed'

    print(f'Parameters: m_ax={axion["m_ax"]:.0e}, f_ax={axion["f_ax"]}, '
          f'lmax={args.lmax}, movH={axion["movH_switch"]}, {lens_label}')

    ax_kw = cosmo_params.get_axicamb_kwargs(cosmo, axion)
    ae_kw = cosmo_params.get_axiecamb_kwargs(cosmo, axion)
    lcdm_kw = cosmo_params.get_lcdm_kwargs(cosmo)

    cls_kw = dict(get_cls=True, do_lensing=do_lensing, lmax=args.lmax)

    print(f'\nRunning LCDM (AxiCAMB)...')
    lcdm = axicamb_runner.get_lcdm(**lcdm_kw, **cls_kw)['cls']

    print('Running LCDM (AxiECAMB, f=1e-6)...')
    ae_lcdm_kw = cosmo_params.get_axiecamb_kwargs(cosmo, {**axion, 'f_ax': 1e-6})
    lcdm_ae = axiecamb_runner.run(**ae_lcdm_kw, **cls_kw)['cls']

    print('Running AxiCAMB...')
    ax = axicamb_runner.run(**ax_kw, **cls_kw)['cls']

    print('Running AxiECAMB...')
    ae = axiecamb_runner.run(**ae_kw, **cls_kw)['cls']

    # Common ell range
    lmin, lmax_plot = 2, min(ax['ell'][-1], ae['ell'][-1], args.lmax)
    ell_ax = ax['ell']
    ell_ae = ae['ell']

    fig, axes = plt.subplots(3, 1, figsize=(10, 11),
                              gridspec_kw={'height_ratios': [2, 1, 1]})

    from scipy.interpolate import interp1d
    f_ax_tt = interp1d(ell_ax, ax['tt'], bounds_error=False, fill_value=np.nan)
    f_ae_tt = interp1d(ell_ae, ae['tt'], bounds_error=False, fill_value=np.nan)
    f_lcdm_tt = interp1d(lcdm['ell'], lcdm['tt'], bounds_error=False, fill_value=np.nan)
    f_lcdm_ae_tt = interp1d(lcdm_ae['ell'], lcdm_ae['tt'], bounds_error=False, fill_value=np.nan)
    ell_common = np.arange(lmin, lmax_plot + 1)

    # Top: TT spectrum
    a = axes[0]
    a.plot(lcdm['ell'], lcdm['tt'], 'k-', lw=1, alpha=0.5, label=r'$\Lambda$CDM')
    a.plot(ell_ax, ax['tt'], 'r-', lw=1.5, label='AxiCAMB')
    a.plot(ell_ae, ae['tt'], 'g--', lw=1.5, label='AxiECAMB')
    a.set_ylabel(r'$\ell(\ell+1)C_\ell^{TT}/2\pi$ [$\mu K^2$]')
    a.set_xlim(lmin, lmax_plot)
    a.legend()
    a.grid(alpha=0.3)
    a.set_title(f'{lens_label.capitalize()} TT (no nonlinear), m={axion["m_ax"]:.0e} eV, f={axion["f_ax"]}')

    # Middle: direct ratio (%)
    a2 = axes[1]
    ratio = f_ax_tt(ell_common) / f_ae_tt(ell_common)
    valid = np.isfinite(ratio)
    a2.plot(ell_common[valid], (ratio[valid] - 1) * 100, 'k-', lw=1)
    a2.axhline(0, color='gray', ls=':', alpha=0.5)
    a2.axhspan(-0.1, 0.1, color='gray', alpha=0.1)
    a2.set_ylabel('AxiCAMB / AxiECAMB - 1 [%]')
    a2.set_xlim(lmin, lmax_plot)
    a2.set_ylim(-0.5, 0.5)
    a2.grid(alpha=0.3)

    # Bottom: ratio to respective LCDM (%)
    a3 = axes[2]
    ratio_ax_lcdm = f_ax_tt(ell_common) / f_lcdm_tt(ell_common)
    ratio_ae_lcdm = f_ae_tt(ell_common) / f_lcdm_ae_tt(ell_common)
    v1 = np.isfinite(ratio_ax_lcdm)
    v2 = np.isfinite(ratio_ae_lcdm)
    a3.plot(ell_common[v1], (ratio_ax_lcdm[v1] - 1) * 100, 'r-', lw=1, label='AxiCAMB / AxiCAMB LCDM')
    a3.plot(ell_common[v2], (ratio_ae_lcdm[v2] - 1) * 100, 'g--', lw=1, label='AxiECAMB / AxiECAMB LCDM')
    a3.axhline(0, color='gray', ls=':', alpha=0.5)
    a3.set_xlabel(r'$\ell$')
    a3.set_ylabel(r'$C_\ell^{axion} / C_\ell^{\Lambda CDM} - 1$ [%]')
    a3.set_xlim(lmin, lmax_plot)
    a3.legend(fontsize=9)
    a3.grid(alpha=0.3)

    plt.tight_layout()
    tag = f'm{axion["m_ax"]:.0e}_f{axion["f_ax"]}_movH{axion["movH_switch"]:.0f}_{lens_label}'.replace('.', 'p')
    outpath = os.path.join(FIGDIR, f'cls_comparison_{tag}.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'\nSaved {outpath}')

    # Summary
    for L in [100, 500, 1000, 2000]:
        if L <= lmax_plot:
            r = f_ax_tt(L) / f_ae_tt(L)
            print(f'  L={L}: TT ratio = {r:.4f}')


if __name__ == '__main__':
    main()
