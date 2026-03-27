"""
Evaluate the likelihood at specific parameter values using Cobaya.

Generate:
    python eval_likelihood.py --m_range -25 -23 5 --f_range 0.01 0.5 5 --output output/chi2_naive.npz
    python eval_likelihood.py --m_range -25 -23 5 --f_range 0.01 0.5 5 --nl_model dome --output output/chi2_dome.npz

Plot saved data:
    python eval_likelihood.py --plot output/chi2_naive.npz
    python eval_likelihood.py --plot output/chi2_naive.npz output/chi2_dome.npz

Single point:
    python eval_likelihood.py --m_axion -24 --f_axion 0.3
"""
import numpy as np
import argparse
import os
import sys

from cobaya.model import get_model


def build_info(m_axion=-24, f_axion=0.3, use_lensing=True,
               bestfit=None, nl_model='naive'):
    """Build Cobaya info dict for likelihood evaluation."""

    # Default best-fit point (from MCMC)
    if bestfit is None:
        # LCDM Planck + ACT best fit
        bestfit = {
            'ombh2': 0.0224823,
            'omch2': 0.119493,
            'H0': 67.5789,
            'logA': 3.04888,
            'ns': 0.970774,
            'tau': 0.0566292,
            'A_act': 1.0009,
            'P_act': 1.00269,
        }

    theory_extra = {
        'kmax': 10,
        'k_per_logint': 130,
        'NonLinear': 'NonLinear_lens',
        'nonlinear': True,
        'lens_potential_accuracy': 8,
        'lens_margin': 2050,
        'lAccuracyBoost': 1.2,
        'min_l_logl_sampling': 6000,
        'DoLateRadTruncation': False,
        'AccuracyBoost': 2.5,
    }

    if nl_model == 'basic':
        theory_extra['axion_nonlinear_model'] = 'axionHMcode'
        theory_extra['axionHMcode_path'] = '/Users/adammoss/work/code/axionHMcode'
    elif nl_model == 'dome':
        theory_extra['axion_nonlinear_model'] = 'axionHMcode_dome'
        theory_extra['axionHMcode_path'] = '/Users/adammoss/work/code/axionHMcode'

    likelihood = {
        'planck_2018_lowl.TT': None,
        'planck_2018_lowl.EE': None,
        'act_dr6_cmbonly.PlanckActCut': {
            'dataset_params': {
                'use_cl': 'tt te ee',
                'lmin_cuts': '0 0 0',
                'lmax_cuts': '1000 600 600',
            },
            'params': {
                'A_planck': {
                    'value': 'lambda A_act: A_act',
                    'latex': r'A_{\rm Planck}',
                },
            },
        },
        'act_dr6_cmbonly': {
            'stop_at_error': True,
        },
    }

    if use_lensing:
        likelihood['act_dr6_lenslike.ACTDR6LensLike'] = {
            'lens_only': False,
            'stop_at_error': True,
            'lmax': 4000,
            'variant': 'act_baseline',
        }

    info = {
        'theory': {
            'camb': {
                'path': '/Users/adammoss/work/code/AxiCAMB',
                'stop_at_error': False,
                'extra_args': theory_extra,
                'params': {
                    'thetastar': {'latex': r'\theta_\star', 'derived': True},
                    'sigma8': {'latex': r'\sigma_8', 'derived': True},
                    'YHe': {'latex': r'Y_\mathrm{He}', 'derived': True},
                    'zrei': {'latex': r'z_\mathrm{reio}', 'derived': True},
                    'taurend': {'latex': r'\tau_\mathrm{rec}', 'derived': True},
                    'zstar': {'latex': r'z_\star', 'derived': True},
                    'rstar': {'latex': r'r_{s,\star}', 'derived': True},
                    'zdrag': {'latex': r'z_d', 'derived': True},
                },
            },
        },
        'likelihood': likelihood,
        'prior': {
            'cal_dip_prior': 'lambda A_act: stats.norm.logpdf(A_act, loc=1.0, scale=0.003)',
        },
        'params': {
            'ombh2': bestfit['ombh2'],
            'omch2': bestfit['omch2'],
            'H0': bestfit['H0'],
            'logA': {
                'value': bestfit['logA'],
                'drop': True,
            },
            'As': {
                'value': 'lambda logA: 1e-10*np.exp(logA)',
                'latex': r'A_\mathrm{s}',
                'derived': True,
            },
            'ns': bestfit['ns'],
            'tau': bestfit['tau'],
            'm_axion': m_axion,
            'f_axion': f_axion,
            'A_act': bestfit['A_act'],
            'P_act': bestfit['P_act'],
            'age': {'latex': r'{\rm{Age}}/\mathrm{Gyr}'},
        },
        'sampler': {
            'evaluate': {},
        },
    }

    return info


def eval_point(m_axion, f_axion, use_lensing=True, bestfit=None, nl_model='naive'):
    """Evaluate likelihood at a single (m_axion, f_axion) point."""
    info = build_info(m_axion=m_axion, f_axion=f_axion,
                      use_lensing=use_lensing, bestfit=bestfit, nl_model=nl_model)
    model = get_model(info)
    point = model.parameterization.sampled_params()
    result = model.loglikes(point, return_derived=False)

    # model.loglikes returns an array of loglike values
    # model.likelihood names gives the keys
    names = list(model.likelihood.keys())
    loglikes = {name: float(result[i]) for i, name in enumerate(names)}
    total = sum(loglikes.values())

    return loglikes, total


FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def plot_grid(files, exclude_lensing=False):
    """Plot Delta chi^2 colour maps from saved .npz files."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 14})
    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for i, fname in enumerate(files):
        ax = axes[0, i]
        d = np.load(fname, allow_pickle=True)
        m_arr = d['m_arr']
        f_arr = d['f_arr']
        chi2 = d['chi2_total'].copy()
        nl_model = str(d.get('nl_model', 'unknown'))
        chi2_lcdm = float(d.get('chi2_lcdm', np.nanmin(chi2)))

        # Subtract lensing component if requested
        title_suffix = ''
        if exclude_lensing:
            for key in d.files:
                if 'LensLike' in key or 'lenslike' in key:
                    chi2 -= d[key]
                    # Also subtract from LCDM reference
                    # (LCDM lensing chi2 is the same across the grid at f~0)
                    chi2_lcdm -= float(d[key][0, 0]) if d[key].ndim == 2 else 0
                    break
            title_suffix = ' (no lens. like.)'

        dchi2 = chi2 - chi2_lcdm
        imin = np.unravel_index(np.nanargmin(dchi2), dchi2.shape)
        imax = np.unravel_index(np.nanargmax(dchi2), dchi2.shape)
        print(f'{fname}: Delta chi2 min = {np.nanmin(dchi2):.2f} at log10(m)={m_arr[imin[0]]:.2f}, f={f_arr[imin[1]]:.4f}')
        print(f'{fname}: Delta chi2 max = {np.nanmax(dchi2):.2f} at log10(m)={m_arr[imax[0]]:.2f}, f={f_arr[imax[1]]:.4f}')

        im = ax.pcolormesh(m_arr, f_arr, dchi2.T,
                           shading='nearest', cmap='RdBu_r',
                           vmin=-10, vmax=10)
        ax.set_xlim(m_arr.min(), m_arr.max())
        ax.set_ylim(f_arr.min(), f_arr.max())
        ax.set_xlabel(r'$\log_{10}(m / \mathrm{eV})$')
        if i == 0:
            ax.set_ylabel(r'$f_{\rm ax}$')
        ax.set_title(nl_model + title_suffix)

        # Contours
        try:
            cs = ax.contour(m_arr, f_arr, dchi2.T,
                            levels=[-9, -4, -1, 1, 4, 9],
                            colors='k', linewidths=0.8, linestyles='--')
            ax.clabel(cs, fmt='%.0f', fontsize=10)
        except Exception:
            pass

        plt.colorbar(im, ax=ax, label=r'$\Delta\chi^2$')

    plt.tight_layout()
    tag = 'nolenslike_' if exclude_lensing else ''
    basename = os.path.splitext(os.path.basename(files[0]))[0]
    outpath = os.path.join(FIGDIR, f'{tag}{basename}_dchi2.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'Saved {outpath}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate likelihood at specific points')
    parser.add_argument('--m_axion', type=float, nargs='+', default=None,
                        help='log10(m_ax/eV) values')
    parser.add_argument('--f_axion', type=float, nargs='+', default=None,
                        help='f_ax values')
    parser.add_argument('--m_range', type=float, nargs=3, default=None,
                        metavar=('MIN', 'MAX', 'N'),
                        help='log10(m) grid: min max num_points (e.g. -25 -23 5)')
    parser.add_argument('--f_range', type=float, nargs=3, default=None,
                        metavar=('MIN', 'MAX', 'N'),
                        help='f_ax grid: min max num_points (e.g. 0.01 0.5 5)')
    parser.add_argument('--no_lensing', action='store_true',
                        help='Exclude ACT lensing likelihood')
    parser.add_argument('--nl_model', type=str, default='naive',
                        choices=['naive', 'basic', 'dome'],
                        help='Nonlinear model: naive (HMCode-2020), basic (axionHMcode), dome (axionHMcode DOME)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to .npz file')
    parser.add_argument('--plot', type=str, nargs='+', default=None,
                        help='Plot Delta chi2 from saved .npz files (no evaluation)')
    args = parser.parse_args()

    if args.plot:
        plot_grid(args.plot, exclude_lensing=args.no_lensing)
        return

    # Build mass and fraction arrays
    if args.m_range is not None:
        m_arr = np.linspace(args.m_range[0], args.m_range[1], int(args.m_range[2]))
    elif args.m_axion is not None:
        m_arr = args.m_axion
    else:
        parser.error('Specify --m_axion or --m_range')

    if args.f_range is not None:
        f_arr = np.linspace(args.f_range[0], args.f_range[1], int(args.f_range[2]))
    elif args.f_axion is not None:
        f_arr = args.f_axion
    else:
        parser.error('Specify --f_axion or --f_range')

    use_lensing = not args.no_lensing

    import logging
    logging.getLogger().setLevel(logging.ERROR)

    m_arr = np.atleast_1d(m_arr)
    f_arr = np.atleast_1d(f_arr)
    chi2_grid = np.full((len(m_arr), len(f_arr)), np.nan)
    chi2_components = {}
    chi2_lcdm = np.nan

    # Load cache from existing output file
    cached = 0
    if args.output and os.path.exists(args.output):
        print(f'Loading cache from {args.output}...')
        cache = np.load(args.output, allow_pickle=True)
        cache_m = cache['m_arr']
        cache_f = cache['f_arr']
        chi2_lcdm = float(cache.get('chi2_lcdm', np.nan))
        for im, m in enumerate(m_arr):
            for jf, f in enumerate(f_arr):
                # Find matching indices in cache
                im_c = np.where(np.isclose(cache_m, m))[0]
                jf_c = np.where(np.isclose(cache_f, f))[0]
                if len(im_c) > 0 and len(jf_c) > 0:
                    val = cache['chi2_total'][im_c[0], jf_c[0]]
                    if not np.isnan(val):
                        chi2_grid[im, jf] = val
                        cached += 1
                        for key in cache.files:
                            if key.startswith('chi2_') and key not in ('chi2_total', 'chi2_lcdm'):
                                comp_name = key[5:]  # strip 'chi2_'
                                if comp_name not in chi2_components:
                                    chi2_components[comp_name] = np.full(
                                        (len(m_arr), len(f_arr)), np.nan)
                                chi2_components[comp_name][im, jf] = cache[key][im_c[0], jf_c[0]]
        print(f'Loaded {cached} cached points')

    # LCDM reference chi2 (always naive NL at f~0)
    if np.isnan(chi2_lcdm):
        print('Evaluating LCDM reference (f_axion=0.001, naive NL)...')
        try:
            loglikes_lcdm, total_lcdm = eval_point(
                m_arr[0], 0.001, use_lensing=use_lensing, nl_model='naive')
            chi2_lcdm = -2 * total_lcdm
            print(f'LCDM chi2 = {chi2_lcdm:.2f}')
        except Exception as e:
            print(f'LCDM evaluation failed: {e}')
            chi2_lcdm = np.nan
    else:
        print(f'LCDM chi2 = {chi2_lcdm:.2f} (cached)')

    remaining = np.isnan(chi2_grid).sum()
    print(f'\nGrid: {len(m_arr)} masses x {len(f_arr)} fractions = {len(m_arr)*len(f_arr)} evaluations')
    print(f'Cached: {cached}, remaining: {remaining}')
    print(f'NL model: {args.nl_model}, lensing: {use_lensing}')
    print()
    print(f'{"log10(m)":>10s} {"f_ax":>8s} {"chi2_total":>12s}', end='')
    header_printed = False

    for im, m in enumerate(m_arr):
        for jf, f in enumerate(f_arr):
            if not np.isnan(chi2_grid[im, jf]):
                continue  # already cached

            try:
                loglikes, total = eval_point(m, f, use_lensing=use_lensing,
                                             nl_model=args.nl_model)

                chi2_grid[im, jf] = -2 * total

                if not header_printed:
                    for name in loglikes:
                        short = 'chi2_' + name.split('.')[-1][:15]
                        print(f' {short:>20s}', end='')
                        if name not in chi2_components:
                            chi2_components[name] = np.full((len(m_arr), len(f_arr)), np.nan)
                    print()
                    header_printed = True

                for name, v in loglikes.items():
                    chi2_components[name][im, jf] = -2 * v

                print(f'{m:10.2f} {f:8.3f} {-2*total:12.2f}', end='')
                for v in loglikes.values():
                    print(f' {-2*v:20.2f}', end='')
                print(flush=True)

                # Save after each point so progress is not lost
                if args.output:
                    save_data = {
                        'm_arr': m_arr, 'f_arr': f_arr,
                        'chi2_total': chi2_grid, 'chi2_lcdm': chi2_lcdm,
                        'nl_model': args.nl_model, 'use_lensing': use_lensing,
                    }
                    for name, grid in chi2_components.items():
                        save_data['chi2_' + name] = grid
                    np.savez_compressed(args.output, **save_data)

            except Exception as e:
                print(f'{m:10.2f} {f:8.3f}  ERROR: {e}', flush=True)

    if args.output:
        print(f'\nSaved {args.output}')


if __name__ == '__main__':
    main()
