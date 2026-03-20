"""
Plot P(k) comparisons: LCDM Halofit vs axionHMcode (basic and DOME).

Shows absolute P(k) and ratios at multiple redshifts.

Requires: AxiCAMB, axionHMcode, matplotlib, scipy

Usage:
    python plot_pk.py [options]
    python plot_pk.py --f_ax 0.3 --m_ax 1e-24 --z 0.0 1.0 2.0
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os, argparse
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import axicamb_runner
import cosmo_params

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

DEFAULT_Z_ARR = np.array([0.0, 0.5, 1.0, 2.0])


def prepare_redshifts(z_arr):
    """Validate plotting redshifts and return unique values for CAMB calls."""
    z_plot = np.asarray(z_arr, dtype=float)
    if z_plot.ndim != 1 or z_plot.size == 0:
        raise ValueError('z_arr must contain at least one redshift')
    if np.any(z_plot < 0):
        raise ValueError('redshifts must be non-negative')
    return z_plot, np.unique(z_plot)


def get_redshift_index(z_values, target, atol=1e-8):
    """Return the exact redshift index and fail if it was not computed."""
    matches = np.where(np.isclose(z_values, target, rtol=0.0, atol=atol))[0]
    if matches.size == 0:
        raise ValueError(f'Redshift z={target} was not computed')
    return int(matches[0])


def format_mass_label(m_ax):
    """Format the axion mass for the plot title."""
    if m_ax <= 0:
        raise ValueError('m_ax must be positive')
    exp = int(np.floor(np.log10(m_ax)))
    mantissa = m_ax / 10**exp
    if np.isclose(mantissa, 1.0):
        return rf'10^{{{exp}}}'
    return rf'{mantissa:.2g} \times 10^{{{exp}}}'


def format_mass_tag(m_ax):
    """Format the axion mass for a filename."""
    mantissa, exp = f'{m_ax:.3e}'.split('e')
    mantissa = mantissa.rstrip('0').rstrip('.').replace('.', 'p')
    return f'm{mantissa}e{int(exp)}'


def save_pk_data(path, z_arr, lcdm_data, axion_basic, axion_dome, axion_naive,
                 metadata):
    """Save P(k) outputs in a common comparison format."""
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    k_lcdm = lcdm_data['k']
    pk_lin_lcdm = lcdm_data['pk']
    pk_nl_lcdm = lcdm_data['pk_nl']

    k_ax = axion_basic['k']
    pk_lin_ax = axion_basic['pk_total']
    pk_nl_ax_basic = axion_basic['pk_nl']
    pk_cold_ax = axion_basic['pk_cold']
    pk_axion_component = axion_basic['pk_axion']
    pk_nl_ax_dome = axion_dome['pk_nl']

    k_ax_naive = axion_naive['k']
    pk_nl_ax_naive = axion_naive['pk_nl']

    np.savez_compressed(
        path,
        source=np.array('axicamb'),
        z=np.asarray(z_arr, dtype=float),
        k_lin_lcdm=np.asarray(k_lcdm, dtype=float),
        pk_lin_lcdm=np.asarray(pk_lin_lcdm, dtype=float),
        k_nl_lcdm=np.asarray(k_lcdm, dtype=float),
        pk_nl_lcdm=np.asarray(pk_nl_lcdm, dtype=float),
        k_lin_ax=np.asarray(k_ax, dtype=float),
        pk_lin_ax=np.asarray(pk_lin_ax, dtype=float),
        k_lin_cold_ax=np.asarray(k_ax, dtype=float),
        pk_lin_cold_ax=np.asarray(pk_cold_ax, dtype=float),
        k_lin_axion_component=np.asarray(k_ax, dtype=float),
        pk_lin_axion_component=np.asarray(pk_axion_component, dtype=float),
        k_nl_ax_basic=np.asarray(k_ax, dtype=float),
        pk_nl_ax_basic=np.asarray(pk_nl_ax_basic, dtype=float),
        k_nl_ax_dome=np.asarray(k_ax, dtype=float),
        pk_nl_ax_dome=np.asarray(pk_nl_ax_dome, dtype=float),
        k_nl_ax_naive=np.asarray(k_ax_naive, dtype=float),
        pk_nl_ax_naive=np.asarray(pk_nl_ax_naive, dtype=float),
        **metadata,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot P(k) comparisons')
    parser.add_argument('--axionHMcode_path', type=str,
                        default='/Users/adammoss/work/code/axionHMcode',
                        help='Path to axionHMcode')
    cosmo_params.add_cli_args(parser)
    parser.add_argument('--z', type=float, nargs='+',
                        default=DEFAULT_Z_ARR.tolist(),
                        help='Redshifts for plot (default: 0.0 0.5 1.0 2.0)')
    parser.add_argument('--save_data', type=str, default='',
                        help='Optional path to save P(k) data as .npz')
    parser.add_argument('--show_naive', action='store_true',
                        help='Include naive CAMB HMCode-2020 nonlinear for axion')
    parser.add_argument('--debug', action='store_true',
                        help='Print axionHMcode inputs/parameters at each redshift')
    parser.add_argument('--layout', type=str, default='row', choices=['row', 'column'],
                        help='Layout: row (side-by-side) or column (stacked, single-column)')
    args = parser.parse_args()

    cosmo, axion = cosmo_params.from_args(args)
    m_ax = axion['m_ax']
    f_ax = axion['f_ax']
    z_plot, z_compute = prepare_redshifts(args.z)

    # --- Compute ---
    print('Computing LCDM Halofit...')
    lcdm_kw = cosmo_params.get_lcdm_kwargs(cosmo)
    lcdm_data = axicamb_runner.get_lcdm(z_arr=z_compute, nonlinear=True, **lcdm_kw)

    print(f'\nRunning AxiCAMB for f={f_ax}...')
    ax_kw = cosmo_params.get_axicamb_kwargs(cosmo, axion)
    ax_result = axicamb_runner.run(z_arr=z_compute, **ax_kw)

    axion_results = {}
    for dome in [False, True]:
        label = 'DOME' if dome else 'basic'
        print(f'\nComputing axionHMcode f={f_ax} {label}...')
        hmcode = axicamb_runner.get_axionhmcode_pk(
            ax_result, m_ax=m_ax, dome_calibrated=dome,
            axionhmcode_path=args.axionHMcode_path)
        axion_results[dome] = hmcode

    axion_naive = None
    if args.show_naive:
        print(f'\nComputing axion CAMB HMCode-2020 f={f_ax} naive...')
        naive_result = axicamb_runner.run(z_arr=z_compute, nonlinear=True, **ax_kw)
        axion_naive = {'k': naive_result['k'], 'z': naive_result['z'],
                       'pk_nl': naive_result['pk_nl']}

    if args.save_data:
        save_pk_data(
            args.save_data, z_compute, lcdm_data,
            axion_results[False], axion_results[True],
            axion_naive if axion_naive else {'k': k_lcdm, 'z': lcdm_data['z'],
                                              'pk_nl': np.zeros_like(lcdm_data['pk_nl'])},
            metadata={
                'm_ax': np.array(m_ax, dtype=float),
                'f_ax': np.array(f_ax, dtype=float),
                'omega_b': np.array(cosmo.get('ombh2', 0.022383), dtype=float),
                'omega_d': np.array(cosmo.get('omch2_total', 0.12011), dtype=float),
                'h': np.array(cosmo.get('H0', 67.32) / 100, dtype=float),
                'ns': np.array(cosmo.get('ns', 0.96605), dtype=float),
                'As': np.array(cosmo.get('As', 2.10058e-9), dtype=float),
                'tau': np.array(cosmo.get('tau', 0.0543), dtype=float),
            },
        )
        print(f'Saved data {args.save_data}')

    # --- Plot ---
    nz = len(z_plot)
    if args.layout == 'column':
        fig = plt.figure(figsize=(4.5, 3.5 * nz))
        import matplotlib.gridspec as gridspec
        outer = gridspec.GridSpec(nz, 1, hspace=0.3)
        axes = np.empty((2, nz), dtype=object)
        for iz in range(nz):
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[iz], height_ratios=[2, 1], hspace=0)
            axes[0, iz] = fig.add_subplot(inner[0])
            axes[1, iz] = fig.add_subplot(inner[1], sharex=axes[0, iz])
            plt.setp(axes[0, iz].get_xticklabels(), visible=False)
    else:
        fig, axes = plt.subplots(2, nz, figsize=(6 * nz, 8),
                                  gridspec_kw={'height_ratios': [2, 1]},
                                  sharex='col', sharey='row')
        if nz == 1:
            axes = axes.reshape(2, 1)

    z_lcdm = lcdm_data['z']
    k_lcdm = lcdm_data['k']
    k_ax = axion_results[False]['k']
    z_ax = axion_results[False]['z']

    is_column = args.layout == 'column'
    show_legend = True  # only show legend once

    for iz, zi in enumerate(z_plot):
        iz_lcdm = get_redshift_index(z_lcdm, zi)
        iz_ax = get_redshift_index(z_ax, zi)

        # --- Top: absolute P(k) ---
        ax = axes[0, iz]

        # Linear spectra (dashed)
        ax.loglog(k_ax, axion_results[False]['pk_total'][iz_ax], color='C0', ls='--',
                  lw=1.2, alpha=0.7,
                  label=r'Axion linear' if show_legend else None)
        ax.loglog(k_lcdm, lcdm_data['pk'][iz_lcdm], color='k', ls='--',
                  lw=1.2, alpha=0.7,
                  label=r'$\Lambda$CDM linear' if show_legend else None)

        # Non-linear spectra (solid)
        for dome in [False, True]:
            color = 'C3' if dome else 'C0'
            tag = 'DOME' if dome else 'basic'
            iz_hm = get_redshift_index(axion_results[dome]['z'], zi)
            ax.loglog(axion_results[dome]['k'], axion_results[dome]['pk_nl'][iz_hm],
                      color=color, ls='-', lw=1.2,
                      label=f'axionHMcode {tag}' if show_legend else None)

        if axion_naive is not None:
            iz_naive = get_redshift_index(axion_naive['z'], zi)
            ax.loglog(axion_naive['k'], axion_naive['pk_nl'][iz_naive], color='C2',
                      ls='-', lw=1.2,
                      label=r'Naive HMCode' if show_legend else None)

        ax.loglog(k_lcdm, lcdm_data['pk_nl'][iz_lcdm], color='k', ls='-', lw=1.2,
                  label=r'$\Lambda$CDM HMCode' if show_legend else None)

        ax.text(0.95, 0.95, f'$z = {zi:.0f}$', transform=ax.transAxes,
                fontsize=9, ha='right', va='top')
        ax.grid(True, alpha=0.2)
        ax.set_xlim(1e-2, 50)
        ax.set_ylabel(r'$P(k)$ [$(h^{-1}\,\mathrm{Mpc})^3$]')
        if show_legend:
            ax.legend(fontsize=6.5, loc='lower left', framealpha=0.8)

        # --- Bottom: ratio to NL LCDM ---
        ax2 = axes[1, iz]
        interp_nl_lcdm = interp1d(k_lcdm, lcdm_data['pk_nl'][iz_lcdm],
                                   bounds_error=False, fill_value=np.nan)

        for dome in [False, True]:
            color = 'C3' if dome else 'C0'
            tag = 'DOME' if dome else 'basic'
            iz_hm = get_redshift_index(axion_results[dome]['z'], zi)
            k_hm = axion_results[dome]['k']
            pk_nl_lcdm_interp = interp_nl_lcdm(k_hm)
            valid = np.isfinite(pk_nl_lcdm_interp) & (pk_nl_lcdm_interp > 0)
            ratio_nl = axion_results[dome]['pk_nl'][iz_hm, valid] / pk_nl_lcdm_interp[valid]
            ax2.semilogx(k_hm[valid], ratio_nl, color=color, ls='-', lw=1.2,
                         label=f'axionHMcode {tag}' if show_legend else None)

        if axion_naive is not None:
            iz_naive = get_redshift_index(axion_naive['z'], zi)
            pk_nl_lcdm_interp = interp_nl_lcdm(axion_naive['k'])
            valid = np.isfinite(pk_nl_lcdm_interp) & (pk_nl_lcdm_interp > 0)
            ratio_nl = axion_naive['pk_nl'][iz_naive, valid] / pk_nl_lcdm_interp[valid]
            ax2.semilogx(axion_naive['k'][valid], ratio_nl, color='C2', ls='-', lw=1.2,
                         label=r'Naive HMCode' if show_legend else None)

        ax2.axhline(1.0, color='k', ls=':', alpha=0.5, lw=0.8)
        if is_column and iz < nz - 1:
            ax2.set_xlabel('')
        else:
            ax2.set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]')
        ax2.grid(True, alpha=0.2)
        ax2.set_ylim(0.3, 1.8)
        ax2.set_xlim(1e-2, 50)
        ax2.set_ylabel(r'$P_\mathrm{NL}^\mathrm{axion} / '
                       r'P_\mathrm{NL}^{\Lambda\mathrm{CDM}}$')

        show_legend = False  # only on first panel

    mass_label = format_mass_label(m_ax)
    if not is_column:
        fig.suptitle(f'$m_a = {mass_label}$ eV, $f_\\mathrm{{ax}} = {f_ax}$',
                     fontsize=10)
    plt.tight_layout()
    tag_file = f'pk_{format_mass_tag(m_ax)}_f{f_ax}'.replace('.', 'p')
    plt.savefig(os.path.join(FIGDIR, f'{tag_file}.pdf'), dpi=150,
                bbox_inches='tight')
    print(f'\nSaved figures/{tag_file}.pdf')
    plt.close()
