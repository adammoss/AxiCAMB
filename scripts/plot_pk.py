"""
Plot P(k) comparisons: LCDM Halofit vs axionHMcode (basic and DOME).

Shows absolute P(k) and ratios at multiple redshifts.

Requires: AxiCAMB, axionHMcode, matplotlib, scipy

Usage:
    python plot_pk.py [options]
    python plot_pk.py --f_ax 0.3 --m_ax 1e-24 --z 0.0 1.0 2.0
    python plot_pk.py --m_ax 1e-26 1e-25 1e-24 --z 0.0 1.0 2.0
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


def compute_mass(m_ax, f_ax, cosmo, axion_base, z_compute, args):
    """Compute all spectra for a single axion mass. Returns dict of results."""
    axion = {**axion_base, 'm_ax': m_ax}
    ax_kw = cosmo_params.get_axicamb_kwargs(cosmo, axion)

    print(f'\nRunning AxiCAMB for m_ax={m_ax:.2e}, f={f_ax}...')
    ax_result = axicamb_runner.run(z_arr=z_compute, **ax_kw)

    axion_results = {}
    for dome in [False, True]:
        label = 'DOME' if dome else 'basic'
        print(f'  axionHMcode {label}...')
        hmcode = axicamb_runner.get_axionhmcode_pk(
            ax_result, m_ax=m_ax, dome_calibrated=dome,
            axionhmcode_path=args.axionHMcode_path)
        axion_results[dome] = hmcode

    axion_naive = None
    if args.show_naive:
        print(f'  Naive HMCode-2020...')
        naive_result = axicamb_runner.run(z_arr=z_compute, nonlinear=True, **ax_kw)
        axion_naive = {'k': naive_result['k'], 'z': naive_result['z'],
                       'pk_nl': naive_result['pk_nl']}

    return {'axion_results': axion_results, 'axion_naive': axion_naive}


def plot_panel(ax_top, ax_bot, zi, lcdm_data, mass_data, show_legend=False,
               show_ylabel=True, show_xlabel=True, label_text=None):
    """Plot a single (P(k), ratio) panel pair for one redshift and one mass."""
    z_lcdm = lcdm_data['z']
    k_lcdm = lcdm_data['k']
    axion_results = mass_data['axion_results']
    axion_naive = mass_data['axion_naive']
    k_ax = axion_results[False]['k']
    z_ax = axion_results[False]['z']

    iz_lcdm = get_redshift_index(z_lcdm, zi)
    iz_ax = get_redshift_index(z_ax, zi)

    # --- Top: absolute P(k) ---
    ax_top.loglog(k_ax, axion_results[False]['pk_total'][iz_ax], color='C0',
                  ls='--', lw=1.2, alpha=0.7,
                  label=r'Axion linear' if show_legend else None)
    ax_top.loglog(k_lcdm, lcdm_data['pk'][iz_lcdm], color='k', ls='--',
                  lw=1.2, alpha=0.7,
                  label=r'$\Lambda$CDM linear' if show_legend else None)

    for dome in [False, True]:
        color = 'C3' if dome else 'C0'
        tag = 'DOME' if dome else 'basic'
        iz_hm = get_redshift_index(axion_results[dome]['z'], zi)
        ax_top.loglog(axion_results[dome]['k'],
                      axion_results[dome]['pk_nl'][iz_hm],
                      color=color, ls='-', lw=1.2,
                      label=f'axionHMcode {tag}' if show_legend else None)

    if axion_naive is not None:
        iz_naive = get_redshift_index(axion_naive['z'], zi)
        ax_top.loglog(axion_naive['k'], axion_naive['pk_nl'][iz_naive],
                      color='C2', ls='-', lw=1.2,
                      label=r'Naive HMCode' if show_legend else None)

    ax_top.loglog(k_lcdm, lcdm_data['pk_nl'][iz_lcdm], color='k', ls='-',
                  lw=1.2,
                  label=r'$\Lambda$CDM HMCode' if show_legend else None)

    if label_text:
        ax_top.text(0.95, 0.95, label_text, transform=ax_top.transAxes,
                    fontsize=9, ha='right', va='top')
    ax_top.grid(True, alpha=0.2)
    ax_top.set_xlim(1e-2, 50)
    if show_ylabel:
        ax_top.set_ylabel(r'$P(k)$ [$(h^{-1}\,\mathrm{Mpc})^3$]')
    if show_legend:
        ax_top.legend(fontsize=6.5, loc='lower left', framealpha=0.8)

    # --- Bottom: ratio to NL LCDM ---
    interp_nl_lcdm = interp1d(k_lcdm, lcdm_data['pk_nl'][iz_lcdm],
                               bounds_error=False, fill_value=np.nan)

    for dome in [False, True]:
        color = 'C3' if dome else 'C0'
        tag = 'DOME' if dome else 'basic'
        iz_hm = get_redshift_index(axion_results[dome]['z'], zi)
        k_hm = axion_results[dome]['k']
        pk_nl_lcdm_interp = interp_nl_lcdm(k_hm)
        valid = np.isfinite(pk_nl_lcdm_interp) & (pk_nl_lcdm_interp > 0)
        ratio_nl = (axion_results[dome]['pk_nl'][iz_hm, valid]
                    / pk_nl_lcdm_interp[valid])
        ax_bot.semilogx(k_hm[valid], ratio_nl, color=color, ls='-', lw=1.2,
                        label=f'axionHMcode {tag}' if show_legend else None)

    if axion_naive is not None:
        iz_naive = get_redshift_index(axion_naive['z'], zi)
        pk_nl_lcdm_interp = interp_nl_lcdm(axion_naive['k'])
        valid = np.isfinite(pk_nl_lcdm_interp) & (pk_nl_lcdm_interp > 0)
        ratio_nl = (axion_naive['pk_nl'][iz_naive, valid]
                    / pk_nl_lcdm_interp[valid])
        ax_bot.semilogx(axion_naive['k'][valid], ratio_nl, color='C2', ls='-',
                        lw=1.2,
                        label=r'Naive HMCode' if show_legend else None)

    ax_bot.axhline(1.0, color='k', ls=':', alpha=0.5, lw=0.8)
    if show_xlabel:
        ax_bot.set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]')
    ax_bot.grid(True, alpha=0.2)
    ax_bot.set_ylim(0.3, 1.8)
    ax_bot.set_xlim(1e-2, 50)
    if show_ylabel:
        ax_bot.set_ylabel(r'$P_\mathrm{NL}^\mathrm{axion} / '
                          r'P_\mathrm{NL}^{\Lambda\mathrm{CDM}}$')


if __name__ == '__main__':
    import matplotlib.gridspec as gridspec

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
    m_ax_list = axion['m_ax_list']
    f_ax = axion['f_ax']
    z_plot, z_compute = prepare_redshifts(args.z)
    nz = len(z_plot)
    nm = len(m_ax_list)

    # --- Compute LCDM (shared across masses) ---
    print('Computing LCDM Halofit...')
    lcdm_kw = cosmo_params.get_lcdm_kwargs(cosmo)
    lcdm_data = axicamb_runner.get_lcdm(z_arr=z_compute, nonlinear=True, **lcdm_kw)

    # --- Compute axion spectra per mass ---
    all_mass_data = {}
    for m_ax in m_ax_list:
        all_mass_data[m_ax] = compute_mass(
            m_ax, f_ax, cosmo, axion, z_compute, args)

    # --- Save data (first mass only, for backward compat) ---
    if args.save_data:
        m0 = m_ax_list[0]
        md = all_mass_data[m0]
        k_lcdm = lcdm_data['k']
        save_pk_data(
            args.save_data, z_compute, lcdm_data,
            md['axion_results'][False], md['axion_results'][True],
            md['axion_naive'] if md['axion_naive'] else {
                'k': k_lcdm, 'z': lcdm_data['z'],
                'pk_nl': np.zeros_like(lcdm_data['pk_nl'])},
            metadata={
                'm_ax': np.array(m0, dtype=float),
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
    if nm > 1:
        # Multi-mass grid: rows = redshifts, columns = masses
        fig = plt.figure(figsize=(4.5 * nm, 3.5 * nz))
        outer = gridspec.GridSpec(nz, nm, hspace=0.3, wspace=0.05)
        # Track axes for sharing
        top_axes = {}
        bot_axes = {}
        for iz, zi in enumerate(z_plot):
            for im, m_ax in enumerate(m_ax_list):
                inner = gridspec.GridSpecFromSubplotSpec(
                    2, 1, subplot_spec=outer[iz, im],
                    height_ratios=[2, 1], hspace=0)
                # Share y axes across columns
                share_top = top_axes.get(iz)
                share_bot = bot_axes.get(iz)
                ax_top = fig.add_subplot(inner[0], sharey=share_top)
                ax_bot = fig.add_subplot(inner[1], sharex=ax_top, sharey=share_bot)
                plt.setp(ax_top.get_xticklabels(), visible=False)
                if im > 0:
                    plt.setp(ax_top.get_yticklabels(), visible=False)
                    plt.setp(ax_bot.get_yticklabels(), visible=False)
                if im == 0:
                    top_axes[iz] = ax_top
                    bot_axes[iz] = ax_bot

                label = f'$z = {zi:.0f}$'
                if iz == 0:
                    mass_label = format_mass_label(m_ax)
                    label = (f'$m_a = {mass_label}$ eV\n' + label)

                plot_panel(
                    ax_top, ax_bot, zi, lcdm_data, all_mass_data[m_ax],
                    show_legend=(iz == 0 and im == 0),
                    show_ylabel=(im == 0),
                    show_xlabel=(iz == nz - 1),
                    label_text=label,
                )

        plt.tight_layout()
        mass_tags = '_'.join(format_mass_tag(m) for m in m_ax_list)
        tag_file = f'pk_{mass_tags}_f{f_ax}'.replace('.', 'p')

    elif args.layout == 'column':
        # Single mass, column layout
        fig = plt.figure(figsize=(4.5, 3.5 * nz))
        outer = gridspec.GridSpec(nz, 1, hspace=0.3)
        m_ax = m_ax_list[0]
        for iz, zi in enumerate(z_plot):
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[iz], height_ratios=[2, 1], hspace=0)
            ax_top = fig.add_subplot(inner[0])
            ax_bot = fig.add_subplot(inner[1], sharex=ax_top)
            plt.setp(ax_top.get_xticklabels(), visible=False)

            plot_panel(
                ax_top, ax_bot, zi, lcdm_data, all_mass_data[m_ax],
                show_legend=(iz == 0),
                show_ylabel=True,
                show_xlabel=(iz == nz - 1),
                label_text=f'$z = {zi:.0f}$',
            )

        plt.tight_layout()
        tag_file = f'pk_{format_mass_tag(m_ax)}_f{f_ax}'.replace('.', 'p')

    else:
        # Single mass, row layout
        fig = plt.figure(figsize=(6 * nz, 8))
        outer = gridspec.GridSpec(1, nz, wspace=0.25)
        m_ax = m_ax_list[0]
        for iz, zi in enumerate(z_plot):
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[0, iz],
                height_ratios=[2, 1], hspace=0)
            ax_top = fig.add_subplot(inner[0])
            ax_bot = fig.add_subplot(inner[1], sharex=ax_top)
            plt.setp(ax_top.get_xticklabels(), visible=False)

            plot_panel(
                ax_top, ax_bot, zi, lcdm_data, all_mass_data[m_ax],
                show_legend=(iz == 0),
                show_ylabel=(iz == 0),
                show_xlabel=True,
                label_text=f'$z = {zi:.0f}$',
            )

        mass_label = format_mass_label(m_ax)
        fig.suptitle(f'$m_a = {mass_label}$ eV, $f_\\mathrm{{ax}} = {f_ax}$',
                     fontsize=10)
        plt.tight_layout()
        tag_file = f'pk_{format_mass_tag(m_ax)}_f{f_ax}'.replace('.', 'p')

    plt.savefig(os.path.join(FIGDIR, f'{tag_file}.pdf'), dpi=150,
                bbox_inches='tight')
    print(f'\nSaved figures/{tag_file}.pdf')
    plt.close()
