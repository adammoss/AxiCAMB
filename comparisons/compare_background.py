"""
Compare axion background evolution between AxiCAMB and AxiECAMB.

Plots comoving density rho*a^3 and ratio.

Usage:
    python compare_background.py --f_ax 0.3 --m_ax 1e-24
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import argparse
import os

import camb
from camb.axion_utils import get_axion_phi_i
import axiecamb_runner
import cosmo_params

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def get_axicamb_background(cosmo, axion):
    """Get axion background from AxiCAMB."""
    ax_kw = cosmo_params.get_axicamb_kwargs(cosmo, axion)

    params = get_axion_phi_i(
        h=ax_kw['H0'] / 100, ombh2=ax_kw['ombh2'],
        omch2_total=ax_kw['omch2_total'],
        f_ax=ax_kw['f_ax'], mass_ev=ax_kw['m_ax'], verbose=False,
        use_PH=ax_kw['use_PH'], mH=ax_kw['mH'],
        mnu=ax_kw['mnu'])

    omch2_cdm = max((1 - ax_kw['f_ax']) * ax_kw['omch2_total'], 1e-7)
    pars = camb.set_params(
        H0=ax_kw['H0'], ombh2=ax_kw['ombh2'], omch2=omch2_cdm,
        omk=0, tau=ax_kw['tau'], As=ax_kw['As'], ns=ax_kw['ns'],
        mnu=ax_kw['mnu'],
        dark_energy_model='EarlyQuintessence',
        m=params['m'], theta_i=params['theta_i'],
        frac_lambda0=params['frac_lambda0'],
        use_zc=False, use_fluid_approximation=True,
        potential_type=1, weighting_factor=10.0,
        oscillation_threshold=1, use_PH=ax_kw['use_PH'], mH=ax_kw['mH'])
    pars.set_matter_power(redshifts=[0.0], kmax=ax_kw['kmax'])
    results = camb.get_results(pars)

    a_arr = np.logspace(-7, 0, 2000)
    rho_de, w_de = results.get_dark_energy_rho_w(a_arr)
    rho_cc = params['frac_lambda0'] * rho_de[-1]
    rho_ax = rho_de - rho_cc

    return {
        'a': a_arr,
        'rho_ax': rho_ax,
        'w_de': w_de,
        'frac_lambda0': params['frac_lambda0'],
        'tau0': results.conformal_time(0),
    }


def get_axiecamb_background(cosmo, axion):
    """Get axion background from AxiECAMB.

    Requires the background dump code in axion_background.F90.
    """
    axiecamb_dir = axiecamb_runner.AXIECAMB_DIR
    bg_file = os.path.join(axiecamb_dir, 'axion_background.txt')

    ae_kw = cosmo_params.get_axiecamb_kwargs(cosmo, axion)
    result = axiecamb_runner.run(z_arr=[0.0], verbose=False, **ae_kw)

    if not os.path.exists(bg_file):
        print('Warning: axion_background.txt not found in AxiECAMB dir')
        return None

    bg = np.loadtxt(bg_file, skiprows=1)
    a = 10**bg[:, 0]
    rho = 10**bg[:, 1]

    # Extend with a^-3 after table ends
    a_osc = a[-1]
    a_ext = np.logspace(np.log10(a_osc), 0, 500)
    a_full = np.concatenate([a, a_ext[1:]])
    rho_full = np.concatenate([rho, rho[-1] * (a_osc / a_ext[1:])**3])

    return {
        'a': a_full,
        'rho': rho_full,
        'a_osc': a_osc,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare axion background evolution')
    cosmo_params.add_cli_args(parser)
    args = parser.parse_args()

    cosmo, axion = cosmo_params.from_args(args)

    print(f'Parameters: m_ax={axion["m_ax"]:.0e}, f_ax={axion["f_ax"]}, '
          f'movH={axion["movH_switch"]}')

    print('\nAxiCAMB background...')
    ax_bg = get_axicamb_background(cosmo, axion)

    print('AxiECAMB background...')
    ae_bg = get_axiecamb_background(cosmo, axion)

    if ae_bg is None:
        print('Cannot compare backgrounds without axion_background.txt')
        return

    # Comoving density
    rho_ax_a3 = ax_bg['rho_ax'] * ax_bg['a']**3
    rho_ax_a3_norm = rho_ax_a3 / rho_ax_a3[-1]

    rho_ae_a3 = ae_bg['rho'] * ae_bg['a']**3
    rho_ae_a3_norm = rho_ae_a3 / rho_ae_a3[-1]

    f_ae = interp1d(np.log10(ae_bg['a']), rho_ae_a3_norm,
                     bounds_error=False, fill_value=np.nan)
    f_ax_interp = interp1d(np.log10(ax_bg['a']), rho_ax_a3_norm,
                            bounds_error=False, fill_value=np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(ae_bg['a'], rho_ae_a3_norm, 'g-', lw=2, label='AxiECAMB')
    ax.plot(ax_bg['a'], rho_ax_a3_norm, 'r--', lw=1.5, label='AxiCAMB')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('a')
    ax.set_ylabel(r'$\rho_{ax} a^3 / (\rho_{ax} a^3)_{z=0}$')
    ax.set_title('Comoving axion density')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(1e-7, 1)

    ax = axes[1]
    a_plot = np.logspace(-6.5, -0.5, 500)
    v_ae = f_ae(np.log10(a_plot))
    v_ax = f_ax_interp(np.log10(a_plot))
    mask = np.isfinite(v_ae) & np.isfinite(v_ax) & (v_ae > 0)
    ax.plot(a_plot[mask], v_ax[mask] / v_ae[mask], 'k-', lw=1.5)
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('a')
    ax.set_ylabel('AxiCAMB / AxiECAMB')
    ax.set_title(r'$\rho_{ax} a^3$ ratio (norm to $z=0$)')
    ax.grid(alpha=0.3)
    ax.set_xlim(3e-7, 0.1)
    ax.set_ylim(0.95, 1.05)

    fig.suptitle(
        f'Background: m={axion["m_ax"]:.0e} eV, f={axion["f_ax"]}, '
        f'movH={axion["movH_switch"]:.0f}',
        fontsize=12)
    plt.tight_layout()

    tag = f'm{axion["m_ax"]:.0e}_f{axion["f_ax"]}_movH{axion["movH_switch"]:.0f}'.replace('.', 'p')
    outpath = os.path.join(FIGDIR, f'background_{tag}.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'\nSaved {outpath}')


if __name__ == '__main__':
    main()
