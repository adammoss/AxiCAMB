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

import camb
from camb import model
from camb.axion_utils import get_axion_phi_i

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


def _format_scalar(value):
    if isinstance(value, (str, np.str_)):
        return value
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f'{float(value):.8g}'


def _print_mapping(title, mapping):
    print(title)
    for key in sorted(mapping):
        value = mapping[key]
        if np.isscalar(value):
            print(f'  {key}: {_format_scalar(value)}')
        else:
            try:
                arr = np.asarray(value)
            except ValueError:
                print(f'  {key}: {value}')
                continue
            if arr.size <= 6:
                print(f'  {key}: {np.array2string(arr, precision=6, separator=", ")}')
            else:
                print(
                    f'  {key}: shape={arr.shape}, min={np.nanmin(arr):.6g}, '
                    f'max={np.nanmax(arr):.6g}'
                )


def _sample_spectrum(k, pk, sample_k=(0.1, 0.5, 1.0, 2.0, 5.0)):
    valid = np.isfinite(k) & np.isfinite(pk) & (k > 0) & (pk > 0)
    if np.count_nonzero(valid) < 2:
        return {}
    values = np.exp(np.interp(np.log(sample_k), np.log(k[valid]), np.log(pk[valid])))
    return {f'k={ks:g}': val for ks, val in zip(sample_k, values)}


def print_hmcode_diagnostics(tag, z, cosmos, hmcode_opts, hmcode_params, axion_params,
                             power_spec):
    print(f'\n[{tag}] z={z:.3f}')
    _print_mapping('cosmos:', cosmos)
    _print_mapping('hmcode_opts:', hmcode_opts)
    _print_mapping('hmcode_params:', hmcode_params)
    _print_mapping('axion_params:', axion_params)
    for spec_key in ['power_total', 'power_cold', 'power_axion']:
        summary = _sample_spectrum(power_spec['k'], power_spec[spec_key])
        print(f'  {spec_key}_samples: ' + ', '.join(
            f'{key}->{value:.6g}' for key, value in summary.items()
        ))


def save_pk_data(path, z_arr, lcdm_data, axion_basic, axion_dome, axion_naive,
                 metadata):
    """Save P(k) outputs in a common comparison format."""
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    k_lcdm, _, pk_lin_lcdm, pk_nl_lcdm = lcdm_data
    k_ax, _, pk_lin_ax, pk_nl_ax_basic, pk_cold_ax, pk_axion_component = axion_basic
    _, _, _, pk_nl_ax_dome, _, _ = axion_dome
    k_ax_naive, _, _, pk_nl_ax_naive = axion_naive

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


def solve_axion_model(f_ax, m_ax, ombh2, omch2, H0, **cosmo_kwargs):
    """Solve the axion background once and reuse it across runs."""
    h = H0 / 100.0
    axion_params = get_axion_phi_i(
        h=h, ombh2=ombh2, omch2_total=omch2,
        f_ax=f_ax, mass_ev=m_ax, verbose=True, accuracy=1, **cosmo_kwargs)
    omch2_cdm = max((1 - f_ax) * omch2, 1e-7)
    return {
        'h': h,
        'axion_params': axion_params,
        'omch2_cdm': omch2_cdm,
    }


def make_axion_camb_pars(axion_setup, ombh2, H0, ns, As, tau, z_arr,
                         nonlinear=False, halofit_version='mead2020',
                         **cosmo_kwargs):
    """Build CAMB parameters for the axion cosmology."""
    axion_params = axion_setup['axion_params']
    pars = camb.set_params(
        H0=H0, ombh2=ombh2, omch2=axion_setup['omch2_cdm'], omk=0, tau=tau,
        As=As, ns=ns, **cosmo_kwargs,
        dark_energy_model='EarlyQuintessence',
        m=axion_params['m'], theta_i=axion_params['theta_i'],
        frac_lambda0=axion_params['frac_lambda0'],
        use_zc=False, use_fluid_approximation=True,
        potential_type=1, weighting_factor=10.0, oscillation_threshold=1,
        use_PH=True, mH=50.0)
    pars.set_for_lmax(2500, lens_potential_accuracy=1)
    pars.DoLateRadTruncation = False
    pars.NonLinear = model.NonLinear_both if nonlinear else model.NonLinear_none
    if nonlinear:
        pars.NonLinearModel.halofit_version = halofit_version
    pars.set_matter_power(redshifts=list(np.asarray(z_arr)[::-1]), kmax=50.0)
    return pars


def get_lcdm_pk(ombh2, omch2, H0, ns, As, tau, z_arr, **cosmo_kwargs):
    """Get LCDM linear and Halofit non-linear P(k)."""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, **cosmo_kwargs)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(2500, lens_potential_accuracy=1)
    pars.NonLinear = model.NonLinear_both
    pars.NonLinearModel.halofit_version = 'mead2020'
    pars.set_matter_power(redshifts=list(np.asarray(z_arr)[::-1]), kmax=50.0)
    results = camb.get_results(pars)

    k_h, z_pk, pk_nl = results.get_nonlinear_matter_power_spectrum(
        hubble_units=True, k_hunit=True)
    _, _, pk_lin = results.get_linear_matter_power_spectrum(
        hubble_units=True, k_hunit=True, nonlinear=False)
    sort_idx = np.argsort(z_pk)
    return k_h, z_pk[sort_idx], pk_lin[sort_idx], pk_nl[sort_idx]


def get_axion_hmcode_pk(axion_setup, m_ax, ombh2, omch2, H0, ns, As, tau,
                         z_arr, dome_calibrated=False, print_hmcode_inputs=False,
                         **cosmo_kwargs):
    """Get axionHMcode P_NL and linear P(k) for given f_axion."""
    from halo_model import HMcode_params, PS_nonlin_axion
    from axion_functions import axion_params as axion_params_module
    from cosmology.overdensities import func_D_z_unnorm_int

    h = axion_setup['h']
    axion_params = axion_setup['axion_params']
    omega_ax_h2 = axion_params['omega_ax_h2']

    # Get linear spectra
    pars = make_axion_camb_pars(
        axion_setup, ombh2, H0, ns, As, tau, z_arr, nonlinear=False,
        **cosmo_kwargs)
    results = camb.get_results(pars)

    k_h, z_pk, pk_total = results.get_linear_matter_power_spectrum(
        hubble_units=True, k_hunit=True, nonlinear=False)
    _, _, pk_cc = results.get_linear_matter_power_spectrum(
        var1='delta_cdm', var2='delta_cdm', hubble_units=True, k_hunit=True)
    _, _, pk_bb = results.get_linear_matter_power_spectrum(
        var1='delta_baryon', var2='delta_baryon', hubble_units=True, k_hunit=True)
    _, _, pk_cb = results.get_linear_matter_power_spectrum(
        var1='delta_cdm', var2='delta_baryon', hubble_units=True, k_hunit=True)
    _, _, pk_axion = results.get_linear_matter_power_spectrum(
        var1='delta_axion', var2='delta_axion', hubble_units=True, k_hunit=True)

    sort_idx = np.argsort(z_pk)
    z_sorted = z_pk[sort_idx]
    pk_total = pk_total[sort_idx]
    pk_cc = pk_cc[sort_idx]
    pk_bb = pk_bb[sort_idx]
    pk_cb = pk_cb[sort_idx]
    pk_axion = pk_axion[sort_idx]
    w_c = axion_setup['omch2_cdm']
    w_b = ombh2
    pk_cold = (
        w_c**2 * pk_cc + w_b**2 * pk_bb + 2 * w_c * w_b * pk_cb
    ) / (w_c + w_b)**2

    # Run axionHMcode
    omega_d = omch2 - omega_ax_h2
    M_arr = np.logspace(7, 18, 100)
    pk_nl = np.zeros_like(pk_total)

    hmcode_opts = {
        'alpha': dome_calibrated, 'eta_given': dome_calibrated,
        'one_halo_damping': True, 'two_halo_damping': dome_calibrated,
        'concentration_param': dome_calibrated, 'full_2h': False,
    }

    for i, zi in enumerate(z_sorted):
        cosmos = {
            'omega_b_0': ombh2, 'omega_d_0': omega_d,
            'omega_ax_0': omega_ax_h2,
            'omega_m_0': ombh2 + omch2,
            'Omega_b_0': ombh2 / h**2, 'Omega_d_0': omega_d / h**2,
            'Omega_ax_0': omega_ax_h2 / h**2,
            'Omega_m_0': (ombh2 + omch2) / h**2,
            'omega_db_0': omega_d + ombh2,
            'Omega_db_0': (omega_d + ombh2) / h**2,
            'h': h, 'ns': ns, 'As': As, 'z': zi, 'm_ax': m_ax,
            'M_min': 7, 'M_max': 18, 'k_piv': 0.05,
            'version': 'dome' if dome_calibrated else 'basic',
        }
        cosmos['Omega_w_0'] = 1 - cosmos['Omega_m_0']
        cosmos['G_a'] = func_D_z_unnorm_int(
            zi, cosmos['Omega_m_0'], cosmos['Omega_w_0'])

        hmcode_params_z = HMcode_params.HMCode_param_dic(
            cosmos, k_h, pk_cold[i])
        power_spec_dic = {
            'k': k_h, 'power_cold': pk_cold[i],
            'power_axion': pk_axion[i], 'power_total': pk_total[i],
        }
        axion_param_z = axion_params_module.func_axion_param_dic(
            M_arr, cosmos, power_spec_dic, hmcode_params_z,
            concentration_param=hmcode_opts['concentration_param'])
        if print_hmcode_inputs:
            tag = 'AxiCAMB DOME' if dome_calibrated else 'AxiCAMB basic'
            print_hmcode_diagnostics(
                tag, zi, cosmos, hmcode_opts, hmcode_params_z, axion_param_z,
                power_spec_dic
            )
        PS_z = PS_nonlin_axion.func_full_halo_model_ax(
            M_arr, power_spec_dic, cosmos, hmcode_params_z, axion_param_z,
            **hmcode_opts)
        pk_nl[i] = PS_z[0]
        print(f'   z={zi:.1f}: done')

    return k_h, z_sorted, pk_total, pk_nl, pk_cold, pk_axion


def get_axion_camb_nl_pk(axion_setup, ombh2, H0, ns, As, tau, z_arr,
                         halofit_version='mead2020', **cosmo_kwargs):
    """Get axion cosmology P(k) with CAMB's built-in nonlinear correction."""
    pars = make_axion_camb_pars(
        axion_setup, ombh2, H0, ns, As, tau, z_arr,
        nonlinear=True, halofit_version=halofit_version, **cosmo_kwargs)
    results = camb.get_results(pars)

    k_h, z_pk, pk_nl = results.get_nonlinear_matter_power_spectrum(
        hubble_units=True, k_hunit=True)
    _, _, pk_lin = results.get_linear_matter_power_spectrum(
        hubble_units=True, k_hunit=True, nonlinear=False)
    sort_idx = np.argsort(z_pk)
    return k_h, z_pk[sort_idx], pk_lin[sort_idx], pk_nl[sort_idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot P(k) comparisons')
    parser.add_argument('--axionHMcode_path', type=str,
                        default='/Users/adammoss/work/code/axionHMcode',
                        help='Path to axionHMcode')
    parser.add_argument('--m_ax', type=float, default=1e-24,
                        help='Axion mass in eV (default: 1e-24)')
    parser.add_argument('--f_ax', type=float, default=0.1,
                        help='Axion fraction of dark matter (default: 0.1)')
    parser.add_argument('--z', type=float, nargs='+',
                        default=DEFAULT_Z_ARR.tolist(),
                        help='Redshifts for plot (default: 0.0 0.5 1.0 2.0)')
    parser.add_argument('--omega_b', type=float, default=0.022383,
                        help='Baryon density omega_b h^2 (default: 0.022383)')
    parser.add_argument('--omega_d', type=float, default=0.12011,
                        help='Total dark matter density omega_d h^2 (default: 0.12011)')
    parser.add_argument('--h', type=float, default=0.6732,
                        help='Reduced Hubble parameter (default: 0.6732)')
    parser.add_argument('--ns', type=float, default=0.96605,
                        help='Scalar spectral index (default: 0.96605)')
    parser.add_argument('--As', type=float, default=2.10058e-9,
                        help='Scalar amplitude (default: 2.10058e-9)')
    parser.add_argument('--tau', type=float, default=0.0543,
                        help='Optical depth (default: 0.0543)')
    parser.add_argument('--mnu', type=float, default=None,
                        help='Sum of neutrino masses in eV (default: CAMB default)')
    parser.add_argument('--save_data', type=str, default='',
                        help='Optional path to save P(k) data as .npz')
    parser.add_argument('--debug', action='store_true',
                        help='Print axionHMcode inputs/parameters at each redshift')
    args = parser.parse_args()

    if args.axionHMcode_path:
        sys.path.insert(0, args.axionHMcode_path)

    # Derived
    H0 = args.h * 100
    ombh2 = args.omega_b
    omch2 = args.omega_d
    ns = args.ns
    As = args.As
    tau = args.tau
    m_ax = args.m_ax
    f_ax = args.f_ax
    cosmo_kwargs = {}
    if args.mnu is not None:
        cosmo_kwargs['mnu'] = args.mnu
    z_plot, z_compute = prepare_redshifts(args.z)

    # --- Compute ---
    print('Computing LCDM Halofit...')
    k_lcdm, z_lcdm, pk_lin_lcdm, pk_nl_lcdm = get_lcdm_pk(
        ombh2, omch2, H0, ns, As, tau, z_compute, **cosmo_kwargs)

    print(f'\nSolving axion background for f={f_ax}...')
    axion_setup = solve_axion_model(f_ax, m_ax, ombh2, omch2, H0, **cosmo_kwargs)

    axion_results = {}
    for dome in [False, True]:
        label = 'DOME' if dome else 'basic'
        print(f'\nComputing axionHMcode f={f_ax} {label}...')
        k_ax, z_ax, pk_lin_ax, pk_nl_ax, pk_cold_ax, pk_axion_component = get_axion_hmcode_pk(
            axion_setup, m_ax, ombh2, omch2, H0, ns, As, tau, z_compute,
            dome_calibrated=dome,
            print_hmcode_inputs=args.debug, **cosmo_kwargs)
        axion_results[dome] = (
            k_ax, z_ax, pk_lin_ax, pk_nl_ax, pk_cold_ax, pk_axion_component
        )

    print(f'\nComputing axion CAMB HMCode-2020 f={f_ax} naive...')
    axion_camb_nl = get_axion_camb_nl_pk(
        axion_setup, ombh2, H0, ns, As, tau, z_compute, **cosmo_kwargs)

    if args.save_data:
        save_pk_data(
            args.save_data,
            z_compute,
            (k_lcdm, z_lcdm, pk_lin_lcdm, pk_nl_lcdm),
            axion_results[False],
            axion_results[True],
            axion_camb_nl,
            metadata={
                'm_ax': np.array(m_ax, dtype=float),
                'f_ax': np.array(f_ax, dtype=float),
                'omega_b': np.array(ombh2, dtype=float),
                'omega_d': np.array(omch2, dtype=float),
                'h': np.array(args.h, dtype=float),
                'ns': np.array(ns, dtype=float),
                'As': np.array(As, dtype=float),
                'tau': np.array(tau, dtype=float),
            },
        )
        print(f'Saved data {args.save_data}')

    # --- Combined plot: columns = redshifts, rows = absolute / ratio ---
    nz = len(z_plot)
    fig, axes = plt.subplots(2, nz, figsize=(6 * nz, 8),
                              gridspec_kw={'height_ratios': [2, 1]},
                              sharex='col', sharey='row')
    if nz == 1:
        axes = axes.reshape(2, 1)

    for iz, zi in enumerate(z_plot):
        iz_lcdm = get_redshift_index(z_lcdm, zi)

        # --- Top: absolute P(k) ---
        ax = axes[0, iz]

        # Linear spectra (dashed)
        k_ax, z_ax, pk_lin_ax, _, _, _ = axion_results[False]
        iz_ax = get_redshift_index(z_ax, zi)
        ax.loglog(k_ax, pk_lin_ax[iz_ax], color='C0', ls='--',
                  lw=1.5, alpha=0.7,
                  label=f'Axion linear ($f_{{ax}}={f_ax}$)'
                  if iz == 0 else None)
        ax.loglog(k_lcdm, pk_lin_lcdm[iz_lcdm], color='k', ls='--',
                  lw=1.5, alpha=0.7,
                  label=r'$\Lambda$CDM linear' if iz == 0 else None)

        # Non-linear spectra (solid)
        for dome in [False, True]:
            k_ax, z_ax, _, pk_nl_ax, _, _ = axion_results[dome]
            iz_ax = get_redshift_index(z_ax, zi)
            color = 'C3' if dome else 'C0'
            tag = 'DOME' if dome else 'basic'
            ax.loglog(k_ax, pk_nl_ax[iz_ax], color=color, ls='-', lw=1.5,
                      label=f'axionHMcode {tag}' if iz == 0 else None)

        k_ax_camb, z_ax_camb, _, pk_nl_ax_camb = axion_camb_nl
        iz_ax_camb = get_redshift_index(z_ax_camb, zi)
        ax.loglog(k_ax_camb, pk_nl_ax_camb[iz_ax_camb], color='C2',
                  ls='-', lw=1.5,
                  label='Axion CAMB HMCode-2020 (naive)' if iz == 0 else None)
        ax.loglog(k_lcdm, pk_nl_lcdm[iz_lcdm], color='k', ls='-', lw=1.5,
                  label=r'$\Lambda$CDM HMCode-2020' if iz == 0 else None)

        ax.set_title(f'$z = {zi}$', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1e-2, 50)
        if iz == 0:
            ax.set_ylabel(r'$P(k)$ [$(h^{-1}\mathrm{Mpc})^3$]')
            ax.legend(fontsize=8, loc='lower left')

        # --- Bottom: ratio to NL LCDM ---
        ax2 = axes[1, iz]

        interp_nl_lcdm = interp1d(k_lcdm, pk_nl_lcdm[iz_lcdm],
                                   bounds_error=False, fill_value=np.nan)

        for dome in [False, True]:
            k_ax, z_ax, _, pk_nl_ax, _, _ = axion_results[dome]
            iz_ax = get_redshift_index(z_ax, zi)
            pk_nl_lcdm_interp = interp_nl_lcdm(k_ax)
            valid = np.isfinite(pk_nl_lcdm_interp) & (pk_nl_lcdm_interp > 0)
            ratio_nl = pk_nl_ax[iz_ax, valid] / pk_nl_lcdm_interp[valid]

            color = 'C3' if dome else 'C0'
            tag = 'DOME' if dome else 'basic'
            ax2.semilogx(k_ax[valid], ratio_nl, color=color, ls='-', lw=1.5,
                         label=f'axionHMcode {tag}' if iz == 0 else None)

        iz_ax_camb = get_redshift_index(z_ax_camb, zi)
        pk_nl_lcdm_interp = interp_nl_lcdm(k_ax_camb)
        valid = np.isfinite(pk_nl_lcdm_interp) & (pk_nl_lcdm_interp > 0)
        ratio_nl = pk_nl_ax_camb[iz_ax_camb, valid] / pk_nl_lcdm_interp[valid]
        ax2.semilogx(k_ax_camb[valid], ratio_nl, color='C2', ls='-', lw=1.5,
                     label='Axion CAMB HMCode-2020 (naive)' if iz == 0 else None)

        ax2.axhline(1.0, color='k', ls=':', alpha=0.5, lw=1)
        ax2.set_xlabel(r'$k$ [$h$/Mpc]')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.5, 1.6)
        ax2.set_xlim(1e-2, 50)
        if iz == 0:
            ax2.set_ylabel(r'$P_\mathrm{NL}^\mathrm{axion} / '
                           r'P_\mathrm{NL}^{\Lambda\mathrm{CDM}}$')
            ax2.legend(fontsize=8, loc='lower left')

    mass_label = format_mass_label(m_ax)
    fig.suptitle(f'$m_a = {mass_label}$ eV, $f_\\mathrm{{ax}} = {f_ax}$',
                 fontsize=13)
    plt.tight_layout()
    tag_file = f'pk_{format_mass_tag(m_ax)}_f{f_ax}'.replace('.', 'p')
    plt.savefig(os.path.join(FIGDIR, f'{tag_file}.pdf'), dpi=150,
                bbox_inches='tight')
    print(f'\nSaved figures/{tag_file}.pdf')
    plt.close()
