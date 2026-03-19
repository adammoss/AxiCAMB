"""
Helper to run AxiCAMB (Python CAMB + EarlyQuintessence) and return results.
"""
import numpy as np
import camb
from camb import model
from camb.axion_utils import get_axion_phi_i


def run(m_ax=1e-24, f_ax=0.3, z_arr=(0.0,),
        ombh2=0.022383, omch2_total=0.12011, H0=67.32,
        ns=0.96605, As=2.10058e-9, tau=0.0543,
        kmax=50.0, use_PH=True, mH=50.0, verbose=True,
        **cosmo_kwargs):
    """Run AxiCAMB and return matter power spectra.

    Returns
    -------
    dict with keys:
        'k': k array in h/Mpc
        'z': redshift array (sorted ascending)
        'pk': P(k) array, shape (n_z, n_k)
        'sigma8': sigma8 at z=0
        'params': axion params dict from shooting
        'results': CAMB results object
    """
    params = get_axion_phi_i(
        h=H0 / 100, ombh2=ombh2, omch2_total=omch2_total,
        f_ax=f_ax, mass_ev=m_ax, verbose=verbose,
        use_PH=use_PH, mH=mH, **cosmo_kwargs)

    if params is None:
        raise RuntimeError('Failed to find axion initial conditions')

    omch2_cdm = max((1 - f_ax) * omch2_total, 1e-7)

    pars = camb.set_params(
        H0=H0, ombh2=ombh2, omch2=omch2_cdm, omk=0, tau=tau,
        As=As, ns=ns, **cosmo_kwargs,
        dark_energy_model='EarlyQuintessence',
        m=params['m'], theta_i=params['theta_i'],
        frac_lambda0=params['frac_lambda0'],
        use_zc=False, use_fluid_approximation=True,
        potential_type=1, weighting_factor=10.0,
        oscillation_threshold=1, use_PH=use_PH, mH=mH)

    pars.set_for_lmax(2500, lens_potential_accuracy=1)
    pars.NonLinear = model.NonLinear_none
    # Disable late radiation truncation for EarlyQuintessence models
    # (the approximation uses grhoc+grhob without the axion contribution)
    pars.DoLateRadTruncation = False
    z_compute = sorted(set(list(z_arr) + [0.0]), reverse=True)
    pars.set_matter_power(redshifts=z_compute, kmax=kmax)

    results = camb.get_results(pars)

    k_h, z_pk, pk = results.get_linear_matter_power_spectrum(
        hubble_units=True, k_hunit=True, nonlinear=False)

    sort_idx = np.argsort(z_pk)
    z_sorted = z_pk[sort_idx]
    pk_sorted = pk[sort_idx]

    s8 = results.get_sigma8_0()
    if verbose:
        print(f'  [AxiCAMB] sigma8 = {s8:.6f}')

    return {
        'k': k_h,
        'z': z_sorted,
        'pk': pk_sorted,
        'sigma8': s8,
        'params': params,
        'results': results,
    }


def get_perturbation_evolution(m_ax=1e-24, f_ax=0.3, k_hMpc=0.5,
                                z_eval=None, **kwargs):
    """Get perturbation evolution delta(a) for given k.

    Returns dict with 'a', 'delta_cdm', 'delta_baryon', 'delta_axion'.
    """
    result = run(m_ax=m_ax, f_ax=f_ax, verbose=False, **kwargs)
    camb_results = result['results']

    if z_eval is None:
        z_eval = np.logspace(-5, np.log10(3e4), 500)[::-1]
    z_eval = z_eval[z_eval >= 0]

    # get_redshift_evolution takes q in Mpc^-1, convert from h/Mpc
    h = result['results'].Params.H0 / 100.0
    q = k_hMpc * h
    ev = camb_results.get_redshift_evolution(
        q, z_eval,
        ['delta_cdm', 'delta_baryon', 'delta_axion'])

    a = 1.0 / (1.0 + z_eval)

    return {
        'a': a,
        'delta_cdm': ev[:, 0],
        'delta_baryon': ev[:, 1],
        'delta_axion': ev[:, 2],
        'k': k_hMpc,
    }


def get_lcdm_pk(z_arr=(0.0,), ombh2=0.022383, omch2=0.12011,
                H0=67.32, ns=0.96605, As=2.10058e-9, tau=0.0543,
                kmax=50.0, **cosmo_kwargs):
    """Get LCDM linear P(k) for reference."""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau,
                       **cosmo_kwargs)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(2500, lens_potential_accuracy=1)
    pars.NonLinear = model.NonLinear_none
    pars.set_matter_power(
        redshifts=sorted(z_arr, reverse=True), kmax=kmax)
    results = camb.get_results(pars)

    k_h, z_pk, pk = results.get_linear_matter_power_spectrum(
        hubble_units=True, k_hunit=True, nonlinear=False)
    sort_idx = np.argsort(z_pk)
    return k_h, z_pk[sort_idx], pk[sort_idx]
