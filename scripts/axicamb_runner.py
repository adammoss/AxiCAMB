"""
Helper to run AxiCAMB and return results.
"""
import numpy as np
import camb
from camb import model
from camb.axion_utils import get_axion_phi_i


def run(m_ax=1e-24, f_ax=0.3, z_arr=(0.0,),
        ombh2=0.022383, omch2_total=0.12011, H0=67.32,
        ns=0.96605, As=2.10058e-9, tau=0.0543,
        kmax=50.0, use_PH=True, mH=50.0, DoLateRadTruncation=True,
        nonlinear=False, halofit_version='mead2020',
        get_cls=False, do_lensing=False, lmax=2500,
        verbose=True, **cosmo_kwargs):
    """Run AxiCAMB and return matter power spectra and optionally Cls.

    Returns
    -------
    dict with keys:
        'k': k array in h/Mpc
        'z': redshift array (sorted ascending)
        'pk': P(k) array, shape (n_z, n_k)
        'sigma8': sigma8 at z=0
        'params': axion params dict from shooting
        'results': CAMB results object
        'cls': dict with 'ell', 'tt', 'ee', 'te' (if get_cls=True)
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

    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    if nonlinear:
        pars.NonLinear = model.NonLinear_both
        pars.NonLinearModel.halofit_version = halofit_version
    else:
        pars.NonLinear = model.NonLinear_none
    pars.DoLateRadTruncation = DoLateRadTruncation
    pars.DoLensing = do_lensing
    z_compute = sorted(set(list(z_arr) + [0.0]), reverse=True)
    pars.set_matter_power(redshifts=z_compute, kmax=kmax)

    results = camb.get_results(pars)

    k_h, z_pk, pk = results.get_linear_matter_power_spectrum(
        hubble_units=True, k_hunit=True, nonlinear=False)

    sort_idx = np.argsort(z_pk)
    z_sorted = z_pk[sort_idx]
    pk_sorted = pk[sort_idx]

    pk_nl = None
    if nonlinear:
        _, _, pk_nl_raw = results.get_nonlinear_matter_power_spectrum(
            hubble_units=True, k_hunit=True)
        pk_nl = pk_nl_raw[sort_idx]

    s8 = results.get_sigma8_0()
    if verbose:
        print(f'  [AxiCAMB] sigma8 = {s8:.6f}')

    cls = None
    if get_cls:
        if do_lensing:
            cls_data = results.get_lensed_scalar_cls(lmax=lmax, CMB_unit='muK')
        else:
            cls_data = results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit='muK')
        ell = np.arange(cls_data.shape[0])
        cls = {
            'ell': ell,
            'tt': cls_data[:, 0],
            'ee': cls_data[:, 1],
            'te': cls_data[:, 3],
        }

    return {
        'k': k_h,
        'z': z_sorted,
        'pk': pk_sorted,
        'pk_nl': pk_nl,
        'sigma8': s8,
        'params': params,
        'results': results,
        'cls': cls,
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


def get_lcdm(z_arr=(0.0,), ombh2=0.022383, omch2=0.12011,
             H0=67.32, ns=0.96605, As=2.10058e-9, tau=0.0543,
             kmax=50.0, get_cls=False, do_lensing=False, lmax=2500,
             nonlinear=False, halofit_version='mead2020',
             **cosmo_kwargs):
    """Get LCDM P(k) and optionally Cls."""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau,
                       **cosmo_kwargs)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    if nonlinear:
        pars.NonLinear = model.NonLinear_both
        pars.NonLinearModel.halofit_version = halofit_version
    else:
        pars.NonLinear = model.NonLinear_none
    pars.DoLensing = do_lensing
    z_compute = sorted(set(list(z_arr) + [0.0]), reverse=True)
    pars.set_matter_power(redshifts=z_compute, kmax=kmax)
    results = camb.get_results(pars)

    k_h, z_pk, pk_lin = results.get_linear_matter_power_spectrum(
        hubble_units=True, k_hunit=True, nonlinear=False)
    sort_idx = np.argsort(z_pk)

    pk_nl = None
    if nonlinear:
        _, _, pk_nl_raw = results.get_nonlinear_matter_power_spectrum(
            hubble_units=True, k_hunit=True)
        pk_nl = pk_nl_raw[sort_idx]

    cls = None
    if get_cls:
        if do_lensing:
            cls_data = results.get_lensed_scalar_cls(lmax=lmax, CMB_unit='muK')
        else:
            cls_data = results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit='muK')
        ell = np.arange(cls_data.shape[0])
        cls = {
            'ell': ell,
            'tt': cls_data[:, 0],
            'ee': cls_data[:, 1],
            'te': cls_data[:, 3],
        }

    return {
        'k': k_h,
        'z': z_pk[sort_idx],
        'pk': pk_lin[sort_idx],
        'pk_nl': pk_nl,
        'cls': cls,
    }


def get_lcdm_pk(z_arr=(0.0,), **kwargs):
    """Get LCDM linear P(k) — backward compatible wrapper."""
    result = get_lcdm(z_arr=z_arr, **kwargs)
    return result['k'], result['z'], result['pk']


def get_component_spectra(result):
    """Extract component power spectra (cold, axion, total) from a run result.

    Parameters
    ----------
    result : dict from run()

    Returns
    -------
    dict with 'k', 'z', 'pk_total', 'pk_cold', 'pk_axion', plus
    'omch2_cdm', 'ombh2', 'omega_ax_h2' for axionHMcode input.
    """
    results = result['results']
    params = result['params']
    pars = results.Params

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

    w_c = pars.omch2
    w_b = pars.ombh2
    pk_cold = (
        w_c**2 * pk_cc + w_b**2 * pk_bb + 2 * w_c * w_b * pk_cb
    ) / (w_c + w_b)**2

    return {
        'k': k_h,
        'z': z_pk[sort_idx],
        'pk_total': pk_total[sort_idx],
        'pk_cold': pk_cold[sort_idx],
        'pk_axion': pk_axion[sort_idx],
        'omch2_cdm': pars.omch2,
        'ombh2': pars.ombh2,
        'omega_ax_h2': params['omega_ax_h2'],
    }


def get_axionhmcode_pk(result, m_ax, dome_calibrated=False,
                        axionhmcode_path=None, verbose=True):
    """Run axionHMcode on linear spectra from an AxiCAMB run.

    Parameters
    ----------
    result : dict from run()
    m_ax : float, axion mass in eV
    dome_calibrated : bool, use DOME calibration
    axionhmcode_path : str, path to axionHMcode repo

    Returns
    -------
    dict with 'k', 'z', 'pk_nl', 'pk_total', 'pk_cold', 'pk_axion'
    """
    import sys
    if axionhmcode_path is None:
        axionhmcode_path = '/Users/adammoss/work/code/axionHMcode'
    if axionhmcode_path not in sys.path:
        sys.path.insert(0, axionhmcode_path)

    from halo_model import HMcode_params, PS_nonlin_axion
    from axion_functions import axion_params as axion_params_module
    from cosmology.overdensities import func_D_z_unnorm_int

    comp = get_component_spectra(result)
    pars = result['results'].Params
    h = pars.H0 / 100.0
    omch2 = comp['omch2_cdm'] + comp['omega_ax_h2']
    omega_d = comp['omch2_cdm']
    omega_ax_h2 = comp['omega_ax_h2']
    ombh2 = comp['ombh2']
    ns = pars.InitPower.ns
    As = pars.InitPower.As

    k_h = comp['k']
    M_arr = np.logspace(7, 18, 100)
    pk_nl = np.zeros_like(comp['pk_total'])

    hmcode_opts = {
        'alpha': dome_calibrated, 'eta_given': dome_calibrated,
        'one_halo_damping': True, 'two_halo_damping': dome_calibrated,
        'concentration_param': dome_calibrated, 'full_2h': False,
    }

    for i, zi in enumerate(comp['z']):
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
            cosmos, k_h, comp['pk_cold'][i])
        power_spec_dic = {
            'k': k_h, 'power_cold': comp['pk_cold'][i],
            'power_axion': comp['pk_axion'][i],
            'power_total': comp['pk_total'][i],
        }
        axion_param_z = axion_params_module.func_axion_param_dic(
            M_arr, cosmos, power_spec_dic, hmcode_params_z,
            concentration_param=hmcode_opts['concentration_param'])
        PS_z = PS_nonlin_axion.func_full_halo_model_ax(
            M_arr, power_spec_dic, cosmos, hmcode_params_z, axion_param_z,
            **hmcode_opts)
        pk_nl[i] = PS_z[0]
        if verbose:
            print(f'   z={zi:.1f}: done')

    return {
        'k': k_h,
        'z': comp['z'],
        'pk_nl': pk_nl,
        'pk_total': comp['pk_total'],
        'pk_cold': comp['pk_cold'],
        'pk_axion': comp['pk_axion'],
    }
