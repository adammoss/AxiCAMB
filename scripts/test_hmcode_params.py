"""
Analyse HMCode internal parameters for EarlyQuintessence axion models.

Generates Fortran P(k) and nonlinear dumps via FeedbackLevel=2, then
reads them to compare LCDM vs axion HMCode internals.

The Fortran dumps (written at the first HMCode redshift, typically z~9.8):
  hmcode_pk_dump.dat — P(k) table (512 points, grown to z=0 via g(z)^2)
  hmcode_nl_dump.dat — nonlinear decomposition (1h, 2h, total) at all z
  hmcode_params_dump.dat — HMCode internal parameters (n_eff, alpha, etc.)

Usage:
    python test_hmcode_params.py
    python test_hmcode_params.py --m_ax 1e-23 --f_ax 0.5
    python test_hmcode_params.py --regenerate   # force regenerate dumps
"""
import numpy as np
import sys, os
import argparse
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import camb
from camb import model
from camb.axion_utils import get_axion_phi_i
import cosmo_params as cp

TESTDIR = os.path.join(os.path.dirname(__file__), 'test_data')
os.makedirs(TESTDIR, exist_ok=True)


# ---- Dump file I/O ----

def load_pk_dump(filename):
    """Load Fortran HMCode P(k) dump (hmcode_pk_dump.dat).

    Written by fill_plintab on the first HMCode redshift (typically z~9.8).
    Columns 1-2 contain P(k) grown to z=0 via the internal growth factor.
    Columns 3-4 contain P(k) at the input redshift.

    Columns:
        0: k [h/Mpc] — 512 log-spaced points in [0.001, 100]
        1: log(Delta^2_tot(k, z=0)) — total matter, grown to z=0 via g(z)^2
        2: log(Delta^2_cold(k, z=0)) — cold (CDM+baryon) via Tcb/Tcbnu correction
        3: Delta^2_tot(k, z_input) — total matter at the input redshift
        4: Delta^2_cold(k, z_input) — cold matter at the input redshift
    """
    data = np.loadtxt(filename)
    return {
        'k': data[:, 0],
        'd2_z0_tot': np.exp(data[:, 1]),
        'd2_z0_cold': np.exp(data[:, 2]),
        'd2_zinput_tot': data[:, 3],
        'd2_zinput_cold': data[:, 4],
    }


def load_params_dump(filename):
    """Load Fortran HMCode parameter dump (hmcode_params_dump.dat).

    Header contains cosmological parameters (constant across z).
    Followed by rows of z-dependent parameters for each redshift.

    Returns dict with:
        - Cosmological params as scalars (om_m, om_v, sigma_8, etc.)
        - z-dependent params as arrays (z, n_eff, alpha, etc.)
    """
    params = {}
    z_data = []
    z_cols = ['z', 'sig8z', 'sig8z_cold', 'dc', 'r_nl', 'k_nl',
              'sigma_rnl', 'n_eff', 'nu_min', 'nu_max', 'sigv', 'sigv100', 'alpha']

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 2:
                # Header: key value
                params[parts[0]] = float(parts[1])
            elif len(parts) == len(z_cols):
                # z-dependent row
                z_data.append([float(x) for x in parts])

    if z_data:
        z_arr = np.array(z_data)
        for i, col in enumerate(z_cols):
            params[col] = z_arr[:, i]

    return params


def load_nl_dump(filename, z_select=None):
    """Load Fortran HMCode nonlinear dump.

    Columns:
        0: k [h/Mpc]
        1: z — redshift
        2: plin — Delta^2_lin(k, z) from p_lin(k, z, itype=0)
        3: p1h — one-halo term Delta^2
        4: p2h — two-halo term Delta^2
        5: pfull — total nonlinear Delta^2 = (p2h^alpha + p1h^alpha)^(1/alpha)
        6: nonlin_ratio — sqrt(pfull / plin)

    Parameters:
        z_select: if given, return only rows closest to this redshift.
    """
    data = np.loadtxt(filename)
    if z_select is not None:
        z_vals = np.unique(data[:, 1])
        z_closest = z_vals[np.argmin(np.abs(z_vals - z_select))]
        data = data[data[:, 1] == z_closest]
    return {
        'k': data[:, 0],
        'z': data[:, 1],
        'plin': data[:, 2],
        'p1h': data[:, 3],
        'p2h': data[:, 4],
        'pfull': data[:, 5],
        'ratio': data[:, 6],
    }


# ---- Run CAMB with dumps ----

def run_and_dump(label, pars):
    """Run CAMB with FeedbackLevel=2 to generate dumps, move to test_data/."""
    pars.NonLinear = model.NonLinear_both
    pars.NonLinearModel.halofit_version = 'mead2020'
    pars.set_matter_power(redshifts=[0], kmax=50)

    camb.set_feedback_level(2)
    res = camb.get_results(pars)
    camb.set_feedback_level(0)

    for src, suffix in [('hmcode_pk_dump.dat', 'pk'),
                        ('hmcode_nl_dump.dat', 'nl'),
                        ('hmcode_params_dump.dat', 'params')]:
        dst = os.path.join(TESTDIR, f'hmcode_{suffix}_{label}.dat')
        if os.path.exists(src):
            shutil.move(src, dst)

    return res


def generate_dumps(cosmo, m_ax, f_ax):
    """Generate LCDM and axion dumps."""
    print('Generating LCDM dumps...')
    pars_l = camb.set_params(
        H0=cosmo['H0'], ombh2=cosmo['ombh2'], omch2=cosmo['omch2_total'],
        tau=cosmo['tau'], As=cosmo['As'], ns=cosmo['ns'], mnu=cosmo['mnu'])
    run_and_dump('lcdm', pars_l)

    label = f'axion_m{m_ax:.0e}_f{f_ax}'.replace('.', 'p')
    print(f'Generating axion dumps (m={m_ax:.0e}, f={f_ax})...')
    params = get_axion_phi_i(
        h=cosmo['H0'] / 100, ombh2=cosmo['ombh2'],
        omch2_total=cosmo['omch2_total'], f_ax=f_ax, mass_ev=m_ax,
        verbose=False, mnu=cosmo['mnu'])
    omch2_cdm = max((1 - f_ax) * cosmo['omch2_total'], 1e-7)
    pars_a = camb.set_params(
        H0=cosmo['H0'], ombh2=cosmo['ombh2'], omch2=omch2_cdm,
        tau=cosmo['tau'], As=cosmo['As'], ns=cosmo['ns'], mnu=cosmo['mnu'],
        dark_energy_model='EarlyQuintessence',
        m=params['m'], theta_i=params['theta_i'],
        frac_lambda0=params['frac_lambda0'],
        use_zc=False, use_fluid_approximation=True,
        potential_type=1, weighting_factor=10.0,
        oscillation_threshold=1, use_PH=True, mH=50.0)
    run_and_dump(label, pars_a)

    return label


# ---- Analysis ----

def print_nl_decomposition(nl_l, nl_a, par_l, par_a):
    """Print nonlinear 1h/2h decomposition table."""
    z_nl = nl_l['z'][0]

    # Get n_eff and alpha at the matching z
    if isinstance(par_l.get('n_eff'), np.ndarray):
        iz_l = np.argmin(np.abs(par_l['z'] - z_nl))
        iz_a = np.argmin(np.abs(par_a['z'] - z_nl))
        neff_l = par_l['n_eff'][iz_l]
        neff_a = par_a['n_eff'][iz_a]
        alpha_l = par_l['alpha'][iz_l]
        alpha_a = par_a['alpha'][iz_a]
    else:
        neff_l = par_l.get('n_eff', 0)
        neff_a = par_a.get('n_eff', 0)
        alpha_l = par_l.get('alpha', 1.875 * 1.603**neff_l)
        alpha_a = par_a.get('alpha', 1.875 * 1.603**neff_a)

    print(f'\n--- Nonlinear decomposition (z={z_nl:.1f}) ---')
    print(f'n_eff:   LCDM={neff_l:.5f}  Axion={neff_a:.5f}')
    print(f'alpha:   LCDM={alpha_l:.4f}  Axion={alpha_a:.4f}')
    print(f'  (lower alpha -> P_NL inflated at quasi-linear scales)')
    print()
    print(f'{"k":>8s} {"lin_A/L":>8s} {"p1h_A/L":>8s} {"p2h_A/L":>8s} '
          f'{"boost_L":>8s} {"boost_A":>8s} {"nl_A/L":>8s}')
    print('-' * 58)
    for kv in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 5.0, 10.0]:
        il = np.argmin(np.abs(nl_l['k'] - kv))
        ia = np.argmin(np.abs(nl_a['k'] - kv))
        lin_ratio = nl_a['plin'][ia] / nl_l['plin'][il]
        p1h_ratio = nl_a['p1h'][ia] / nl_l['p1h'][il]
        p2h_ratio = nl_a['p2h'][ia] / nl_l['p2h'][il]
        boost_l = nl_l['pfull'][il] / nl_l['plin'][il]
        boost_a = nl_a['pfull'][ia] / nl_a['plin'][ia]
        nl_ratio = nl_a['pfull'][ia] / nl_l['pfull'][il]
        print(f'{kv:8.2f} {lin_ratio:8.4f} {p1h_ratio:8.4f} {p2h_ratio:8.4f} '
              f'{boost_l:8.4f} {boost_a:8.4f} {nl_ratio:8.4f}')

    print()
    print('lin_A/L:  linear P(k) ratio (axion suppression)')
    print('p1h_A/L:  one-halo ratio')
    print('p2h_A/L:  two-halo ratio (tracks linear suppression)')
    print('boost:    pfull/plin (nonlinear boost factor)')
    print('nl_A/L:   nonlinear P(k) ratio (>1 = spurious excess)')


def main():
    parser = argparse.ArgumentParser()
    cp.add_cli_args(parser)
    parser.add_argument('--regenerate', action='store_true',
                        help='Force regenerate Fortran dumps')
    parser.add_argument('--z_nl', type=float, nargs='*', default=[0, 1, 2],
                        help='Redshifts for nonlinear decomposition (default: 0 1 2)')
    args = parser.parse_args()

    cosmo, axion = cp.from_args(args)
    m_ax_list = axion['m_ax_list']
    f_ax = axion['f_ax']

    # LCDM dumps (shared)
    pk_l_file = os.path.join(TESTDIR, 'hmcode_pk_lcdm.dat')
    nl_l_file = os.path.join(TESTDIR, 'hmcode_nl_lcdm.dat')
    par_l_file = os.path.join(TESTDIR, 'hmcode_params_lcdm.dat')

    lcdm_exists = all(os.path.exists(f) for f in [pk_l_file, nl_l_file, par_l_file])
    if args.regenerate or not lcdm_exists:
        generate_dumps(cosmo, m_ax_list[0], f_ax)  # LCDM generated here

    par_l = load_params_dump(par_l_file)

    for m_ax in m_ax_list:
        label = f'axion_m{m_ax:.0e}_f{f_ax}'.replace('.', 'p')
        pk_a_file = os.path.join(TESTDIR, f'hmcode_pk_{label}.dat')
        nl_a_file = os.path.join(TESTDIR, f'hmcode_nl_{label}.dat')
        par_a_file = os.path.join(TESTDIR, f'hmcode_params_{label}.dat')

        axion_exists = all(os.path.exists(f) for f in [pk_a_file, nl_a_file, par_a_file])
        if args.regenerate or not axion_exists:
            generate_dumps(cosmo, m_ax, f_ax)

        par_a = load_params_dump(par_a_file)

        print(f'\n{"="*60}')
        print(f'm_ax = {m_ax:.0e} eV, f_ax = {f_ax}')
        print(f'{"="*60}')

        # Cosmological parameters (z-independent)
        print(f'\n{"Quantity":25s} {"LCDM":>14s} {"Axion":>14s}')
        print('-' * 55)
        for key in ['om_m', 'om_v', 'sigma_8', 'sigma_8_cold']:
            print(f'{key:25s} {par_l[key]:14.6g} {par_a[key]:14.6g}')

        # z-dependent parameters
        z_show = [0, 1, 2]
        z_cols = ['n_eff', 'alpha', 'r_nl', 'sigma_rnl', 'nu_min', 'sig8z']
        if isinstance(par_l.get('n_eff'), np.ndarray):
            z_l = par_l['z']
            z_a = par_a['z']
            for z_sel in z_show:
                iz_l = np.argmin(np.abs(z_l - z_sel))
                iz_a = np.argmin(np.abs(z_a - z_sel))
                print(f'\n  z = {z_l[iz_l]:.1f}:')
                for col in z_cols:
                    print(f'  {col:23s} {par_l[col][iz_l]:14.6g} {par_a[col][iz_a]:14.6g}')

        # Nonlinear decomposition
        if os.path.exists(nl_l_file) and os.path.exists(nl_a_file):
            for z_nl in args.z_nl:
                nl_l = load_nl_dump(nl_l_file, z_select=z_nl)
                nl_a = load_nl_dump(nl_a_file, z_select=z_nl)
                print_nl_decomposition(nl_l, nl_a, par_l, par_a)


if __name__ == '__main__':
    main()
