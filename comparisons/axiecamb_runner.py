"""
Helper to run AxiECAMB (compiled binary) from Python and load results.

Requires AxiECAMB compiled at AXIECAMB_DIR.
"""
import os
import re
import subprocess
import tempfile
import numpy as np

AXIECAMB_DIR = os.environ.get('AXIECAMB_DIR',
                               '/Users/adammoss/work/code/AxiECAMB')

PARAMS_TEMPLATE = """\
output_root = {output_root}
get_scalar_cls = F
get_vector_cls = F
get_tensor_cls = F
CMB_outputscale = 7.4311e12
get_transfer = T
accuracy_boost = 1
l_accuracy_boost = 1
high_accuracy_default = T
do_nonlinear = 0
l_max_scalar = 2700
k_eta_max_scalar = 6000
do_lensing = F
lensing_method = 1
use_tabulated_w = F
w = -1
wa = 0.0
cs2_lam = 1
hubble = {H0}
omk = 0.0
use_physical = T
omnuh2 = {omnuh2}
temp_cmb = 2.7255
helium_fraction = {YHe}
massless_neutrinos = {Neff}
nu_mass_eigenstates = 1
massive_neutrinos = {massive_neutrinos}
share_delta_neff = T
nu_mass_fractions = 1
ombh2 = {ombh2}
use_axfrac = T
m_ax = {m_ax}
omdah2 = {omdah2}
axfrac = {f_ax}
movH_switch = {movH_switch}
Hinf = 13.7
pert_output_kh = {pert_output_kh}
axion_isocurvature = F
transfer_high_precision = T
transfer_kmax = {kmax}
transfer_k_per_logint = 50
transfer_num_redshifts = {num_z}
transfer_interp_matterpower = T
transfer_power_var = 9
{redshift_block}
DebugParam = 0.0
Alens = 1.0
reionization = T
re_use_optical_depth = T
re_optical_depth = {tau}
re_delta_redshift = 1.5
re_ionization_frac = -1
pivot_scalar = 0.05
pivot_tensor = 0.05
initial_power_num = 1
scalar_spectral_index(1) = {ns}
scalar_nrun(1) = 0
scalar_amp(1) = {As}
RECFAST_fudge_He = 0.86
RECFAST_Heswitch = 6
RECFAST_Hswitch = T
RECFAST_fudge = 1.14
AGauss1 = -0.14
AGauss2 = 0.079
zGauss1 = 7.28
zGauss2 = 6.73
wGauss1 = 0.18
wGauss2 = 0.33
do_lensing_bispectrum = F
do_primordial_bispectrum = F
initial_condition = 1
scalar_output_file = scalCls.dat
lensed_output_file = lensedCls.dat
lens_potential_output_file = lenspotentialCls.dat
accurate_polarization = T
accurate_reionization = T
accurate_BB = T
derived_parameters = T
version_check = Nov13
do_late_rad_truncation = T
do_tensor_neutrinos = T
feedback_level = 1
massive_nu_approx = 1
number_of_threads = 0
use_spline_template = T
l_sample_boost = 1
"""


def _build_redshift_block(z_arr):
    """Build the redshift/filename entries for the params file."""
    lines = []
    for i, z in enumerate(sorted(z_arr, reverse=True), 1):
        lines.append(f'transfer_redshift({i}) = {z}')
        lines.append(f'transfer_filename({i}) = transfer_out_{i}.dat')
        lines.append(f'transfer_matterpower({i}) = matterpower_{i}.dat')
    return '\n'.join(lines)


def run(m_ax=1e-24, f_ax=0.3, z_arr=(0.0,),
        ombh2=0.022383, omdah2=0.12011, H0=67.32,
        ns=0.96605, As=2.10058e-9, tau=0.0543,
        YHe=0.245861, omnuh2=0.0, Neff=3.046,
        massive_neutrinos=0, kmax=50.0,
        movH_switch=50.0, pert_output_kh=0.0,
        axiecamb_dir=None, verbose=True):
    """Run AxiECAMB and return matter power spectra.

    Returns
    -------
    dict with keys:
        'k': k array in h/Mpc
        'z': redshift array (sorted ascending)
        'pk': P(k) array, shape (n_z, n_k)
        'sigma8': list of sigma8 values per redshift
        'stdout': raw output text
    """
    if axiecamb_dir is None:
        axiecamb_dir = AXIECAMB_DIR

    camb_exe = os.path.join(axiecamb_dir, 'camb')
    if not os.path.exists(camb_exe):
        raise FileNotFoundError(f'AxiECAMB binary not found at {camb_exe}')

    z_arr = sorted(set(list(z_arr) + [0.0]), reverse=True)
    output_root = 'compare'

    params_text = PARAMS_TEMPLATE.format(
        output_root=output_root,
        H0=H0, ombh2=ombh2, omdah2=omdah2,
        m_ax=m_ax, f_ax=f_ax, movH_switch=movH_switch,
        pert_output_kh=pert_output_kh,
        ns=ns, As=As, tau=tau,
        YHe=YHe, omnuh2=omnuh2, Neff=Neff,
        massive_neutrinos=massive_neutrinos,
        kmax=kmax, num_z=len(z_arr),
        redshift_block=_build_redshift_block(z_arr),
    )

    # Run from axiecamb_dir so auxiliary files (lensing template, background dump) are accessible
    with tempfile.TemporaryDirectory() as tmpdir:
        params_path = os.path.join(tmpdir, 'params.ini')
        with open(params_path, 'w') as f:
            f.write(params_text)

        result = subprocess.run(
            [camb_exe, params_path],
            capture_output=True, text=True, cwd=axiecamb_dir,
            timeout=300,
        )

        if verbose:
            for line in result.stdout.split('\n'):
                if 'sigma8' in line.lower() or 'error' in line.lower():
                    print(f'  [AxiECAMB] {line.strip()}')

        # Parse sigma8
        sigma8_list = []
        for line in result.stdout.split('\n'):
            if 'sigma8' in line:
                m = re.search(r'sigma8.*=\s*([\d.eE+-]+)', line)
                if m:
                    sigma8_list.append(float(m.group(1)))

        # Load matter power spectra
        z_sorted = sorted(z_arr, reverse=True)
        pk_list = []
        k_ref = None
        for i in range(1, len(z_sorted) + 1):
            pk_file = os.path.join(axiecamb_dir, f'{output_root}_matterpower_{i}.dat')
            if os.path.exists(pk_file):
                data = np.loadtxt(pk_file)
                if k_ref is None:
                    k_ref = data[:, 0]
                pk_list.append(data[:, 1])

        if k_ref is None:
            raise RuntimeError(
                f'No power spectrum files found. AxiECAMB output:\n'
                f'{result.stdout[-500:]}\n{result.stderr[-500:]}')

        pk_arr = np.array(pk_list)
        # Reverse so z is ascending
        z_out = np.array(z_sorted[::-1])
        pk_out = pk_arr[::-1]

        # Load perturbation evolution if requested
        pert_evolution = None
        pert_file = os.path.join(axiecamb_dir, 'evolve_pert.txt')
        if pert_output_kh > 0 and os.path.exists(pert_file):
            pert_data = np.loadtxt(pert_file, skiprows=1)
            if pert_data.size > 0:
                pert_evolution = {
                    'tau': pert_data[:, 0],
                    'a': pert_data[:, 1],
                    'delta_cdm': pert_data[:, 2],
                    'delta_baryon': pert_data[:, 3],
                    'delta_axion': pert_data[:, 4],
                    'k': pert_output_kh,
                }
                if verbose:
                    print(f'  [AxiECAMB] Loaded perturbation evolution: '
                          f'{len(pert_data)} steps, k={pert_output_kh} h/Mpc')

    return {
        'k': k_ref,
        'z': z_out,
        'pk': pk_out,
        'sigma8': sigma8_list[::-1],
        'stdout': result.stdout,
        'pert_evolution': pert_evolution,
    }


def get_linear_pk(m_ax=1e-24, f_ax=0.3, z_arr=(0.0,), **kwargs):
    """Convenience wrapper matching the AxiCAMB interface.

    Returns k, z, pk arrays.
    """
    result = run(m_ax=m_ax, f_ax=f_ax, z_arr=z_arr, **kwargs)
    return result['k'], result['z'], result['pk']
