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
import subprocess
import tempfile
import re

import camb
from camb import model
from camb.axion_utils import get_axion_phi_i
import axiecamb_runner
import cosmo_params

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

AXIECAMB_CLS_TEMPLATE = """\
output_root = {output_root}
get_scalar_cls = T
get_vector_cls = F
get_tensor_cls = F
CMB_outputscale = 7.4311e12
get_transfer = F
accuracy_boost = 1
l_accuracy_boost = 1
high_accuracy_default = T
do_nonlinear = 0
l_max_scalar = {lmax}
k_eta_max_scalar = {k_eta_max}
do_lensing = {do_lensing}
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
axion_isocurvature = F
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


def run_axiecamb_cls(m_ax, f_ax, lmax=3000,
                     ombh2=0.022383, omdah2=0.12011, H0=67.32,
                     ns=0.96605, As=2.10058e-9, tau=0.0543,
                     YHe=0.245861, omnuh2=0.0, Neff=3.046,
                     massive_neutrinos=0, movH_switch=10.0,
                     do_lensing=True, verbose=True):
    """Run AxiECAMB for Cls."""
    axiecamb_dir = axiecamb_runner.AXIECAMB_DIR
    camb_exe = os.path.join(axiecamb_dir, 'camb')
    output_root = 'compare_cls'

    params_text = AXIECAMB_CLS_TEMPLATE.format(
        output_root=output_root,
        H0=H0, ombh2=ombh2, omdah2=omdah2,
        m_ax=m_ax, f_ax=f_ax, movH_switch=movH_switch,
        ns=ns, As=As, tau=tau,
        YHe=YHe, omnuh2=omnuh2, Neff=Neff,
        massive_neutrinos=massive_neutrinos,
        lmax=lmax, k_eta_max=2 * lmax,
        do_lensing='T' if do_lensing else 'F',
    )

    params_path = os.path.join(axiecamb_dir, f'{output_root}_params.ini')
    with open(params_path, 'w') as f:
        f.write(params_text)

    result = subprocess.run(
        [camb_exe, params_path],
        capture_output=True, text=True, cwd=axiecamb_dir,
        timeout=600,
    )

    if verbose:
        for line in result.stdout.split('\n'):
            if 'sigma8' in line.lower() or 'error' in line.lower():
                print(f'  [AxiECAMB] {line.strip()}')

    if do_lensing:
        cls_file = os.path.join(axiecamb_dir, f'{output_root}_lensedCls.dat')
    else:
        cls_file = os.path.join(axiecamb_dir, f'{output_root}_scalCls.dat')

    if not os.path.exists(cls_file):
        raise RuntimeError(
            f'Cls file not found at {cls_file}\n'
            f'{result.stdout[-500:]}\n{result.stderr[-500:]}')

    data = np.loadtxt(cls_file)
    ell = data[:, 0].astype(int)
    cls_tt = data[:, 1]
    cls_ee = data[:, 2]
    cls_te = data[:, 3] if data.shape[1] > 3 else None

    return {'ell': ell, 'tt': cls_tt, 'ee': cls_ee, 'te': cls_te}


def run_axicamb_cls(m_ax, f_ax, lmax=3000,
                    ombh2=0.022383, omch2_total=0.12011, H0=67.32,
                    ns=0.96605, As=2.10058e-9, tau=0.0543,
                    mH=50.0, do_lensing=True, verbose=True, **cosmo_kwargs):
    """Run AxiCAMB for Cls."""
    params = get_axion_phi_i(
        h=H0/100, ombh2=ombh2, omch2_total=omch2_total,
        f_ax=f_ax, mass_ev=m_ax, verbose=False,
        mH=mH, **cosmo_kwargs)

    omch2_cdm = max((1 - f_ax) * omch2_total, 1e-7)

    pars = camb.set_params(
        H0=H0, ombh2=ombh2, omch2=omch2_cdm, omk=0, tau=tau,
        As=As, ns=ns, **cosmo_kwargs,
        dark_energy_model='EarlyQuintessence',
        m=params['m'], theta_i=params['theta_i'],
        frac_lambda0=params['frac_lambda0'],
        use_zc=False, use_fluid_approximation=True,
        potential_type=1, weighting_factor=10.0,
        oscillation_threshold=1, mH=mH)

    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.NonLinear = model.NonLinear_none
    pars.DoLensing = do_lensing
    pars.set_matter_power(redshifts=[0.0], kmax=10.0)

    results = camb.get_results(pars)

    if do_lensing:
        cls = results.get_lensed_scalar_cls(lmax=lmax, CMB_unit='muK')
    else:
        cls = results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit='muK')
    ell = np.arange(cls.shape[0])

    if verbose:
        s8 = results.get_sigma8_0()
        print(f'  [AxiCAMB] sigma8 = {s8:.6f}')

    return {
        'ell': ell,
        'tt': cls[:, 0],
        'ee': cls[:, 1],
        'te': cls[:, 3],
    }


def run_lcdm_cls(lmax=3000, ombh2=0.022383, omch2=0.12011, H0=67.32,
                 ns=0.96605, As=2.10058e-9, tau=0.0543,
                 do_lensing=True, **cosmo_kwargs):
    """Run LCDM for Cls."""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau,
                       **cosmo_kwargs)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.NonLinear = model.NonLinear_none
    pars.DoLensing = do_lensing
    results = camb.get_results(pars)

    if do_lensing:
        cls = results.get_lensed_scalar_cls(lmax=lmax, CMB_unit='muK')
    else:
        cls = results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit='muK')
    ell = np.arange(cls.shape[0])

    return {
        'ell': ell,
        'tt': cls[:, 0],
        'ee': cls[:, 1],
        'te': cls[:, 3],
    }


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

    print(f'\nRunning LCDM (AxiCAMB)...')
    lcdm_cls_kw = {k: v for k, v in lcdm_kw.items() if k != 'kmax'}
    lcdm = run_lcdm_cls(lmax=args.lmax, do_lensing=do_lensing, **lcdm_cls_kw)

    # Filter keys not accepted by the cls functions
    ae_cls_keys = {'H0', 'ombh2', 'omdah2', 'ns', 'As', 'tau', 'YHe',
                   'omnuh2', 'Neff', 'massive_neutrinos', 'movH_switch'}
    ax_cls_keys = {'H0', 'ombh2', 'omch2_total', 'ns', 'As', 'tau', 'mnu'}

    print('Running LCDM (AxiECAMB, f=1e-6)...')
    ae_lcdm_kw = cosmo_params.get_axiecamb_kwargs(cosmo, {**axion, 'f_ax': 1e-6})
    lcdm_ae = run_axiecamb_cls(axion['m_ax'], 1e-6, lmax=args.lmax,
                                do_lensing=do_lensing,
                                **{k: v for k, v in ae_lcdm_kw.items()
                                   if k in ae_cls_keys})

    print('Running AxiCAMB...')
    ax = run_axicamb_cls(axion['m_ax'], axion['f_ax'], lmax=args.lmax,
                          mH=axion['movH_switch'], do_lensing=do_lensing,
                          **{k: v for k, v in ax_kw.items()
                             if k in ax_cls_keys})

    print('Running AxiECAMB...')
    ae = run_axiecamb_cls(axion['m_ax'], axion['f_ax'], lmax=args.lmax,
                          do_lensing=do_lensing,
                          **{k: v for k, v in ae_kw.items()
                             if k in ae_cls_keys})

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

    # Middle: direct ratio
    a2 = axes[1]
    ratio = f_ax_tt(ell_common) / f_ae_tt(ell_common)
    valid = np.isfinite(ratio)
    a2.plot(ell_common[valid], ratio[valid], 'k-', lw=1)
    a2.axhline(1.0, color='gray', ls=':', alpha=0.5)
    a2.set_ylabel('AxiCAMB / AxiECAMB')
    a2.set_xlim(lmin, lmax_plot)
    a2.set_ylim(0.99, 1.01)
    a2.grid(alpha=0.3)

    # Bottom: ratio to respective LCDM
    a3 = axes[2]
    ratio_ax_lcdm = f_ax_tt(ell_common) / f_lcdm_tt(ell_common)
    ratio_ae_lcdm = f_ae_tt(ell_common) / f_lcdm_ae_tt(ell_common)
    v1 = np.isfinite(ratio_ax_lcdm)
    v2 = np.isfinite(ratio_ae_lcdm)
    a3.plot(ell_common[v1], ratio_ax_lcdm[v1], 'r-', lw=1, label='AxiCAMB / AxiCAMB LCDM')
    a3.plot(ell_common[v2], ratio_ae_lcdm[v2], 'g--', lw=1, label='AxiECAMB / AxiECAMB LCDM')
    a3.axhline(1.0, color='gray', ls=':', alpha=0.5)
    a3.set_xlabel(r'$\ell$')
    a3.set_ylabel(r'$C_\ell^{axion} / C_\ell^{\Lambda CDM}$')
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
