"""
Test external non-linear P(k,z) ratio support with axionHMcode.

This script:
1. Computes lensed Cls using standard CAMB (Halofit)
2. Computes lensed Cls using external ratio from CAMB's own Halofit (verification)
3. Computes lensed Cls using external ratio from axionHMcode
4. Plots comparison and ratios

When ax_fraction > 0, uses proper axion cosmology via EarlyQuintessence.
"""

import numpy as np
import camb
from camb import model
from camb.results import CAMBdata
from camb.axion_utils import get_axion_phi_i, get_omega_ax_h2
import matplotlib.pyplot as plt
import sys
import argparse

# Add axionHMcode to path
sys.path.insert(0, '/Users/adammoss/work/code/axionHMcode')


def run_test(m_ax=1e-25, ax_fraction=0.0, lmax=3000, halofit_version='mead2020'):
    """
    Run comparison test with axionHMcode P_NL.
    """
    from halo_model import HMcode_params, PS_nonlin_cold, PS_nonlin_axion
    from axion_functions import axion_params
    from cosmology.overdensities import func_D_z_unnorm_int

    print("=" * 60)
    print(f"External P_NL ratio test (m_ax={m_ax:.0e}, f_ax={ax_fraction}, halofit={halofit_version})")
    print("=" * 60)

    # Cosmology
    H0, ombh2, omch2, As, ns = 67.5, 0.022, 0.122, 2.1e-9, 0.965
    h = H0 / 100
    z_grid = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])

    # ============================================
    # Set up axion cosmology if f_ax > 0
    # ============================================
    axion_params_dict = None
    if ax_fraction > 0:
        print(f"\n0. Finding axion initial conditions for f_ax={ax_fraction}...")
        axion_params_dict = get_axion_phi_i(
            h=h,
            ombh2=ombh2,
            omch2_total=omch2,
            f_ax=ax_fraction,
            mass_ev=m_ax,
            verbose=True,
        )
        if axion_params_dict is None:
            raise RuntimeError("Failed to find axion initial conditions")
        print(f"   theta_i = {axion_params_dict['theta_i']:.6f}")
        print(f"   frac_lambda0 = {axion_params_dict['frac_lambda0']:.6f}")

    def make_cosmos_dict(z, omega_ax_h2_actual=None):
        """Make cosmology dict for axionHMcode."""
        total_dark = omch2
        if ax_fraction == 0:
            omega_ax, omega_d = 1e-20, total_dark
        else:
            # Use actual axion density from CAMB if available
            if omega_ax_h2_actual is not None:
                omega_ax = omega_ax_h2_actual
            else:
                omega_ax = total_dark * ax_fraction
            omega_d = total_dark - omega_ax

        cosmos = {
            'omega_b_0': ombh2, 'omega_d_0': omega_d, 'omega_ax_0': omega_ax,
            'omega_m_0': ombh2 + total_dark,
            'Omega_b_0': ombh2 / h**2, 'Omega_d_0': omega_d / h**2,
            'Omega_ax_0': omega_ax / h**2, 'Omega_m_0': (ombh2 + total_dark) / h**2,
            'omega_db_0': omega_d + ombh2, 'Omega_db_0': (omega_d + ombh2) / h**2,
            'h': h, 'ns': ns, 'As': As, 'z': z, 'm_ax': m_ax,
            'M_min': 7, 'M_max': 18, 'k_piv': 0.05, 'version': 'basic',
        }
        cosmos['Omega_w_0'] = 1 - cosmos['Omega_m_0']
        cosmos['G_a'] = func_D_z_unnorm_int(z, cosmos['Omega_m_0'], cosmos['Omega_w_0'])
        return cosmos

    # ============================================
    # 1. Standard CAMB with Halofit (reference) - LCDM for comparison
    # ============================================
    print(f"\n1. Computing standard CAMB (LCDM) with {halofit_version}...")
    params = camb.CAMBparams()
    params.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
    params.InitPower.set_params(As=As, ns=ns)
    params.set_for_lmax(lmax, lens_potential_accuracy=1)
    params.DoLensing = True
    params.NonLinear = camb.model.NonLinear_both
    params.NonLinearModel.halofit_version = halofit_version
    params.set_matter_power(redshifts=list(z_grid[::-1]), kmax=50.0)

    results_standard = camb.get_results(params)
    cls_standard = results_standard.get_lensed_scalar_cls(lmax=lmax, CMB_unit='muK')

    # ============================================
    # 2. Linear lensing (no non-linear corrections)
    # ============================================
    print("2. Computing linear lensing (LCDM)...")
    params_linear = params.copy()
    params_linear.NonLinear = camb.model.NonLinear_none
    results_linear = camb.get_results(params_linear)
    cls_linear = results_linear.get_lensed_scalar_cls(lmax=lmax, CMB_unit='muK')

    # Also compute axion linear Cls if f_ax > 0
    cls_axion_linear = None
    if ax_fraction > 0:
        print("2b. Computing linear lensing (axion cosmology)...")
        omch2_cdm = (1 - ax_fraction) * omch2
        omch2_cdm = max(omch2_cdm, 1e-7)

        params_axion_linear = camb.set_params(
            H0=H0,
            ombh2=ombh2,
            omch2=omch2_cdm,
            omk=0,
            tau=0.05,
            As=As,
            ns=ns,
            dark_energy_model='EarlyQuintessence',
            m=axion_params_dict['m'],
            theta_i=axion_params_dict['theta_i'],
            frac_lambda0=axion_params_dict['frac_lambda0'],
            use_zc=False,
            use_fluid_approximation=True,
            potential_type=1,
        )
        params_axion_linear.set_for_lmax(lmax, lens_potential_accuracy=1)
        params_axion_linear.DoLensing = True
        params_axion_linear.NonLinear = camb.model.NonLinear_none
        results_axion_linear = camb.get_results(params_axion_linear)
        cls_axion_linear = results_axion_linear.get_lensed_scalar_cls(lmax=lmax, CMB_unit='muK')

    # ============================================
    # 3. Get P_L and P_NL from CAMB for external ratio
    # ============================================
    print("3. Extracting CAMB Halofit ratio...")
    k_h, z_pk, pk_lin_all = results_standard.get_linear_matter_power_spectrum(
        hubble_units=True, k_hunit=True, nonlinear=False
    )
    _, _, pk_nl_all = results_standard.get_nonlinear_matter_power_spectrum(
        hubble_units=True, k_hunit=True
    )

    # Sort by ascending z
    sort_idx = np.argsort(z_pk)
    z_sorted = z_pk[sort_idx]
    pk_lin_sorted = pk_lin_all[sort_idx, :]
    pk_nl_sorted = pk_nl_all[sort_idx, :]

    # Compute CAMB Halofit ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_camb = np.sqrt(pk_nl_sorted / pk_lin_sorted)
        ratio_camb = np.nan_to_num(ratio_camb, nan=1.0, posinf=1.0, neginf=1.0)

    # ============================================
    # 3b. Get axion linear P(k) if f_ax > 0
    # ============================================
    pk_lin_axion = None
    results_axion_lin = None
    if ax_fraction > 0:
        print("3b. Computing axion linear P(k) from EarlyQuintessence...")
        omch2_cdm = (1 - ax_fraction) * omch2
        omch2_cdm = max(omch2_cdm, 1e-7)

        params_axion_pk = camb.set_params(
            H0=H0,
            ombh2=ombh2,
            omch2=omch2_cdm,
            omk=0,
            tau=0.05,
            As=As,
            ns=ns,
            dark_energy_model='EarlyQuintessence',
            m=axion_params_dict['m'],
            theta_i=axion_params_dict['theta_i'],
            frac_lambda0=axion_params_dict['frac_lambda0'],
            use_zc=False,
            use_fluid_approximation=True,
            potential_type=1,
        )
        params_axion_pk.set_for_lmax(lmax, lens_potential_accuracy=1)
        params_axion_pk.DoLensing = True
        params_axion_pk.NonLinear = camb.model.NonLinear_none  # Linear only
        params_axion_pk.set_matter_power(redshifts=list(z_grid[::-1]), kmax=50.0)

        results_axion_lin = camb.get_results(params_axion_pk)

        # Get linear power spectrum from axion cosmology
        k_h_ax, z_pk_ax, pk_lin_ax_all = results_axion_lin.get_linear_matter_power_spectrum(
            hubble_units=True, k_hunit=True, nonlinear=False
        )

        # Sort by ascending z
        sort_idx_ax = np.argsort(z_pk_ax)
        pk_lin_axion = pk_lin_ax_all[sort_idx_ax, :]

        print(f"   Got P_lin(k) for {len(z_sorted)} redshifts")

    # ============================================
    # 4. Compute axionHMcode P_NL
    # ============================================
    print("4. Computing axionHMcode P_NL...")
    M_arr = np.logspace(7, 18, 100)
    pk_nl_axion_hm = np.zeros_like(pk_lin_sorted)

    for i, zi in enumerate(z_sorted):
        if ax_fraction > 0:
            # Use axion linear P(k) from CAMB
            pk_lin_z = pk_lin_axion[i, :]
            omega_ax_h2 = axion_params_dict['omega_ax_h2']
            cosmos_z = make_cosmos_dict(zi, omega_ax_h2_actual=omega_ax_h2)
        else:
            pk_lin_z = pk_lin_sorted[i, :]
            cosmos_z = make_cosmos_dict(zi)

        hmcode_params_z = HMcode_params.HMCode_param_dic(cosmos_z, k_h, pk_lin_z)

        if ax_fraction > 0:
            # For axion cosmology, we need to estimate cold vs axion power
            # Simple approximation: cold traces total on large scales,
            # axion is suppressed on small scales
            # TODO: Get proper transfer functions from CAMB
            f_cold = (1 - ax_fraction)
            f_ax = ax_fraction

            power_spec_dic_z = {
                'k': k_h,
                'power_cold': pk_lin_z * f_cold,  # Approximate cold component
                'power_axion': pk_lin_z * f_ax,   # Approximate axion component
                'power_total': pk_lin_z,
            }
            axion_param_z = axion_params.func_axion_param_dic(
                M_arr, cosmos_z, power_spec_dic_z, hmcode_params_z, concentration_param=False
            )
            PS_z = PS_nonlin_axion.func_full_halo_model_ax(
                M_arr, power_spec_dic_z, cosmos_z, hmcode_params_z, axion_param_z,
                alpha=False, eta_given=False, one_halo_damping=True,
                two_halo_damping=False, concentration_param=False, full_2h=False
            )
        else:
            PS_z = PS_nonlin_cold.func_non_lin_PS_matter(
                M_arr, k_h, pk_lin_z, cosmos_z, hmcode_params_z, cosmos_z['Omega_m_0'],
                alpha=False, eta_given=False, one_halo_damping=True,
                two_halo_damping=False, concentration_param=False, full_2h=False
            )

        pk_nl_axion_hm[i, :] = PS_z[0]
        print(f"   z={zi:.1f}: done")

    # Compute axionHMcode ratio
    # Use axion linear P(k) as denominator if available
    pk_lin_for_ratio = pk_lin_axion if ax_fraction > 0 else pk_lin_sorted
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_axion = np.sqrt(pk_nl_axion_hm / pk_lin_for_ratio)
        ratio_axion = np.nan_to_num(ratio_axion, nan=1.0, posinf=1.0, neginf=1.0)

    # ============================================
    # 5. Compute lensed Cls with external CAMB ratio (verification)
    # ============================================
    print("5. Computing lensed Cls with external CAMB Halofit ratio...")
    params_ext = camb.CAMBparams()
    params_ext.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
    params_ext.InitPower.set_params(As=As, ns=ns)
    params_ext.set_for_lmax(lmax, lens_potential_accuracy=1)
    params_ext.DoLensing = True
    params_ext.NonLinear = camb.model.NonLinear_lens
    params_ext.set_matter_power(redshifts=list(z_sorted[::-1]), kmax=50.0)

    results_camb_ext = CAMBdata()
    results_camb_ext.set_nonlin_ratio(k_h, z_sorted, ratio_camb.T)
    results_camb_ext.calc_power_spectra(params_ext)
    cls_camb_ext = results_camb_ext.get_lensed_scalar_cls(lmax=lmax, CMB_unit='muK')

    # ============================================
    # 6. Compute lensed Cls with external axionHMcode ratio
    # ============================================
    print("6. Computing lensed Cls with external axionHMcode ratio...")
    if ax_fraction > 0:
        # Use axion cosmology for CMB calculation - create fresh params
        omch2_cdm = (1 - ax_fraction) * omch2
        omch2_cdm = max(omch2_cdm, 1e-7)

        params_axion_ext = camb.set_params(
            H0=H0,
            ombh2=ombh2,
            omch2=omch2_cdm,
            omk=0,
            tau=0.05,
            As=As,
            ns=ns,
            dark_energy_model='EarlyQuintessence',
            m=axion_params_dict['m'],
            theta_i=axion_params_dict['theta_i'],
            frac_lambda0=axion_params_dict['frac_lambda0'],
            use_zc=False,
            use_fluid_approximation=True,
            potential_type=1,
        )
        params_axion_ext.set_for_lmax(lmax, lens_potential_accuracy=1)
        params_axion_ext.DoLensing = True
        params_axion_ext.NonLinear = camb.model.NonLinear_lens
        params_axion_ext.set_matter_power(redshifts=list(z_sorted[::-1]), kmax=50.0)
    else:
        params_axion_ext = params_ext

    results_axion_ext = CAMBdata()
    results_axion_ext.set_nonlin_ratio(k_h, z_sorted, ratio_axion.T)
    results_axion_ext.calc_power_spectra(params_axion_ext)
    cls_axion_ext = results_axion_ext.get_lensed_scalar_cls(lmax=lmax, CMB_unit='muK')

    # ============================================
    # Plot results
    # ============================================
    print("\n7. Plotting results...")
    ell = np.arange(lmax + 1)
    suffix = f'_m{m_ax:.0e}_f{ax_fraction}'.replace('+', '') if ax_fraction > 0 else '_LCDM'
    suffix += f'_{halofit_version}'

    # Plot 1: Lensed Cls
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    labels = ['TT', 'EE', 'BB', 'TE']

    for i, (ax, label) in enumerate(zip(axes.flat, labels)):
        ax.plot(ell[2:], cls_linear[2:, i], 'g-', label='Linear (LCDM)', alpha=0.7, lw=1.5)
        if cls_axion_linear is not None:
            ax.plot(ell[2:], cls_axion_linear[2:, i], 'g--', label='Linear (axion)', alpha=0.7, lw=1.5)
        ax.plot(ell[2:], cls_standard[2:, i], 'b-', label=f'LCDM {halofit_version}', alpha=0.8, lw=1.5)
        ax.plot(ell[2:], cls_camb_ext[2:, i], 'c--', label='External (CAMB ratio)', alpha=0.8, lw=1.5)
        ax.plot(ell[2:], cls_axion_ext[2:, i], 'r:', label='External (axionHMcode)', alpha=0.8, lw=2)
        ax.set_xlabel('$\\ell$')
        ax.set_ylabel(f'$D_\\ell^{{{label}}}$ [$\\mu K^2$]')
        ax.set_title(label)
        ax.set_xlim(2, lmax)
        ax.legend(fontsize=8)
        if label in ['TT', 'EE', 'BB']:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'external_ratio_cls{suffix}.png', dpi=150)
    print(f"   Saved: external_ratio_cls{suffix}.png")
    plt.close()

    # Plot 2: Ratios relative to standard CAMB Halofit
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (ax, label) in enumerate(zip(axes2.flat, ['TT', 'EE'])):
        ref_cl = cls_standard[:, idx]
        valid = np.abs(ref_cl) > 1e-10 * np.max(np.abs(ref_cl))

        # Linear LCDM / Standard
        ratio_lin = np.ones_like(cls_linear[:, idx])
        ratio_lin[valid] = cls_linear[valid, idx] / ref_cl[valid]

        # Linear axion / Standard (if available)
        ratio_axion_lin = None
        if cls_axion_linear is not None:
            ratio_axion_lin = np.ones_like(cls_axion_linear[:, idx])
            ratio_axion_lin[valid] = cls_axion_linear[valid, idx] / ref_cl[valid]

        # External CAMB ratio / Standard
        ratio_camb_ext = np.ones_like(cls_camb_ext[:, idx])
        ratio_camb_ext[valid] = cls_camb_ext[valid, idx] / ref_cl[valid]

        # External axionHMcode / Standard
        ratio_axion_ext = np.ones_like(cls_axion_ext[:, idx])
        ratio_axion_ext[valid] = cls_axion_ext[valid, idx] / ref_cl[valid]

        ax.plot(ell[2:], ratio_lin[2:], 'g-', label='Linear (LCDM)', alpha=0.8, lw=1.5)
        if ratio_axion_lin is not None:
            ax.plot(ell[2:], ratio_axion_lin[2:], 'g--', label='Linear (axion)', alpha=0.8, lw=1.5)
        ax.plot(ell[2:], ratio_camb_ext[2:], 'c--', label='External (CAMB ratio)', alpha=0.8, lw=1.5)
        ax.plot(ell[2:], ratio_axion_ext[2:], 'r:', label='External (axionHMcode)', alpha=0.8, lw=2)
        ax.axhline(1.0, color='b', linestyle='-', alpha=0.5, lw=1, label=f'LCDM {halofit_version}')

        ax.set_xlabel('$\\ell$')
        ax.set_ylabel(f'$C_\\ell^{{{label}}}$ / $C_\\ell^{{{label}, \\mathrm{{{halofit_version}}}}}$')
        ax.set_title(f'{label}: Ratio to CAMB {halofit_version}')
        ax.set_xlim(2, lmax)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'external_ratio_comparison{suffix}.png', dpi=150)
    print(f"   Saved: external_ratio_comparison{suffix}.png")
    plt.close()

    # ============================================
    # Print summary
    # ============================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Print summary at specific L values
    L_summary = [1000, min(2500, lmax)]
    for L in L_summary:
        print(f"\nTT at L={L}:")
        print(f"  CAMB {halofit_version} (ref): {cls_standard[L, 0]:.4f} muK^2")
        print(f"  Linear:                   {cls_linear[L, 0]:.4f} muK^2  (ratio: {cls_linear[L, 0]/cls_standard[L, 0]:.6f})")
        print(f"  External (CAMB ratio):    {cls_camb_ext[L, 0]:.4f} muK^2  (ratio: {cls_camb_ext[L, 0]/cls_standard[L, 0]:.6f})")
        print(f"  External (axionHMcode):   {cls_axion_ext[L, 0]:.4f} muK^2  (ratio: {cls_axion_ext[L, 0]/cls_standard[L, 0]:.6f})")

    # Verification check
    L_verify_max = min(2500, lmax)
    tt_diff = np.abs(cls_camb_ext[100:L_verify_max, 0] - cls_standard[100:L_verify_max, 0]) / cls_standard[100:L_verify_max, 0]
    print(f"\nVerification (External CAMB ratio vs CAMB {halofit_version}):")
    print(f"  TT max relative diff (L=100-{L_verify_max}): {np.max(tt_diff):.2e}")
    print(f"  TT mean relative diff:              {np.mean(tt_diff):.2e}")

    # Chi-squared vs CAMB Halofit assuming cosmic variance
    # Cosmic variance: Var(C_l) = 2/(2l+1) * C_l^2
    # chi2 = sum_l (C_l^model - C_l^baseline)^2 * (2l+1) / (2 * C_l^baseline^2)
    print(f"\nChi-squared vs CAMB {halofit_version} (cosmic variance, L=2-{lmax}):")
    ell_chi2 = np.arange(2, lmax + 1)

    for i, label in enumerate(['TT', 'EE']):
        baseline = cls_standard[2:lmax + 1, i]
        linear_cl = cls_linear[2:lmax + 1, i]
        camb_ext_cl = cls_camb_ext[2:lmax + 1, i]
        axion_ext_cl = cls_axion_ext[2:lmax + 1, i]

        # Avoid division by zero
        valid = np.abs(baseline) > 1e-30

        # Cosmic variance weight: (2l+1)/2
        weight = (2 * ell_chi2 + 1) / 2.0

        chi2_linear = np.sum(weight[valid] * (linear_cl[valid] - baseline[valid])**2 / baseline[valid]**2)
        chi2_camb_ext = np.sum(weight[valid] * (camb_ext_cl[valid] - baseline[valid])**2 / baseline[valid]**2)
        chi2_axion = np.sum(weight[valid] * (axion_ext_cl[valid] - baseline[valid])**2 / baseline[valid]**2)

        n_ell = np.sum(valid)
        print(f"  {label}: Linear chi2={chi2_linear:.1f}, External(CAMB) chi2={chi2_camb_ext:.1f}, External(axionHMcode) chi2={chi2_axion:.1f} (N_ell={n_ell})")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_ax', type=float, default=1e-25, help='Axion mass in eV')
    parser.add_argument('--ax_fraction', type=float, default=0.0, help='Axion fraction (0=LCDM)')
    parser.add_argument('--lmax', type=int, default=3000, help='Maximum multipole')
    parser.add_argument('--halofit_version', type=str, default='mead2020',
                        help='CAMB halofit version (default: mead2020)')
    args = parser.parse_args()

    run_test(m_ax=args.m_ax, ax_fraction=args.ax_fraction, lmax=args.lmax,
             halofit_version=args.halofit_version)
