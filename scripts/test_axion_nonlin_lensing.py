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
import matplotlib.pyplot as plt
import sys, os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import axicamb_runner
import cosmo_params as cp

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def run_test(m_ax=1e-24, ax_fraction=0.3, lmax=3000, halofit_version='mead2020',
             accuracy_boost=1, dome_calibrated=False, cosmo=None):
    """Run comparison test with axionHMcode P_NL."""
    if cosmo is None:
        cosmo = {}

    axion = {'m_ax': m_ax, 'f_ax': ax_fraction, 'movH_switch': 50.0}
    lcdm_kw = cp.get_lcdm_kwargs(cosmo)
    ax_kw = cp.get_axicamb_kwargs(cosmo, axion)

    hmcode_mode = "Dome+24" if dome_calibrated else "basic"
    print("=" * 60)
    print(f"External P_NL ratio test (m_ax={m_ax:.0e}, f_ax={ax_fraction}, "
          f"halofit={halofit_version})")
    print(f"axionHMcode mode: {hmcode_mode}")
    print("=" * 60)

    z_grid = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])

    # 1. LCDM with Halofit (reference)
    print(f"\n1. Computing standard CAMB (LCDM) with {halofit_version}...")
    lcdm_nl = axicamb_runner.get_lcdm(
        z_arr=z_grid, nonlinear=True, halofit_version=halofit_version,
        get_cls=True, do_lensing=True, lmax=lmax, **lcdm_kw)

    # 2. Linear lensing (LCDM)
    print("2. Computing linear lensing (LCDM)...")
    lcdm_lin = axicamb_runner.get_lcdm(
        z_arr=z_grid, get_cls=True, do_lensing=True, lmax=lmax, **lcdm_kw)

    # Axion linear and wrong-NL Cls
    print("2b. Computing linear lensing (axion cosmology)...")
    ax_lin = axicamb_runner.run(
        z_arr=z_grid, get_cls=True, do_lensing=True, lmax=lmax, **ax_kw)

    print("2c. Computing axion with standard Halofit (wrong NL model)...")
    ax_wrong_nl = axicamb_runner.run(
        z_arr=z_grid, nonlinear=True, halofit_version=halofit_version,
        get_cls=True, do_lensing=True, lmax=lmax, **ax_kw)

    # 3. CAMB Halofit ratio (verification)
    print("3. Extracting CAMB Halofit ratio...")
    k_h = lcdm_nl['k']
    z_sorted = lcdm_nl['z']
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_camb = np.sqrt(lcdm_nl['pk_nl'] / lcdm_nl['pk'])
        ratio_camb = np.nan_to_num(ratio_camb, nan=1.0, posinf=1.0, neginf=1.0)

    # 4. axionHMcode pipeline
    if ax_fraction <= 0:
        raise ValueError("ax_fraction must be > 0 for this test script")

    print("4. Computing axionHMcode P_NL...")
    ax_hmcode = axicamb_runner.run_with_axionhmcode(
        z_arr=z_grid, dome_calibrated=dome_calibrated,
        lmax=lmax, accuracy_boost=accuracy_boost, **ax_kw)

    comp = axicamb_runner.get_component_spectra(ax_hmcode['linear'])
    k_h_ax = comp['k']

    # 5. Verification: external CAMB ratio
    print("5. Computing lensed Cls with external CAMB Halofit ratio...")
    camb_ext = axicamb_runner.get_lcdm(
        z_arr=z_grid, external_ratio=(k_h, z_sorted, ratio_camb),
        get_cls=True, do_lensing=True, lmax=lmax,
        accuracy_boost=accuracy_boost, **lcdm_kw)

    # ============================================
    # Plots
    # ============================================
    print("\n6. Plotting results...")
    ell = np.arange(lmax + 1)
    suffix = f'_m{m_ax:.0e}_f{ax_fraction}'.replace('+', '')
    suffix += f'_{halofit_version}'
    if dome_calibrated:
        suffix += '_dome'

    # Collect Cls arrays
    cl_std = lcdm_nl['cls']
    cl_lin = lcdm_lin['cls']
    cl_camb_ext = camb_ext['cls']
    cl_ax_ext = ax_hmcode['cls']
    cl_ax_lin = ax_lin['cls']
    cl_ax_hf = ax_wrong_nl['cls']

    # Plot 1: Lensed Cls
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, (ax, label, key) in enumerate(zip(
            axes.flat, ['TT', 'EE', 'BB', 'TE'], ['tt', 'ee', None, 'te'])):
        if key is None:
            ax.set_visible(False)
            continue
        ax.plot(ell[2:], cl_lin[key][2:], 'g-', label='Linear (LCDM)', alpha=0.7, lw=1.5)
        ax.plot(ell[2:], cl_ax_lin[key][2:], 'g--', label='Linear (axion)', alpha=0.7, lw=1.5)
        ax.plot(ell[2:], cl_std[key][2:], 'b-', label=f'LCDM {halofit_version}', alpha=0.8, lw=1.5)
        ax.plot(ell[2:], cl_ax_hf[key][2:], 'm-', label=f'Axion {halofit_version} (wrong NL)', alpha=0.8, lw=1.5)
        ax.plot(ell[2:], cl_camb_ext[key][2:], 'c--', label='External (CAMB ratio)', alpha=0.8, lw=1.5)
        ax.plot(ell[2:], cl_ax_ext[key][2:], 'r:', label='External (axionHMcode)', alpha=0.8, lw=2)
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(f'$D_\\ell^{{{label}}}$ [$\\mu K^2$]')
        ax.set_title(label)
        ax.set_xlim(2, lmax)
        ax.legend(fontsize=8)
        if label in ['TT', 'EE']:
            ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, f'external_ratio_cls{suffix}.png'), dpi=150)
    print(f"   Saved: external_ratio_cls{suffix}.png")
    plt.close()

    # Plot 2: Ratios
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (ax, label, key) in enumerate(zip(axes2.flat, ['TT', 'EE'], ['tt', 'ee'])):
        ref = cl_std[key]
        valid = np.abs(ref) > 1e-10 * np.max(np.abs(ref))
        def ratio_of(cl):
            r = np.ones_like(ref)
            r[valid] = cl[valid] / ref[valid]
            return r

        ax.plot(ell[2:], ratio_of(cl_lin[key])[2:], 'g-', label='Linear (LCDM)', alpha=0.8, lw=1.5)
        ax.plot(ell[2:], ratio_of(cl_ax_lin[key])[2:], 'g--', label='Linear (axion)', alpha=0.8, lw=1.5)
        ax.plot(ell[2:], ratio_of(cl_ax_hf[key])[2:], 'm-', label=f'Axion {halofit_version} (wrong NL)', alpha=0.8, lw=1.5)
        ax.plot(ell[2:], ratio_of(cl_camb_ext[key])[2:], 'c--', label='External (CAMB ratio)', alpha=0.8, lw=1.5)
        ax.plot(ell[2:], ratio_of(cl_ax_ext[key])[2:], 'r:', label='External (axionHMcode)', alpha=0.8, lw=2)
        ax.axhline(1.0, color='b', ls='-', alpha=0.5, lw=1, label=f'LCDM {halofit_version}')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(f'Ratio to CAMB {halofit_version}')
        ax.set_title(label)
        ax.set_xlim(2, lmax)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, f'external_ratio_comparison{suffix}.png'), dpi=150)
    print(f"   Saved: external_ratio_comparison{suffix}.png")
    plt.close()

    # Plot 3: P(k) at z=0
    from scipy.interpolate import interp1d
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    z0 = np.argmin(np.abs(z_sorted))
    hm = ax_hmcode['hmcode']

    ax = axes3[0]
    ax.loglog(k_h, lcdm_nl['pk'][z0], 'g-', label='Linear (LCDM)', alpha=0.7, lw=1.5)
    ax.loglog(k_h, lcdm_nl['pk_nl'][z0], 'b-', label=f'LCDM {halofit_version}', alpha=0.8, lw=1.5)
    ax.loglog(k_h_ax, hm['pk_total'][z0], 'g--', label='Linear (axion)', alpha=0.7, lw=1.5)
    ax.loglog(ax_wrong_nl['k'], ax_wrong_nl['pk_nl'][z0], 'm-',
              label=f'Axion {halofit_version} (wrong NL)', alpha=0.8, lw=1.5)
    ax.loglog(hm['k'], hm['pk_nl'][z0], 'r:', label='Axion HMcode', alpha=0.8, lw=2)
    ax.set_xlabel('$k$ [$h$/Mpc]')
    ax.set_ylabel('$P(k)$ [(Mpc/$h$)$^3$]')
    ax.set_title(f'P(k) at z={z_sorted[z0]:.1f}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes3[1]
    pk_ref = lcdm_nl['pk_nl'][z0]
    valid_k = pk_ref > 0
    ax.semilogx(k_h[valid_k], lcdm_nl['pk'][z0, valid_k] / pk_ref[valid_k],
                'g-', label='Linear (LCDM)', alpha=0.7, lw=1.5)
    f_i = interp1d(k_h_ax, hm['pk_total'][z0], bounds_error=False, fill_value=np.nan)
    v = np.isfinite(f_i(k_h)) & valid_k
    ax.semilogx(k_h[v], f_i(k_h[v]) / pk_ref[v], 'g--', label='Linear (axion)', alpha=0.7, lw=1.5)
    f_i = interp1d(ax_wrong_nl['k'], ax_wrong_nl['pk_nl'][z0], bounds_error=False, fill_value=np.nan)
    v = np.isfinite(f_i(k_h)) & valid_k
    ax.semilogx(k_h[v], f_i(k_h[v]) / pk_ref[v], 'm-',
                label=f'Axion {halofit_version} (wrong NL)', alpha=0.8, lw=1.5)
    f_i = interp1d(hm['k'], hm['pk_nl'][z0], bounds_error=False, fill_value=np.nan)
    v = np.isfinite(f_i(k_h)) & valid_k
    ax.semilogx(k_h[v], f_i(k_h[v]) / pk_ref[v], 'r:', label='Axion HMcode', alpha=0.8, lw=2)
    ax.axhline(1.0, color='b', ls='-', alpha=0.5, lw=1)
    ax.set_xlabel('$k$ [$h$/Mpc]')
    ax.set_ylabel(f'Ratio to LCDM {halofit_version}')
    ax.set_title(f'P(k) Ratio at z={z_sorted[z0]:.1f}')
    ax.legend(fontsize=9, loc='lower left')
    ax.set_ylim(0.5, 1.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, f'pk_comparison{suffix}.png'), dpi=150)
    print(f"   Saved: pk_comparison{suffix}.png")
    plt.close()

    # Plot 4: Components
    comp = axicamb_runner.get_component_spectra(ax_hmcode['linear'])
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes4[0]
    ax.loglog(k_h_ax, comp['pk_total'][z0], 'k-', label='Total', lw=2)
    ax.loglog(k_h_ax, comp['pk_cold'][z0], 'b-', label='Cold', alpha=0.7, lw=1.5)
    ax.loglog(k_h_ax, comp['pk_axion'][z0], 'r--', label='Axion', alpha=0.7, lw=1.5)
    ax.set_xlabel('$k$ [$h$/Mpc]')
    ax.set_ylabel('$P(k)$')
    ax.set_title(f'Components at z={z_sorted[z0]:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes4[1]
    v = comp['pk_cold'][z0] > 0
    ax.semilogx(k_h_ax[v], np.sqrt(comp['pk_axion'][z0, v] / comp['pk_cold'][z0, v]), 'r-', lw=2)
    ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('$k$ [$h$/Mpc]')
    ax.set_ylabel('$T_{axion} / T_{CDM}$')
    ax.set_title(f'Transfer ratio at z={z_sorted[z0]:.1f}')
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, f'pk_components{suffix}.png'), dpi=150)
    print(f"   Saved: pk_components{suffix}.png")
    plt.close()

    # ============================================
    # Summary
    # ============================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for L in [1000, min(2500, lmax)]:
        print(f"\nTT at L={L}:")
        print(f"  CAMB {halofit_version} (ref): {cl_std['tt'][L]:.4f} muK^2")
        print(f"  Linear (LCDM):            {cl_lin['tt'][L]:.4f} muK^2  "
              f"(ratio: {cl_lin['tt'][L]/cl_std['tt'][L]:.6f})")
        print(f"  Linear (axion):           {cl_ax_lin['tt'][L]:.4f} muK^2  "
              f"(ratio: {cl_ax_lin['tt'][L]/cl_std['tt'][L]:.6f})")
        print(f"    -> Axion/LCDM linear:   {cl_ax_lin['tt'][L]/cl_lin['tt'][L]:.6f}")
        print(f"  Axion {halofit_version} (wrong NL): {cl_ax_hf['tt'][L]:.4f} muK^2  "
              f"(ratio: {cl_ax_hf['tt'][L]/cl_std['tt'][L]:.6f})")
        print(f"  External (CAMB ratio):    {cl_camb_ext['tt'][L]:.4f} muK^2  "
              f"(ratio: {cl_camb_ext['tt'][L]/cl_std['tt'][L]:.6f})")
        print(f"  External (axionHMcode):   {cl_ax_ext['tt'][L]:.4f} muK^2  "
              f"(ratio: {cl_ax_ext['tt'][L]/cl_std['tt'][L]:.6f})")

    # Verification
    L_max = min(2500, lmax)
    tt_diff = np.abs(cl_camb_ext['tt'][100:L_max] - cl_std['tt'][100:L_max]) / cl_std['tt'][100:L_max]
    print(f"\nVerification (External CAMB ratio vs CAMB {halofit_version}):")
    print(f"  TT max relative diff (L=100-{L_max}): {np.max(tt_diff):.2e}")
    print(f"  TT mean relative diff:              {np.mean(tt_diff):.2e}")

    # Chi-squared
    print(f"\nChi-squared vs CAMB {halofit_version} (cosmic variance, L=2-{lmax}):")
    ell_chi2 = np.arange(2, lmax + 1)
    weight = (2 * ell_chi2 + 1) / 2.0
    for label, key in [('TT', 'tt'), ('EE', 'ee')]:
        base = cl_std[key][2:lmax+1]
        valid = np.abs(base) > 1e-30
        chi2_lin = np.sum(weight[valid] * (cl_lin[key][2:lmax+1][valid] - base[valid])**2 / base[valid]**2)
        chi2_camb = np.sum(weight[valid] * (cl_camb_ext[key][2:lmax+1][valid] - base[valid])**2 / base[valid]**2)
        chi2_ax = np.sum(weight[valid] * (cl_ax_ext[key][2:lmax+1][valid] - base[valid])**2 / base[valid]**2)
        print(f"  {label}: Linear chi2={chi2_lin:.1f}, External(CAMB) chi2={chi2_camb:.1f}, "
              f"External(axionHMcode) chi2={chi2_ax:.1f} (N_ell={np.sum(valid)})")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cp.add_cli_args(parser)
    parser.add_argument('--ax_fraction', type=float, default=0.3)
    parser.add_argument('--lmax', type=int, default=3000)
    parser.add_argument('--halofit_version', type=str, default='mead2020')
    parser.add_argument('--accuracy_boost', type=int, default=1)
    parser.add_argument('--dome', action='store_true')
    args = parser.parse_args()

    cosmo, axion = cp.from_args(args)
    run_test(m_ax=axion['m_ax'], ax_fraction=args.ax_fraction,
             lmax=args.lmax, halofit_version=args.halofit_version,
             accuracy_boost=args.accuracy_boost, dome_calibrated=args.dome,
             cosmo=cosmo)
