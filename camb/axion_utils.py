"""
Utility to find the initial axion field value (theta_i) that produces
a target axion dark matter fraction.
"""

import numpy as np
from scipy.optimize import brentq
import camb


def get_omega_ax_h2(results):
    """
    Get the axion density Omega_ax * h^2 at z=0.

    This extracts the axion contribution by subtracting the cosmological
    constant piece from the total dark energy density.

    Parameters
    ----------
    results : CAMBdata
        CAMB results object from a run with EarlyQuintessence

    Returns
    -------
    float
        Omega_ax * h^2 at z=0
    """
    h = results.hubble_parameter(0) / 100.0

    # Get dark energy density at a=1 (z=0)
    rho_de, _ = results.get_dark_energy_rho_w(1.0)

    # Get the cosmological constant contribution
    frac_lambda0 = results.Params.DarkEnergy.frac_lambda0
    rho_lambda = frac_lambda0 * results.grhov

    # Axion density is total DE minus CC (scaled by a^3 for matter-like behavior at late times)
    # Note: at z=0 (a=1), no scaling needed
    rho_ax = rho_de - rho_lambda

    # Convert to Omega_ax * h^2
    omega_ax_h2 = (rho_ax / results.grhocrit) * h**2

    return omega_ax_h2


def get_axion_phi_i(
    h=0.67,
    ombh2=0.022,
    omch2_total=0.12,
    f_ax=0.1,
    mass_ev=1e-25,
    mH=50.0,
    use_PH=True,
    weighting_factor=10.0,
    oscillation_threshold=1,
    accuracy=1,
    tol=1e-6,
    max_iter=100,
    verbose=True,
):
    """
    Find the initial axion field value theta_i that gives a target axion fraction.

    Parameters
    ----------
    h : float
        Hubble parameter H0/100
    ombh2 : float
        Baryon density Omega_b * h^2
    omch2_total : float
        Total cold dark matter density (CDM + axion) Omega_c * h^2
    f_ax : float
        Target axion fraction f_ax = Omega_ax / (Omega_ax + Omega_cdm)
    mass_ev : float
        Axion mass in eV
    mH : float
        Mass scale for PH potential (if use_PH=True)
    use_PH : bool
        Whether to use Peccei-Quinn-Higgs potential
    weighting_factor : float
        Weighting factor for shooting method
    oscillation_threshold : int
        Threshold for oscillation detection
    accuracy : int
        CAMB accuracy boost level
    tol : float
        Tolerance for Brent's method (absolute tolerance on theta_i)
    max_iter : int
        Maximum iterations for bracketing stage
    verbose : bool
        Print progress information

    Returns
    -------
    dict or None
        Dictionary containing:
        - 'theta_i': Initial field value
        - 'frac_lambda0': Fraction of DE that is cosmological constant
        - 'm': Axion mass in CAMB units (reduced Planck mass)
        - 'omega_ax_h2': Achieved axion density
        - 'params': CAMBparams object ready for get_results()
        Returns None if convergence failed.
    """
    # Convert mass to CAMB units (reduced Planck mass units)
    # M_pl = sqrt(hbar*c / 8*pi*G) = 2.435323e27 eV
    # m_camb = m_eV / M_pl
    ev_to_reduced_planck = 4.106231e-28
    m_camb = mass_ev * ev_to_reduced_planck

    if verbose:
        print(f"m_camb = {m_camb:.6e}")

    # Target axion density
    omega_ax_h2_target = f_ax * omch2_total

    # CDM density (non-axion part)
    omch2_cdm = (1.0 - f_ax) * omch2_total
    # Use small non-zero value to avoid numerical issues
    omch2_cdm = max(omch2_cdm, 1e-7)

    if verbose:
        print(f"Target: omega_ax_h2 = {omega_ax_h2_target:.6e}")
        print(f"CDM:    omch2_cdm   = {omch2_cdm:.6e}")

    # First, run LCDM to get reference values and compute frac_lambda0
    pars_lcdm = camb.set_params(
        H0=h * 100,
        ombh2=ombh2,
        omch2=omch2_total,
        omk=0,
        tau=0.05,
        As=2.196e-9,
        ns=0.9655,
    )
    results_lcdm = camb.get_background(pars_lcdm)

    # frac_lambda0 = fraction of DE that is CC (vs axion contribution)
    # At z=0: Omega_DE = Omega_Lambda + Omega_ax
    # We want Omega_ax = f_ax * omch2_total / h^2
    # So Omega_Lambda = Omega_DE_LCDM (stays the same)
    # frac_lambda0 = Omega_Lambda / (Omega_Lambda + Omega_ax)
    omega_de_lcdm = results_lcdm.grhov / results_lcdm.grhocrit
    omega_ax_target = omega_ax_h2_target / h**2
    frac_lambda0 = omega_de_lcdm / (omega_de_lcdm + omega_ax_target)

    if verbose:
        print(f"frac_lambda0 = {frac_lambda0:.6f}")

    def run_axion_background(theta_i):
        """Run CAMB with given theta_i and return results."""
        pars = camb.set_params(
            H0=h * 100,
            ombh2=ombh2,
            omch2=omch2_cdm,
            omk=0,
            tau=0.05,
            As=2.196e-9,
            ns=0.9655,
            dark_energy_model='EarlyQuintessence',
            m=m_camb,
            theta_i=theta_i,
            frac_lambda0=frac_lambda0,
            use_zc=False,
            mH=mH,
            use_PH=use_PH,
            use_fluid_approximation=True,
            potential_type=1,
            weighting_factor=weighting_factor,
            oscillation_threshold=oscillation_threshold,
        )
        pars.set_accuracy(AccuracyBoost=accuracy)
        return camb.get_background(pars)

    def run_axion_background_fast(theta_i):
        """Run CAMB with given theta_i at low accuracy for quick probing."""
        pars = camb.set_params(
            H0=h * 100,
            ombh2=ombh2,
            omch2=omch2_cdm,
            omk=0,
            tau=0.05,
            As=2.196e-9,
            ns=0.9655,
            dark_energy_model='EarlyQuintessence',
            m=m_camb,
            theta_i=theta_i,
            frac_lambda0=frac_lambda0,
            use_zc=False,
            mH=mH,
            use_PH=use_PH,
            use_fluid_approximation=True,
            potential_type=1,
            weighting_factor=weighting_factor,
            oscillation_threshold=oscillation_threshold,
        )
        pars.set_accuracy(AccuracyBoost=1)  # Low accuracy for speed
        return camb.get_background(pars)

    def objective(theta_i):
        """Objective function: returns omega_ax_h2 - target (zero at solution)."""
        results = run_axion_background(theta_i)
        return get_omega_ax_h2(results) - omega_ax_h2_target

    # Stage 1: Quick probe to estimate theta_i using scaling omega_ax ∝ theta_i²
    if verbose:
        print("\nStage 1: Estimating theta_i from probe...")

    theta_probe = 0.1  # Reference probe value
    try:
        results_probe = run_axion_background_fast(theta_probe)
        omega_probe = get_omega_ax_h2(results_probe)

        # Estimate using scaling: omega_ax ∝ theta_i²
        # theta_target / theta_probe = sqrt(omega_target / omega_probe)
        theta_estimate = theta_probe * np.sqrt(omega_ax_h2_target / omega_probe)

        if verbose:
            print(f"  Probe: theta_i={theta_probe:.4e} -> omega_ax_h2={omega_probe:.6e}")
            print(f"  Estimated theta_i = {theta_estimate:.4e}")

        # Set bracket around estimate (factor of 2 each side for safety)
        theta_lower = theta_estimate / 2.0
        theta_upper = theta_estimate * 2.0

    except Exception as e:
        if verbose:
            print(f"  Probe failed ({e}), falling back to exponential search...")

        # Fallback to exponential search
        theta_i = 1e-4
        factor = 2.0

        for i in range(max_iter):
            theta_i *= factor

            try:
                results = run_axion_background_fast(theta_i)
                omega_ax_h2 = get_omega_ax_h2(results)
            except Exception as e2:
                if verbose:
                    print(f"  theta_i={theta_i:.4e}: failed ({e2})")
                continue

            if omega_ax_h2 > omega_ax_h2_target:
                if verbose:
                    print(f"  Bracketed at theta_i={theta_i:.4e}")
                break
        else:
            print("Failed to bracket solution (stage 1)")
            return None

        theta_lower = theta_i / factor
        theta_upper = theta_i

    # Stage 2: Brent's method for root finding
    if verbose:
        print(f"\nStage 2: Brent's method [{theta_lower:.4e}, {theta_upper:.4e}]...")

    try:
        theta_i = brentq(objective, theta_lower, theta_upper, xtol=tol)

        # Get final result for reporting
        results = run_axion_background(theta_i)
        omega_ax_h2 = get_omega_ax_h2(results)
        rel_error = abs(omega_ax_h2 - omega_ax_h2_target) / omega_ax_h2_target

        if verbose:
            print(f"\nConverged: theta_i = {theta_i:.8e}")
            print(f"           omega_ax_h2 = {omega_ax_h2:.6e} (target: {omega_ax_h2_target:.6e})")
            print(f"           rel_error = {rel_error:.2e}")

        # Build final CAMBparams object
        pars = camb.set_params(
            H0=h * 100,
            ombh2=ombh2,
            omch2=omch2_cdm,
            omk=0,
            tau=0.05,
            As=2.196e-9,
            ns=0.9655,
            dark_energy_model='EarlyQuintessence',
            m=m_camb,
            theta_i=theta_i,
            frac_lambda0=frac_lambda0,
            use_zc=False,
            mH=mH,
            use_PH=use_PH,
            use_fluid_approximation=True,
            potential_type=1,
            weighting_factor=weighting_factor,
            oscillation_threshold=oscillation_threshold,
        )
        pars.set_accuracy(AccuracyBoost=accuracy)

        return {
            'theta_i': theta_i,
            'frac_lambda0': frac_lambda0,
            'm': m_camb,
            'omega_ax_h2': omega_ax_h2,
            'params': pars,
        }

    except ValueError as e:
        print(f"Brent's method failed: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    result = get_axion_phi_i(
        h=0.6736,
        ombh2=0.0224,
        omch2_total=0.12,
        f_ax=0.05,  # 5% axion fraction
        mass_ev=1e-25,
        verbose=True,
    )

    if result is not None:
        print(f"\nResult dict keys: {list(result.keys())}")
        print(f"theta_i = {result['theta_i']}")
        print(f"frac_lambda0 = {result['frac_lambda0']}")
        print(f"m = {result['m']}")
        print(f"omega_ax_h2 = {result['omega_ax_h2']}")
        print(f"params = {type(result['params'])}")
