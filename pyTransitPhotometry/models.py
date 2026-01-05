"""
Transit model fitting using batman.

Implements:
- Batman transit model integration
- Parameter fitting with scipy.optimize.curve_fit
- Error estimation from covariance matrix
- Physical parameter derivation
"""

import numpy as np
from scipy.optimize import curve_fit
import warnings
from typing import Dict, Tuple, Optional

try:
    import batman
    BATMAN_AVAILABLE = True
except ImportError:
    BATMAN_AVAILABLE = False
    warnings.warn(
        "batman-package not installed. Transit fitting unavailable. "
        "Install with: pip install batman-package"
    )


def batman_transit_model(
    times: np.ndarray,
    t0: float,
    period: float,
    rp: float,
    a: float,
    inc: float,
    ecc: float = 0.0,
    w: float = 90.0,
    u1: float = 0.4,
    u2: float = 0.26,
    limb_dark: str = "quadratic"
) -> np.ndarray:
    """
    Generate batman transit model light curve.
    
    Parameters
    ----------
    times : np.ndarray
        Observation times (any consistent unit, typically MJD or BJD)
    t0 : float
        Transit center time
    period : float
        Orbital period (same units as times)
    rp : float
        Planet-to-star radius ratio (Rp/Rs)
    a : float
        Scaled semi-major axis (a/Rs)
    inc : float
        Orbital inclination (degrees)
    ecc : float, optional
        Eccentricity (default: 0.0 for circular orbit)
    w : float, optional
        Argument of periastron (degrees, default: 90.0)
    u1, u2 : float, optional
        Quadratic limb darkening coefficients (default: 0.4, 0.26)
    limb_dark : str, optional
        Limb darkening law (default: 'quadratic')
    
    Returns
    -------
    model_flux : np.ndarray
        Normalized transit light curve (1.0 = out of transit)
    
    Notes
    -----
    Uses batman (Kreidberg 2015) for fast, accurate transit computation.
    
    Transit depth: δ = (Rp/Rs)²
    Impact parameter: b = (a/Rs) × cos(i)
    
    Examples
    --------
    >>> model = batman_transit_model(
    ...     times, t0=2460000.5, period=2.48,
    ...     rp=0.103, a=7.2, inc=82.0
    ... )
    """
    if not BATMAN_AVAILABLE:
        raise ImportError("batman-package required for transit modeling")
    
    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = w
    params.u = [u1, u2]
    params.limb_dark = limb_dark
    
    m = batman.TransitModel(params, times)
    return m.light_curve(params)


class TransitFitter:
    """
    Fit transit model to light curve data.
    
    Parameters
    ----------
    period : float
        Orbital period (fixed, from literature or previous analysis)
    t0_guess : float
        Initial guess for transit center time
    limb_dark_u1, limb_dark_u2 : float, optional
        Limb darkening coefficients (default: 0.4, 0.26 for V-band)
        Typically from stellar atmosphere models
    
    Examples
    --------
    >>> fitter = TransitFitter(
    ...     period=2.4842,
    ...     t0_guess=2460235.752,
    ...     limb_dark_u1=0.40,
    ...     limb_dark_u2=0.26
    ... )
    >>> result = fitter.fit(
    ...     times, fluxes, errors,
    ...     fit_params=['rp', 'a', 'inc', 'baseline', 'slope'],
    ...     fix_t0=True
    ... )
    """
    
    def __init__(
        self,
        period: float,
        t0_guess: float,
        limb_dark_u1: float = 0.40,
        limb_dark_u2: float = 0.26,
        ecc: float = 0.0,
        w: float = 90.0
    ):
        if not BATMAN_AVAILABLE:
            raise ImportError("batman-package required for transit fitting")
        
        self.period = period
        self.t0_guess = t0_guess
        self.limb_dark_u1 = limb_dark_u1
        self.limb_dark_u2 = limb_dark_u2
        self.ecc = ecc
        self.w = w
    
    def model_with_detrending(
        self,
        times: np.ndarray,
        rp: float,
        a: float,
        inc: float,
        baseline: float,
        slope: float,
        t0: Optional[float] = None
    ) -> np.ndarray:
        """
        Transit model with linear detrending.
        
        Model: F = baseline × transit(t) × (1 + slope × (t - t_mean))
        """
        if t0 is None:
            t0 = self.t0_guess
        
        t_mean = np.mean(times)
        
        # Generate transit model
        transit = batman_transit_model(
            times, t0, self.period, rp, a, inc,
            ecc=self.ecc, w=self.w,
            u1=self.limb_dark_u1, u2=self.limb_dark_u2
        )
        
        # Apply baseline and linear detrending
        detrend = 1.0 + slope * (times - t_mean)
        model = baseline * transit * detrend
        
        return model
    
    def fit(
        self,
        times: np.ndarray,
        fluxes: np.ndarray,
        errors: np.ndarray,
        initial_params: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        fix_t0: bool = True,
        maxfev: int = 10000
    ) -> Dict:
        """
        Fit transit model to data.
        
        Parameters
        ----------
        times : np.ndarray
            Observation times
        fluxes : np.ndarray
            Normalized flux measurements
        errors : np.ndarray
            Flux uncertainties
        initial_params : dict, optional
            Initial guesses: {'rp': 0.1, 'a': 7.0, 'inc': 82, ...}
            Defaults provided for WASP-75b-like parameters
        bounds : dict, optional
            Parameter bounds: {'rp': (0.08, 0.12), ...}
        fix_t0 : bool, optional
            Fix t0 to initial guess (default: True)
            Recommended unless you have full transit coverage
        maxfev : int, optional
            Maximum function evaluations (default: 10000)
        
        Returns
        -------
        result : dict
            Fitted parameters and uncertainties:
            - fitted_params: {param: (value, uncertainty)}
            - t0: transit center (fixed or fitted)
            - residuals: fit residuals
            - chi_squared: χ²
            - reduced_chi_squared: χ²/(N - n_params)
            - model_flux: best-fit model
        
        Notes
        -----
        Fits 5 parameters (if fix_t0=True):
        1. rp: radius ratio (Rp/Rs)
        2. a: scaled semi-major axis (a/Rs)
        3. inc: inclination (degrees)
        4. baseline: flux normalization
        5. slope: linear trend (flux/time)
        
        Uses scipy.optimize.curve_fit with errors for proper weighting.
        """
        # Default initial parameters (WASP-75b-like)
        if initial_params is None:
            initial_params = {
                'rp': 0.103,
                'a': 7.17,
                'inc': 82.0,
                'baseline': np.median(fluxes),
                'slope': 0.0
            }
        
        # Default bounds (20% for a, 15% for rp, ±6° for inc)
        if bounds is None:
            bounds = {
                'rp': (0.085, 0.120),
                'a': (initial_params['a'] * 0.8, initial_params['a'] * 1.2),
                'inc': (initial_params['inc'] - 6, initial_params['inc'] + 6),
                'baseline': (fluxes.min() * 0.9, fluxes.max() * 1.1),
                'slope': (-0.1, 0.1)
            }
        
        # Prepare for curve_fit
        param_names = ['rp', 'a', 'inc', 'baseline', 'slope']
        p0 = [initial_params[p] for p in param_names]
        lower_bounds = [bounds[p][0] for p in param_names]
        upper_bounds = [bounds[p][1] for p in param_names]
        
        print(f"\n{'='*70}")
        print("TRANSIT MODEL FITTING")
        print(f"{'='*70}")
        print(f"Transit center (t0): {self.t0_guess:.6f} [{'FIXED' if fix_t0 else 'FREE'}]")
        print(f"Period: {self.period:.6f} days [FIXED]")
        print(f"Limb darkening: u1={self.limb_dark_u1:.2f}, u2={self.limb_dark_u2:.2f} [FIXED]")
        print(f"\nInitial parameters:")
        for name, val in zip(param_names, p0):
            print(f"  {name}: {val:.4f}")
        
        # Fit
        try:
            popt, pcov = curve_fit(
                self.model_with_detrending,
                times, fluxes,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                sigma=errors,
                absolute_sigma=True,
                maxfev=maxfev
            )
        except Exception as e:
            raise RuntimeError(f"Transit fit failed: {e}")
        
        # Extract results
        fitted_params = {}
        perr = np.sqrt(np.diag(pcov))
        
        for name, val, err in zip(param_names, popt, perr):
            fitted_params[name] = (float(val), float(err))
        
        # Generate best-fit model
        model_flux = self.model_with_detrending(times, *popt)
        residuals = fluxes - model_flux
        
        # Compute goodness of fit
        chi_squared = np.sum((residuals / errors)**2)
        n_params = len(param_names)
        reduced_chi_squared = chi_squared / (len(times) - n_params)
        
        # Check for boundary hits
        hitting_bounds = []
        for name, val, (low, high) in zip(param_names, popt, zip(lower_bounds, upper_bounds)):
            if abs(val - low) < 0.01 * abs(high - low):
                hitting_bounds.append(f"{name} at LOWER bound")
            elif abs(val - high) < 0.01 * abs(high - low):
                hitting_bounds.append(f"{name} at UPPER bound")
        
        if hitting_bounds:
            warnings.warn(
                f"Parameters hitting bounds: {', '.join(hitting_bounds)}. "
                "Consider adjusting bounds or initial guesses."
            )
        
        # Print results
        print(f"\n{'='*70}")
        print("FITTED PARAMETERS")
        print(f"{'='*70}")
        for name, (val, err) in fitted_params.items():
            print(f"{name:>12}: {val:.6f} ± {err:.6f}")
        
        print(f"\n{'='*70}")
        print("GOODNESS OF FIT")
        print(f"{'='*70}")
        print(f"χ² = {chi_squared:.2f}")
        print(f"Reduced χ² = {reduced_chi_squared:.2f}")
        print(f"RMS residuals: {np.std(residuals):.6f}")
        print(f"Mean error bar: {np.mean(errors):.6f}")
        print(f"RMS/error: {np.std(residuals)/np.mean(errors):.2f}")
        print(f"{'='*70}\n")
        
        return {
            'fitted_params': fitted_params,
            't0': self.t0_guess,
            'period': self.period,
            'residuals': residuals,
            'chi_squared': float(chi_squared),
            'reduced_chi_squared': float(reduced_chi_squared),
            'model_flux': model_flux,
            'covariance': pcov,
            'hitting_bounds': hitting_bounds
        }
    
    def derive_physical_params(
        self,
        fit_result: Dict,
        r_star_solar: Optional[float] = None,
        m_star_solar: Optional[float] = None
    ) -> Dict:
        """
        Derive physical parameters from fit results.
        
        Parameters
        ----------
        fit_result : dict
            Output from fit() method
        r_star_solar : float, optional
            Stellar radius in solar radii (for physical units)
        m_star_solar : float, optional
            Stellar mass in solar masses (for verification)
        
        Returns
        -------
        derived : dict
            Physical parameters:
            - transit_depth: (Rp/Rs)² in percent
            - impact_parameter: b = (a/Rs) × cos(i)
            - stellar_density: ρ★ from Kepler's 3rd law (g/cm³)
            - planet_radius: Rp in Jupiter radii (if r_star_solar given)
            - semi_major_axis: a in AU (if r_star_solar given)
            - orbital_velocity: v_orb in km/s (if r_star_solar given)
        
        Notes
        -----
        Stellar density can be measured from photometry alone:
            ρ★ = (3π)/(GP²) × (a/R★)³
        
        This provides a consistency check with spectroscopic estimates.
        """
        rp = fit_result['fitted_params']['rp'][0]
        rp_err = fit_result['fitted_params']['rp'][1]
        
        a = fit_result['fitted_params']['a'][0]
        a_err = fit_result['fitted_params']['a'][1]
        
        inc = fit_result['fitted_params']['inc'][0]
        inc_err = fit_result['fitted_params']['inc'][1]
        
        # Transit depth
        depth = rp**2
        depth_err = 2 * rp * rp_err
        
        # Impact parameter
        inc_rad = np.deg2rad(inc)
        b = a * np.cos(inc_rad)
        b_err = np.sqrt(
            (np.cos(inc_rad) * a_err)**2 +
            (a * np.sin(inc_rad) * np.deg2rad(inc_err))**2
        )
        
        # Stellar density from Kepler's 3rd law
        G = 6.67430e-11  # m³ kg⁻¹ s⁻²
        P_seconds = self.period * 24 * 3600
        rho_star_SI = (3 * np.pi) / (G * P_seconds**2) * a**3  # kg/m³
        rho_star_cgs = rho_star_SI * 0.001  # g/cm³
        rho_star_err = rho_star_cgs * 3 * (a_err / a)  # δρ/ρ = 3×δa/a
        
        derived = {
            'transit_depth_pct': (depth * 100, depth_err * 100),
            'impact_parameter': (b, b_err),
            'stellar_density_cgs': (rho_star_cgs, rho_star_err)
        }
        
        # Physical units (if stellar radius provided)
        if r_star_solar is not None:
            R_sun = 6.96e8  # m
            R_jupiter = 7.1492e7  # m
            AU = 1.496e11  # m
            R_star = r_star_solar * R_sun
            
            # Planet radius
            Rp_jupiter = rp * R_star / R_jupiter
            Rp_err_jupiter = rp_err * R_star / R_jupiter
            derived['planet_radius_jupiter'] = (Rp_jupiter, Rp_err_jupiter)
            
            # Semi-major axis
            a_physical = a * R_star
            a_AU = a_physical / AU
            a_AU_err = a_err * R_star / AU
            derived['semi_major_axis_AU'] = (a_AU, a_AU_err)
            
            # Orbital velocity
            v_orb = 2 * np.pi * a_physical / P_seconds / 1000  # km/s
            v_orb_err = 2 * np.pi * a_err * R_star / P_seconds / 1000
            derived['orbital_velocity_kms'] = (v_orb, v_orb_err)
        
        # Print summary
        print(f"\n{'='*70}")
        print("DERIVED PHYSICAL PARAMETERS")
        print(f"{'='*70}")
        for key, (val, err) in derived.items():
            print(f"{key:>30}: {val:.4f} ± {err:.4f}")
        print(f"{'='*70}\n")
        
        return derived
