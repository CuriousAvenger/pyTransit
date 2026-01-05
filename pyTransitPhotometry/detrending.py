"""
Detrending and outlier removal for light curves.

Implements:
- Sigma clipping for outlier rejection
- Airmass correlation analysis
- Linear trend removal
- Systematic effects diagnostics
"""

import numpy as np
from typing import Tuple, Optional
import warnings


def sigma_clip(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    sigma_threshold: float = 3.0,
    max_iterations: int = 5,
    method: str = "median"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove outliers using iterative sigma clipping.
    
    Parameters
    ----------
    times : np.ndarray
        Observation times
    fluxes : np.ndarray
        Flux measurements
    errors : np.ndarray
        Flux uncertainties
    sigma_threshold : float, optional
        Sigma threshold for clipping (default: 3.0)
    max_iterations : int, optional
        Maximum number of clipping iterations (default: 5)
    method : str, optional
        'median' (default, robust) or 'mean'
    
    Returns
    -------
    times_clipped : np.ndarray
        Clipped times
    fluxes_clipped : np.ndarray
        Clipped fluxes
    errors_clipped : np.ndarray
        Clipped errors
    mask : np.ndarray
        Boolean mask (True = kept, False = rejected)
    
    Notes
    -----
    Iterative sigma clipping:
    1. Compute central value (median/mean) and scatter (std)
    2. Reject points > sigma_threshold × scatter from center
    3. Repeat until no new outliers or max_iterations reached
    
    Also rejects points with errors > 3× median error (bad photometry).
    
    Examples
    --------
    >>> times_clean, fluxes_clean, errs_clean, mask = sigma_clip(
    ...     times, fluxes, errors, sigma_threshold=3.0
    ... )
    >>> print(f"Rejected {np.sum(~mask)} outliers")
    """
    mask = np.ones(len(fluxes), dtype=bool)
    
    for iteration in range(max_iterations):
        current_fluxes = fluxes[mask]
        current_errors = errors[mask]
        
        if len(current_fluxes) < 3:
            warnings.warn("Too few points remaining after sigma clipping")
            break
        
        # Compute central value and scatter
        if method == "median":
            center = np.median(current_fluxes)
            # Use MAD for robust scatter estimate
            mad = np.median(np.abs(current_fluxes - center))
            scatter = 1.4826 * mad  # Convert MAD to std for normal distribution
        elif method == "mean":
            center = np.mean(current_fluxes)
            scatter = np.std(current_fluxes)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Identify outliers
        deviation = np.abs(fluxes - center)
        flux_outliers = deviation > sigma_threshold * scatter
        
        # Also clip points with unusually large errors
        median_error = np.median(errors[mask])
        error_outliers = errors > 3 * median_error
        
        # Update mask
        new_mask = mask & ~flux_outliers & ~error_outliers
        
        # Check for convergence
        if np.array_equal(new_mask, mask):
            break
        
        mask = new_mask
    
    n_rejected = np.sum(~mask)
    n_total = len(fluxes)
    
    print(f"✓ Sigma clipping: removed {n_rejected}/{n_total} outliers "
          f"({n_rejected/n_total*100:.1f}%)")
    
    return times[mask], fluxes[mask], errors[mask], mask


def test_airmass_correlation(
    airmass: np.ndarray,
    fluxes: np.ndarray
) -> dict:
    """
    Test for correlation between airmass and flux.
    
    Parameters
    ----------
    airmass : np.ndarray
        Airmass values
    fluxes : np.ndarray
        Flux measurements
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - correlation: Pearson correlation coefficient
        - slope: linear fit slope (dF/dX)
        - intercept: linear fit intercept
        - trend_percent: flux change over airmass range (%)
        - needs_correction: bool, True if |r| > 0.3
    
    Notes
    -----
    Airmass affects flux via atmospheric extinction:
        F_obs = F_0 × 10^(-k×X/2.5)
    
    where k is extinction coefficient and X is airmass.
    
    Differential photometry should cancel most airmass effects if
    target and references have similar colors. Residual correlation
    suggests:
    - Color mismatch between target and references
    - Non-photometric conditions
    - Poor reference star selection
    
    Examples
    --------
    >>> result = test_airmass_correlation(airmass, fluxes)
    >>> if result['needs_correction']:
    ...     print("Airmass correction recommended")
    """
    # Remove NaN values
    valid = np.isfinite(airmass) & np.isfinite(fluxes)
    
    if np.sum(valid) < 3:
        warnings.warn("Too few valid points for airmass correlation test")
        return {
            'correlation': np.nan,
            'slope': np.nan,
            'intercept': np.nan,
            'trend_percent': np.nan,
            'needs_correction': False
        }
    
    airmass_valid = airmass[valid]
    fluxes_valid = fluxes[valid]
    
    # Compute correlation
    correlation = np.corrcoef(airmass_valid, fluxes_valid)[0, 1]
    
    # Linear fit
    coeffs = np.polyfit(airmass_valid, fluxes_valid, deg=1)
    slope, intercept = coeffs[0], coeffs[1]
    
    # Compute trend magnitude
    airmass_range = airmass_valid.max() - airmass_valid.min()
    flux_change = slope * airmass_range
    trend_percent = abs(flux_change) / np.mean(fluxes_valid) * 100
    
    # Decision threshold
    needs_correction = abs(correlation) > 0.3
    
    print(f"\n{'='*60}")
    print("AIRMASS CORRELATION TEST")
    print(f"{'='*60}")
    print(f"Airmass range: {airmass_valid.min():.3f} - {airmass_valid.max():.3f}")
    print(f"Correlation (r): {correlation:+.4f}")
    print(f"Trend: {trend_percent:.2f}% of mean flux")
    
    if abs(correlation) < 0.3:
        verdict = "✅ WEAK - No correction needed"
    elif abs(correlation) < 0.5:
        verdict = "⚠️  MODERATE - Correction optional"
    else:
        verdict = "❌ STRONG - Correction recommended"
    
    print(f"Result: {verdict} (|r| = {abs(correlation):.3f})")
    print(f"{'='*60}\n")
    
    return {
        'correlation': float(correlation),
        'slope': float(slope),
        'intercept': float(intercept),
        'trend_percent': float(trend_percent),
        'needs_correction': needs_correction
    }


def remove_linear_trend(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: Optional[np.ndarray] = None,
    return_model: bool = False
) -> Tuple[np.ndarray, float, float]:
    """
    Remove linear trend from light curve.
    
    Parameters
    ----------
    times : np.ndarray
        Observation times
    fluxes : np.ndarray
        Flux measurements
    errors : np.ndarray, optional
        Flux uncertainties (for weighted fit)
    return_model : bool, optional
        If True, return trend model instead of detrended fluxes
    
    Returns
    -------
    detrended_fluxes : np.ndarray
        Fluxes with linear trend removed (or model if return_model=True)
    slope : float
        Linear trend slope
    intercept : float
        Linear trend intercept
    
    Notes
    -----
    Fits F = slope × (t - t_mean) + intercept
    Then returns F_detrended = F - (slope × (t - t_mean))
    
    This preserves the mean flux level while removing the trend.
    
    Examples
    --------
    >>> detrended, slope, intercept = remove_linear_trend(
    ...     times, fluxes, errors=errors
    ... )
    >>> print(f"Removed trend: {slope:.6f} flux/day")
    """
    # Center times for numerical stability
    t_mean = np.mean(times)
    t_centered = times - t_mean
    
    # Fit linear trend
    if errors is not None and np.all(errors > 0):
        # Weighted least squares
        weights = 1.0 / errors**2
        coeffs = np.polyfit(t_centered, fluxes, deg=1, w=weights)
    else:
        # Ordinary least squares
        coeffs = np.polyfit(t_centered, fluxes, deg=1)
    
    slope, intercept = coeffs[0], coeffs[1]
    
    # Compute trend model
    trend_model = slope * t_centered + intercept
    
    if return_model:
        return trend_model, slope, intercept
    else:
        # Remove only the slope component, preserving mean level
        detrended = fluxes - (slope * t_centered)
        
        print(f"✓ Linear trend removed: slope = {slope:.6f} flux/day")
        print(f"  Trend over observation: {slope*(times.max()-times.min())*100:.2f}% of mean")
        
        return detrended, slope, intercept


def detrend_lightcurve(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    airmass: Optional[np.ndarray] = None,
    sigma_threshold: float = 3.0,
    remove_linear: bool = True,
    test_airmass: bool = True
) -> dict:
    """
    Full detrending pipeline: outlier removal + trend removal.
    
    Parameters
    ----------
    times : np.ndarray
        Observation times
    fluxes : np.ndarray
        Flux measurements
    errors : np.ndarray
        Flux uncertainties
    airmass : np.ndarray, optional
        Airmass values for correlation test
    sigma_threshold : float, optional
        Sigma clipping threshold (default: 3.0)
    remove_linear : bool, optional
        Remove linear trend (default: True)
    test_airmass : bool, optional
        Test for airmass correlation (default: True)
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - times: detrended times
        - fluxes: detrended fluxes
        - errors: corresponding errors
        - mask: boolean mask of kept points
        - linear_slope: fitted linear slope (if remove_linear=True)
        - airmass_test: airmass correlation results (if test_airmass=True)
    
    Examples
    --------
    >>> result = detrend_lightcurve(
    ...     times, fluxes, errors, airmass=airmass_values,
    ...     sigma_threshold=3.0, remove_linear=True
    ... )
    >>> lc = result['fluxes']
    """
    result = {}
    
    # Step 1: Sigma clipping
    times_clean, fluxes_clean, errors_clean, mask = sigma_clip(
        times, fluxes, errors, sigma_threshold=sigma_threshold
    )
    
    result['mask'] = mask
    
    # Step 2: Airmass correlation test
    if test_airmass and airmass is not None:
        airmass_clean = airmass[mask]
        airmass_result = test_airmass_correlation(airmass_clean, fluxes_clean)
        result['airmass_test'] = airmass_result
    
    # Step 3: Linear detrending
    if remove_linear:
        fluxes_detrended, slope, intercept = remove_linear_trend(
            times_clean, fluxes_clean, errors_clean
        )
        result['linear_slope'] = slope
        result['linear_intercept'] = intercept
    else:
        fluxes_detrended = fluxes_clean
        result['linear_slope'] = 0.0
        result['linear_intercept'] = np.mean(fluxes_clean)
    
    result['times'] = times_clean
    result['fluxes'] = fluxes_detrended
    result['errors'] = errors_clean
    
    print(f"\n✓ Detrending complete")
    print(f"  Initial points: {len(times)}")
    print(f"  After sigma clip: {len(times_clean)}")
    print(f"  RMS before: {np.std(fluxes_clean):.6f}")
    print(f"  RMS after: {np.std(fluxes_detrended):.6f}")
    
    return result
