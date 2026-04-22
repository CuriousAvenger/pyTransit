"""
Detrending and outlier removal for light curves.

Implements:
- Sigma clipping for outlier rejection
- Rolling-window MAD filter for tracking drift rejection
- Isolation Forest anomaly detection
- Airmass correlation analysis
- Robust Huber regression for airmass detrending
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
    method: str = "median",
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

    print(
        f"✓ Sigma clipping: removed {n_rejected}/{n_total} outliers "
        f"({n_rejected/n_total*100:.1f}%)"
    )

    return times[mask], fluxes[mask], errors[mask], mask


def test_airmass_correlation(airmass: np.ndarray, fluxes: np.ndarray) -> dict:
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
            "correlation": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "trend_percent": np.nan,
            "needs_correction": False,
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
        "correlation": float(correlation),
        "slope": float(slope),
        "intercept": float(intercept),
        "trend_percent": float(trend_percent),
        "needs_correction": needs_correction,
    }


def remove_linear_trend(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: Optional[np.ndarray] = None,
    return_model: bool = False,
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
    test_airmass: bool = True,
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

    result["mask"] = mask

    # Step 2: Airmass correlation test
    if test_airmass and airmass is not None:
        airmass_clean = airmass[mask]
        airmass_result = test_airmass_correlation(airmass_clean, fluxes_clean)
        result["airmass_test"] = airmass_result

    # Step 3: Linear detrending
    if remove_linear:
        fluxes_detrended, slope, intercept = remove_linear_trend(
            times_clean, fluxes_clean, errors_clean
        )
        result["linear_slope"] = slope
        result["linear_intercept"] = intercept
    else:
        fluxes_detrended = fluxes_clean
        result["linear_slope"] = 0.0
        result["linear_intercept"] = np.mean(fluxes_clean)

    result["times"] = times_clean
    result["fluxes"] = fluxes_detrended
    result["errors"] = errors_clean

    print("\n✓ Detrending complete")
    print(f"  Initial points: {len(times)}")
    print(f"  After sigma clip: {len(times_clean)}")
    print(f"  RMS before: {np.std(fluxes_clean):.6f}")
    print(f"  RMS after: {np.std(fluxes_detrended):.6f}")

    return result


# ============================================================================
# ROLLING-WINDOW MAD FILTER
# ============================================================================


def rolling_mad_filter(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    window_size: int = 20,
    sigma_mad: float = 3.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reject outliers using a rolling-window Median Absolute Deviation filter.

    Unlike global sigma clipping, each point is compared only to its local
    neighbourhood, preventing genuine transit ingress/egress from being
    misidentified as outliers.

    Parameters
    ----------
    times : np.ndarray
        Observation times.
    fluxes : np.ndarray
        Flux measurements.
    errors : np.ndarray
        Flux uncertainties.
    window_size : int, optional
        Number of consecutive points in each rolling window (default: 20).
        Should be small enough to track local scatter but large enough to
        average over noise (10–30 is typical).
    sigma_mad : float, optional
        Rejection threshold in units of MAD-derived σ (default: 3.5).
        3.5 is recommended: conservative enough to protect ingress/egress
        while still catching sharp tracking-drift spikes.

    Returns
    -------
    times_clean, fluxes_clean, errors_clean : np.ndarray
        Filtered arrays.
    mask : np.ndarray of bool
        True where the point was *kept*.

    Notes
    -----
    MAD-derived σ:  σ_MAD = 1.4826 × MAD (consistent with Gaussian σ).
    """
    n = len(fluxes)
    half = window_size // 2
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = fluxes[lo:hi]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        sigma_equiv = 1.4826 * mad
        if sigma_equiv > 0 and np.abs(fluxes[i] - med) > sigma_mad * sigma_equiv:
            mask[i] = False

    n_rejected = np.sum(~mask)
    print(
        f"✓ Rolling MAD filter (window={window_size}, {sigma_mad}σ): "
        f"removed {n_rejected}/{n} outliers ({n_rejected / n * 100:.1f}%)"
    )
    return times[mask], fluxes[mask], errors[mask], mask


# ============================================================================
# ISOLATION FOREST ANOMALY DETECTION
# ============================================================================


def isolation_forest_filter(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reject anomalies (tracking drifts, scintillation spikes) via Isolation Forest.

    The algorithm isolates anomalies by randomly partitioning the feature space;
    anomalous points require fewer splits than normal inliers.

    Parameters
    ----------
    times : np.ndarray
        Observation times.
    fluxes : np.ndarray
        Flux measurements.
    errors : np.ndarray
        Flux uncertainties.
    contamination : float, optional
        Expected fraction of outliers in [0, 0.5] (default: 0.05).
        Set higher (~0.10) for data with severe tracking problems.
    n_estimators : int, optional
        Number of isolation trees (default: 200).
    random_state : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    times_clean, fluxes_clean, errors_clean : np.ndarray
        Filtered arrays.
    mask : np.ndarray of bool
        True where the point was classified as an inlier.

    Notes
    -----
    Features used: normalised (time, flux, local flux gradient).
    The gradient distinguishes sharp tracking spikes (large gradient) from
    smooth transit ingress/egress, protecting genuine astrophysical signal.
    Requires ``scikit-learn``.
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        raise ImportError(
            "scikit-learn is required for Isolation Forest filtering. "
            "Install with: pip install scikit-learn"
        )

    eps = 1e-10
    t_norm = (times - times.mean()) / (times.std() + eps)
    f_norm = (fluxes - fluxes.mean()) / (fluxes.std() + eps)
    grad = np.gradient(fluxes, times)
    g_norm = (grad - grad.mean()) / (grad.std() + eps)

    features = np.column_stack([t_norm, f_norm, g_norm])

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    predictions = clf.fit_predict(features)
    mask = predictions == 1  # 1 = inlier, -1 = outlier

    n_rejected = np.sum(~mask)
    n_total = len(fluxes)
    print(
        f"✓ Isolation Forest: rejected {n_rejected}/{n_total} anomalies "
        f"({n_rejected / n_total * 100:.1f}%, contamination={contamination})"
    )
    return times[mask], fluxes[mask], errors[mask], mask


# ============================================================================
# HUBER REGRESSION AIRMASS DETRENDING
# ============================================================================


def huber_airmass_detrend(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    airmass: np.ndarray,
    epsilon: float = 1.35,
) -> Tuple[np.ndarray, float, float]:
    """
    Remove airmass-correlated atmospheric extinction via robust Huber regression.

    Ordinary least squares (OLS) is sensitive to outliers that inflate the
    fitted extinction slope, biasing the out-of-transit baseline and
    underestimating transit depth (producing high χ²_red).  Huber regression
    minimises a combined L1/L2 loss, making it robust to such outliers.

    Parameters
    ----------
    times : np.ndarray
        Observation times (unused in fit; retained for API consistency).
    fluxes : np.ndarray
        Flux measurements.
    errors : np.ndarray
        Flux uncertainties (used as inverse-variance weights).
    airmass : np.ndarray
        Airmass values at each observation epoch.
    epsilon : float, optional
        Huber transition parameter (default: 1.35 ≈ 95% Gaussian efficiency).
        Smaller values increase robustness at the cost of efficiency.

    Returns
    -------
    fluxes_detrended : np.ndarray
        Airmass-corrected fluxes, normalised so that the out-of-transit
        baseline is preserved.
    slope : float
        Fitted extinction slope (Δflux / Δairmass).
    intercept : float
        Fitted intercept.

    Notes
    -----
    Detrending: ``F_corr = F_obs / (trend / mean(trend))``
    so the mean flux level is preserved.  Requires ``scikit-learn``.
    """
    try:
        from sklearn.linear_model import HuberRegressor
    except ImportError:
        raise ImportError(
            "scikit-learn is required for Huber regression. "
            "Install with: pip install scikit-learn"
        )

    airmass_c = airmass - airmass.mean()
    X = airmass_c.reshape(-1, 1)
    weights = 1.0 / (errors**2 + 1e-20)

    try:
        huber = HuberRegressor(epsilon=epsilon, max_iter=300)
        huber.fit(X, fluxes, sample_weight=weights)
        slope = float(huber.coef_[0])
        intercept = float(huber.intercept_)
    except Exception as exc:
        warnings.warn(f"Huber fit failed ({exc}); falling back to weighted OLS.")
        w_norm = weights / weights.sum()
        coeffs = np.polyfit(airmass_c, fluxes, 1, w=np.sqrt(w_norm))
        slope, intercept = float(coeffs[0]), float(coeffs[1])

    trend = slope * airmass_c + intercept
    trend_norm = trend / np.mean(trend)
    fluxes_detrended = fluxes / trend_norm

    print(
        f"✓ Huber airmass detrending: slope={slope:.6f}, "
        f"intercept={intercept:.6f} (ε={epsilon})"
    )
    return fluxes_detrended, slope, intercept


# ============================================================================
# ENHANCED FULL DETRENDING PIPELINE
# ============================================================================


def detrend_lightcurve_advanced(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    airmass: Optional[np.ndarray] = None,
    outlier_method: str = "rolling_mad",
    sigma_threshold: float = 3.0,
    window_size: int = 20,
    mad_sigma: float = 3.5,
    contamination: float = 0.05,
    remove_linear: bool = True,
    test_airmass: bool = True,
    airmass_regression: str = "huber",
    huber_epsilon: float = 1.35,
) -> dict:
    """
    Full detrending pipeline with advanced anomaly rejection and robust regression.

    Parameters
    ----------
    times, fluxes, errors : np.ndarray
        Light curve arrays.
    airmass : np.ndarray, optional
        Airmass values for extinction detrending.
    outlier_method : str, optional
        ``'sigma_clip'``, ``'rolling_mad'`` (default), or ``'isolation_forest'``.
    sigma_threshold : float, optional
        Sigma threshold for ``'sigma_clip'`` (default: 3.0).
    window_size : int, optional
        Rolling window size for ``'rolling_mad'`` (default: 20).
    mad_sigma : float, optional
        MAD rejection threshold (default: 3.5).
    contamination : float, optional
        Expected outlier fraction for ``'isolation_forest'`` (default: 0.05).
    remove_linear : bool, optional
        Remove linear time trend (default: True).
    test_airmass : bool, optional
        Test for airmass–flux correlation (default: True).
    airmass_regression : str, optional
        ``'ols'`` or ``'huber'`` (default) for airmass detrending.
    huber_epsilon : float, optional
        Huber ε parameter (default: 1.35).

    Returns
    -------
    result : dict
        Keys: ``'times'``, ``'fluxes'``, ``'errors'``, ``'mask'``,
        ``'linear_slope'``, ``'airmass_test'``, ``'huber_slope'``,
        ``'huber_intercept'``.
    """
    result = {}

    # ── Step 1: Anomaly / outlier rejection ──────────────────────────────────
    if outlier_method == "sigma_clip":
        times_clean, fluxes_clean, errors_clean, mask = sigma_clip(
            times, fluxes, errors, sigma_threshold=sigma_threshold
        )
    elif outlier_method == "rolling_mad":
        times_clean, fluxes_clean, errors_clean, mask = rolling_mad_filter(
            times,
            fluxes,
            errors,
            window_size=window_size,
            sigma_mad=mad_sigma,
        )
    elif outlier_method == "isolation_forest":
        times_clean, fluxes_clean, errors_clean, mask = isolation_forest_filter(
            times,
            fluxes,
            errors,
            contamination=contamination,
        )
    else:
        raise ValueError(
            f"Unknown outlier_method '{outlier_method}'. "
            "Choose 'sigma_clip', 'rolling_mad', or 'isolation_forest'."
        )

    result["mask"] = mask

    # ── Step 2: Airmass correlation test and robust detrending ───────────────
    if test_airmass and airmass is not None:
        airmass_clean = airmass[mask]
        airmass_result = test_airmass_correlation(airmass_clean, fluxes_clean)
        result["airmass_test"] = airmass_result

        if airmass_result["needs_correction"]:
            if airmass_regression == "huber":
                fluxes_clean, huber_slope, huber_intercept = huber_airmass_detrend(
                    times_clean,
                    fluxes_clean,
                    errors_clean,
                    airmass_clean,
                    epsilon=huber_epsilon,
                )
                result["huber_slope"] = huber_slope
                result["huber_intercept"] = huber_intercept
            else:
                # OLS via existing remove_linear_trend on airmass
                coeffs = np.polyfit(airmass_clean - airmass_clean.mean(), fluxes_clean, 1)
                trend = np.polyval(coeffs, airmass_clean - airmass_clean.mean())
                fluxes_clean = fluxes_clean / (trend / np.mean(trend))
                result["huber_slope"] = float(coeffs[0])
                result["huber_intercept"] = float(coeffs[1])

    # ── Step 3: Linear time detrending ───────────────────────────────────────
    if remove_linear:
        fluxes_detrended, slope, intercept = remove_linear_trend(
            times_clean, fluxes_clean, errors_clean
        )
        result["linear_slope"] = slope
        result["linear_intercept"] = intercept
    else:
        fluxes_detrended = fluxes_clean
        result["linear_slope"] = 0.0
        result["linear_intercept"] = float(np.mean(fluxes_clean))

    result["times"] = times_clean
    result["fluxes"] = fluxes_detrended
    result["errors"] = errors_clean

    print("\n✓ Advanced detrending complete")
    print(f"  Initial: {len(times)} pts  →  After rejection: {len(times_clean)} pts")
    print(f"  RMS before: {np.std(fluxes_clean):.6f}  →  " f"after: {np.std(fluxes_detrended):.6f}")

    return result
