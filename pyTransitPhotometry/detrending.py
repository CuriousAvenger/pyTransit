import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
import warnings

def sigma_clip(times: np.ndarray, fluxes: np.ndarray, errors: np.ndarray, sigma_threshold: float=3.0, max_iterations: int=5, method: str='median') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones(len(fluxes), dtype=bool)
    for iteration in range(max_iterations):
        current_fluxes = fluxes[mask]
        if len(current_fluxes) < 3:
            warnings.warn('Too few points remaining after sigma clipping')
            break
        if method == 'median':
            center = np.median(current_fluxes)
            mad = np.median(np.abs(current_fluxes - center))
            scatter = 1.4826 * mad
        elif method == 'mean':
            center = np.mean(current_fluxes)
            scatter = np.std(current_fluxes)
        else:
            raise ValueError(f'Unknown method: {method}')
        deviation = np.abs(fluxes - center)
        flux_outliers = deviation > sigma_threshold * scatter
        median_error = np.median(errors[mask])
        error_outliers = errors > 3 * median_error
        new_mask = mask & ~flux_outliers & ~error_outliers
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    n_rejected = np.sum(~mask)
    n_total = len(fluxes)
    print(f'✓ Sigma clipping: removed {n_rejected}/{n_total} outliers ({n_rejected / n_total * 100:.1f}%)')
    return (times[mask], fluxes[mask], errors[mask], mask)

def test_airmass_correlation(airmass: NDArray[np.float64], fluxes: NDArray[np.float64]) -> dict:
    valid = np.isfinite(airmass) & np.isfinite(fluxes)
    if np.sum(valid) < 3:
        warnings.warn('Too few valid points for airmass correlation test')
        return {'correlation': np.nan, 'slope': np.nan, 'intercept': np.nan, 'trend_percent': np.nan, 'needs_correction': False}
    airmass_valid = airmass[valid]
    fluxes_valid = fluxes[valid]
    correlation = np.corrcoef(airmass_valid, fluxes_valid)[0, 1]
    coeffs = np.polyfit(airmass_valid, fluxes_valid, deg=1)
    slope, intercept = (coeffs[0], coeffs[1])
    airmass_range = airmass_valid.max() - airmass_valid.min()
    flux_change = slope * airmass_range
    trend_percent = abs(flux_change) / np.mean(fluxes_valid) * 100
    needs_correction = abs(correlation) > 0.3
    print(f"\n{'=' * 60}")
    print('AIRMASS CORRELATION TEST')
    print(f"{'=' * 60}")
    print(f'Airmass range: {airmass_valid.min():.3f} - {airmass_valid.max():.3f}')
    print(f'Correlation (r): {correlation:+.4f}')
    print(f'Trend: {trend_percent:.2f}% of mean flux')
    if abs(correlation) < 0.3:
        verdict = '✅ WEAK - No correction needed'
    elif abs(correlation) < 0.5:
        verdict = '⚠️  MODERATE - Correction optional'
    else:
        verdict = '❌ STRONG - Correction recommended'
    print(f'Result: {verdict} (|r| = {abs(correlation):.3f})')
    print(f"{'=' * 60}\n")
    return {'correlation': float(correlation), 'slope': float(slope), 'intercept': float(intercept), 'trend_percent': float(trend_percent), 'needs_correction': needs_correction}

def remove_linear_trend(times: np.ndarray, fluxes: np.ndarray, errors: Optional[np.ndarray]=None, return_model: bool=False) -> Tuple[np.ndarray, float, float]:
    t_mean = np.mean(times)
    t_centered = times - t_mean
    if errors is not None and np.all(errors > 0):
        weights = 1.0 / errors ** 2
        coeffs = np.polyfit(t_centered, fluxes, deg=1, w=weights)
    else:
        coeffs = np.polyfit(t_centered, fluxes, deg=1)
    slope, intercept = (coeffs[0], coeffs[1])
    trend_model = slope * t_centered + intercept
    if return_model:
        return (trend_model, slope, intercept)
    else:
        detrended = fluxes - slope * t_centered
        print(f'✓ Linear trend removed: slope = {slope:.6f} flux/day')
        print(f'  Trend over observation: {slope * (times.max() - times.min()) * 100:.2f}% of mean')
        return (detrended, slope, intercept)

def detrend_lightcurve(times: np.ndarray, fluxes: np.ndarray, errors: np.ndarray, airmass: Optional[np.ndarray]=None, sigma_threshold: float=3.0, remove_linear: bool=True, test_airmass: bool=True) -> dict:
    result = {}
    times_clean, fluxes_clean, errors_clean, mask = sigma_clip(times, fluxes, errors, sigma_threshold=sigma_threshold)
    result['mask'] = mask
    if test_airmass and airmass is not None:
        airmass_clean = airmass[mask]
        airmass_result = test_airmass_correlation(airmass_clean, fluxes_clean)
        result['airmass_test'] = airmass_result
    if remove_linear:
        fluxes_detrended, slope, intercept = remove_linear_trend(times_clean, fluxes_clean, errors_clean)
        result['linear_slope'] = slope
        result['linear_intercept'] = intercept
    else:
        fluxes_detrended = fluxes_clean
        result['linear_slope'] = 0.0
        result['linear_intercept'] = np.mean(fluxes_clean)
    result['times'] = times_clean
    result['fluxes'] = fluxes_detrended
    result['errors'] = errors_clean
    print('\n✓ Detrending complete')
    print(f'  Initial points: {len(times)}')
    print(f'  After sigma clip: {len(times_clean)}')
    print(f'  RMS before: {np.std(fluxes_clean):.6f}')
    print(f'  RMS after: {np.std(fluxes_detrended):.6f}')
    return result

def rolling_mad_filter(times: np.ndarray, fluxes: np.ndarray, errors: np.ndarray, window_size: int=20, sigma_mad: float=3.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(times)
    mask = np.ones(n, dtype=bool)
    half = window_size // 2
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
    print(f'✓ Rolling MAD filter (window={window_size}, {sigma_mad}σ): removed {n_rejected}/{n} outliers ({n_rejected / n * 100:.1f}%)')
    return (times[mask], fluxes[mask], errors[mask], mask)

def isolation_forest_filter(times: np.ndarray, fluxes: np.ndarray, errors: np.ndarray, contamination: float=0.05, n_estimators: int=200, random_state: int=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        raise ImportError('scikit-learn is required for Isolation Forest filtering. Install with: pip install scikit-learn')
    eps = 1e-10
    t_norm = (times - times.mean()) / (times.std() + eps)
    f_norm = (fluxes - fluxes.mean()) / (fluxes.std() + eps)
    grad = np.gradient(fluxes, times)
    g_norm = (grad - grad.mean()) / (grad.std() + eps)
    features = np.column_stack([t_norm, f_norm, g_norm])
    clf = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state, n_jobs=-1)
    predictions = clf.fit_predict(features)
    mask = predictions == 1
    n_rejected = np.sum(~mask)
    n_total = len(fluxes)
    print(f'✓ Isolation Forest: rejected {n_rejected}/{n_total} anomalies ({n_rejected / n_total * 100:.1f}%, contamination={contamination})')
    return (times[mask], fluxes[mask], errors[mask], mask)

def huber_airmass_detrend(times: np.ndarray, fluxes: np.ndarray, errors: np.ndarray, airmass: np.ndarray, epsilon: float=1.35) -> Tuple[np.ndarray, float, float]:
    try:
        from sklearn.linear_model import HuberRegressor
    except ImportError:
        raise ImportError('scikit-learn is required for Huber regression. Install with: pip install scikit-learn')
    airmass_c = airmass - airmass.mean()
    X = airmass_c.reshape(-1, 1)
    weights = 1.0 / (errors ** 2 + 1e-20)
    try:
        huber = HuberRegressor(epsilon=epsilon, max_iter=300)
        huber.fit(X, fluxes, sample_weight=weights)
        slope = float(huber.coef_[0])
        intercept = float(huber.intercept_)
    except Exception as exc:
        warnings.warn(f'Huber fit failed ({exc}); falling back to weighted OLS.')
        w_norm = weights / weights.sum()
        coeffs = np.polyfit(airmass_c, fluxes, 1, w=np.sqrt(w_norm))
        slope, intercept = (float(coeffs[0]), float(coeffs[1]))
    trend = slope * airmass_c + intercept
    trend_norm = trend / np.mean(trend)
    fluxes_detrended = fluxes / trend_norm
    print(f'✓ Huber airmass detrending: slope={slope:.6f}, intercept={intercept:.6f} (ε={epsilon})')
    return (fluxes_detrended, slope, intercept)

def detrend_lightcurve_advanced(times: np.ndarray, fluxes: np.ndarray, errors: np.ndarray, airmass: Optional[np.ndarray]=None, outlier_method: str='rolling_mad', sigma_threshold: float=3.0, window_size: int=20, mad_sigma: float=3.5, contamination: float=0.05, remove_linear: bool=True, test_airmass: bool=True, airmass_regression: str='huber', huber_epsilon: float=1.35) -> dict:
    result = {}
    if outlier_method == 'sigma_clip':
        times_clean, fluxes_clean, errors_clean, mask = sigma_clip(times, fluxes, errors, sigma_threshold=sigma_threshold)
    elif outlier_method == 'rolling_mad':
        times_clean, fluxes_clean, errors_clean, mask = rolling_mad_filter(times, fluxes, errors, window_size=window_size, sigma_mad=mad_sigma)
    elif outlier_method == 'isolation_forest':
        times_clean, fluxes_clean, errors_clean, mask = isolation_forest_filter(times, fluxes, errors, contamination=contamination)
    else:
        raise ValueError(f"Unknown outlier_method '{outlier_method}'. Choose 'sigma_clip', 'rolling_mad', or 'isolation_forest'.")
    result['mask'] = mask
    if test_airmass and airmass is not None:
        airmass_clean = airmass[mask]
        airmass_result = test_airmass_correlation(airmass_clean, fluxes_clean)
        result['airmass_test'] = airmass_result
        if airmass_result['needs_correction']:
            if airmass_regression == 'huber':
                fluxes_clean, huber_slope, huber_intercept = huber_airmass_detrend(times_clean, fluxes_clean, errors_clean, airmass_clean, epsilon=huber_epsilon)
                result['huber_slope'] = huber_slope
                result['huber_intercept'] = huber_intercept
            else:
                coeffs = np.polyfit(airmass_clean - airmass_clean.mean(), fluxes_clean, 1)
                trend = np.polyval(coeffs, airmass_clean - airmass_clean.mean())
                fluxes_clean = fluxes_clean / (trend / np.mean(trend))
                result['huber_slope'] = float(coeffs[0])
                result['huber_intercept'] = float(coeffs[1])
    if remove_linear:
        fluxes_detrended, slope, intercept = remove_linear_trend(times_clean, fluxes_clean, errors_clean)
        result['linear_slope'] = slope
        result['linear_intercept'] = intercept
    else:
        fluxes_detrended = fluxes_clean
        result['linear_slope'] = 0.0
        result['linear_intercept'] = float(np.mean(fluxes_clean))
    result['times'] = times_clean
    result['fluxes'] = fluxes_detrended
    result['errors'] = errors_clean
    print('\n✓ Advanced detrending complete')
    print(f'  Initial: {len(times)} pts  →  After rejection: {len(times_clean)} pts')
    print(f'  RMS before: {np.std(fluxes_clean):.6f}  →  after: {np.std(fluxes_detrended):.6f}')
    return result

def detrend_oot(times: NDArray[np.float64], fluxes: NDArray[np.float64], errors: NDArray[np.float64], oot_percentile: float=25.0, sigma_threshold: float=3.0) -> 'LightCurve':
    t_low = np.percentile(times, oot_percentile)
    t_high = np.percentile(times, 100.0 - oot_percentile)
    oot_mask_full = (times <= t_low) | (times >= t_high)
    if np.sum(oot_mask_full) < 4:
        warnings.warn('Too few OOT points for trend fitting; using all points instead.')
        oot_mask_full = np.ones(len(times), dtype=bool)
    t_mean = np.mean(times)
    t_centered = times - t_mean
    t_oot = t_centered[oot_mask_full]
    f_oot = fluxes[oot_mask_full]
    coeffs = np.polyfit(t_oot, f_oot, deg=1)
    trend = np.polyval(coeffs, t_centered)
    detrended = fluxes / trend
    errors_detrended = errors / trend
    oot_median = np.median(detrended[oot_mask_full])
    detrended = detrended / oot_median
    errors_detrended = errors_detrended / oot_median
    times_out, fluxes_out, errors_out, clip_mask = sigma_clip(times, detrended, errors_detrended, sigma_threshold=sigma_threshold, max_iterations=5)
    print(f'✓ OOT detrending: {np.sum(oot_mask_full)} OOT points used for trend fit')
    print(f'  Trend slope : {coeffs[0]:.6f} flux/time-unit')
    print(f'  Normalisation factor : {oot_median:.6f}')
    print(f'  Clipped outliers : {np.sum(~clip_mask)}/{len(times)}')
    from .lightcurve import LightCurve
    return LightCurve(times=times_out, fluxes=fluxes_out, errors=errors_out, mask=clip_mask, linear_slope=float(coeffs[0]), oot_mask=oot_mask_full)
