import warnings
from dataclasses import dataclass, field, fields
from typing import Callable, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

@dataclass
class LightCurve:
    times: NDArray[np.float64]
    fluxes: NDArray[np.float64]
    errors: NDArray[np.float64]
    target_fluxes: Optional[NDArray[np.float64]] = field(default=None)
    reference_fluxes: Optional[NDArray[np.float64]] = field(default=None)
    centroids: Optional[NDArray[np.float64]] = field(default=None)
    valid_frames: Optional[NDArray[np.bool_]] = field(default=None)
    mask: Optional[NDArray[np.bool_]] = field(default=None)
    linear_slope: float = field(default=0.0)
    oot_mask: Optional[NDArray[np.bool_]] = field(default=None)

    def __getitem__(self, key: str) -> object:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def __len__(self) -> int:
        return len(self.times)

    def keys(self) -> List[str]:
        return [f.name for f in fields(self)]

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

def differential_photometry(target_flux: float, target_err: float, reference_fluxes: NDArray[np.float64], reference_errs: NDArray[np.float64], weighting: str='inverse_variance') -> Tuple[float, float]:
    if len(reference_fluxes) == 0:
        raise ValueError('No reference stars provided')
    if len(reference_fluxes) != len(reference_errs):
        raise ValueError('reference_fluxes and reference_errs must have equal length')
    if target_flux <= 0:
        warnings.warn(f'Target flux non-positive: {target_flux:.1f}')
        return (np.nan, np.nan)
    valid = (reference_fluxes > 0) & (reference_errs > 0)
    if not np.any(valid):
        warnings.warn('No valid reference stars (all fluxes ≤ 0)')
        return (np.nan, np.nan)
    ref_f = reference_fluxes[valid]
    ref_e = reference_errs[valid]
    if weighting == 'inverse_variance':
        weights = 1.0 / ref_e ** 2
        weighted_ref = np.sum(ref_f * weights) / np.sum(weights)
        weighted_ref_err = np.sqrt(1.0 / np.sum(weights))
    elif weighting == 'equal':
        weighted_ref = np.mean(ref_f)
        weighted_ref_err = np.sqrt(np.sum(ref_e ** 2)) / len(ref_e)
    else:
        raise ValueError(f"Unknown weighting: '{weighting}'. Choose 'inverse_variance' or 'equal'.")
    ratio = target_flux / weighted_ref
    ratio_err = ratio * np.sqrt((target_err / target_flux) ** 2 + (weighted_ref_err / weighted_ref) ** 2)
    return (float(ratio), float(ratio_err))

class LightCurveBuilder:

    def __init__(self, target_index: int, reference_indices: List[int], weighting: str='inverse_variance') -> None:
        if target_index in reference_indices:
            raise ValueError('Target cannot be in reference list')
        if len(reference_indices) == 0:
            raise ValueError('At least one reference star required')
        self.target_index = target_index
        self.reference_indices = reference_indices
        self.weighting = weighting
        print('✓ Light curve builder initialized')
        print(f'  Target: star #{target_index}')
        print(f'  References: stars {reference_indices}')
        print(f'  Weighting: {weighting}')

    def build(self, images: NDArray[np.float32], sources_per_frame: List, photometry_func: Callable, time_extractor: Callable, verbose: bool=True) -> LightCurve:
        n_frames = len(images)
        times_list: List[float] = []
        ratios_list: List[float] = []
        ratio_errs_list: List[float] = []
        target_fluxes_list: List[float] = []
        reference_fluxes_list: List[float] = []
        centroids_list: List[Tuple[float, float]] = []
        valid_frames_list: List[bool] = []
        for i in range(n_frames):
            try:
                image = images[i]
                sources = sources_per_frame[i]
                time = time_extractor(i)
                required_indices = [self.target_index] + self.reference_indices
                if max(required_indices) >= len(sources):
                    if verbose:
                        print(f'⚠  Frame {i + 1}/{n_frames}: only {len(sources)} sources detected (need index {max(required_indices)}), skipping')
                    valid_frames_list.append(False)
                    continue
                photometry_func._frame_idx = i
                target_result = photometry_func(image, self.target_index)
                target_flux = target_result['flux']
                target_err = target_result['flux_err']
                target_centroid = target_result['centroid']
                ref_fluxes: List[float] = []
                ref_errs: List[float] = []
                for ref_idx in self.reference_indices:
                    try:
                        r = photometry_func(image, ref_idx)
                        ref_fluxes.append(r['flux'])
                        ref_errs.append(r['flux_err'])
                    except Exception as exc:
                        if verbose:
                            print(f'⚠  Frame {i + 1}/{n_frames}: reference star {ref_idx} failed: {exc}')
                if not ref_fluxes:
                    if verbose:
                        print(f'⚠  Frame {i + 1}/{n_frames}: no valid references, skipping')
                    valid_frames_list.append(False)
                    continue
                ratio, ratio_err = differential_photometry(target_flux, target_err, np.array(ref_fluxes), np.array(ref_errs), weighting=self.weighting)
                if not (np.isfinite(ratio) and np.isfinite(ratio_err)):
                    if verbose:
                        print(f'⚠  Frame {i + 1}/{n_frames}: non-finite ratio, skipping')
                    valid_frames_list.append(False)
                    continue
                times_list.append(time)
                ratios_list.append(ratio)
                ratio_errs_list.append(ratio_err)
                target_fluxes_list.append(target_flux)
                reference_fluxes_list.append(float(np.mean(ref_fluxes)))
                centroids_list.append(target_centroid)
                valid_frames_list.append(True)
                if verbose and (i + 1) % 10 == 0:
                    print(f'  Processed {i + 1}/{n_frames} frames...')
            except Exception as exc:
                if verbose:
                    print(f'⚠  Frame {i + 1}/{n_frames}: {exc}, skipping')
                valid_frames_list.append(False)
        if not times_list:
            raise RuntimeError('No valid frames processed. Check data integrity and pipeline parameters.')
        ratios_arr = np.array(ratios_list)
        print(f'\n✓ Light curve built: {len(times_list)}/{n_frames} frames valid')
        print(f'  Mean flux ratio : {np.mean(ratios_arr):.6f}')
        print(f'  RMS scatter     : {np.std(ratios_arr):.6f} ({np.std(ratios_arr) / np.mean(ratios_arr) * 100:.2f}%)')
        return LightCurve(times=np.array(times_list, dtype=np.float64), fluxes=ratios_arr.astype(np.float64), errors=np.array(ratio_errs_list, dtype=np.float64), target_fluxes=np.array(target_fluxes_list, dtype=np.float64), reference_fluxes=np.array(reference_fluxes_list, dtype=np.float64), centroids=np.array(centroids_list, dtype=np.float64), valid_frames=np.array(valid_frames_list, dtype=bool))

def normalize_lightcurve(fluxes: NDArray[np.float64], errors: NDArray[np.float64], method: str='median') -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    if method == 'median':
        baseline = float(np.median(fluxes))
    elif method == 'mean':
        baseline = float(np.mean(fluxes))
    else:
        raise ValueError(f"Unknown method: '{method}'. Choose 'median' or 'mean'.")
    return (fluxes / baseline, errors / baseline)
