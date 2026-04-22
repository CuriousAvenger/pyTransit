"""
Light curve construction via differential photometry.

Implements:
- Multi-frame photometry extraction
- Differential photometry with weighted reference stars
- Error propagation
"""

import numpy as np
from typing import List, Tuple
import warnings


def differential_photometry(
    target_flux: float,
    target_err: float,
    reference_fluxes: np.ndarray,
    reference_errs: np.ndarray,
    weighting: str = "inverse_variance",
) -> Tuple[float, float]:
    """
    Compute differential photometry ratio with error propagation.

    Parameters
    ----------
    target_flux : float
        Target star flux
    target_err : float
        Target star flux uncertainty
    reference_fluxes : np.ndarray
        Array of reference star fluxes
    reference_errs : np.ndarray
        Array of reference star uncertainties
    weighting : str, optional
        Method for combining references:
        - 'inverse_variance' (default): weight by 1/σ²
        - 'equal': simple average

    Returns
    -------
    ratio : float
        Target flux / reference flux
    ratio_err : float
        Propagated uncertainty on ratio

    Notes
    -----
    Differential photometry divides target flux by reference ensemble,
    removing common systematic effects (airmass, clouds, transparency).

    For weighted mean of references:
        R = Σ(w_i × F_i) / Σ(w_i)  where w_i = 1/σ_i²

    Error propagation:
        σ_ratio² = ratio² × [(σ_target/target)² + (σ_ref/ref)²]

    Examples
    --------
    >>> ratio, ratio_err = differential_photometry(
    ...     target_flux=50000, target_err=100,
    ...     reference_fluxes=np.array([40000, 45000]),
    ...     reference_errs=np.array([80, 90])
    ... )
    """
    # Validate inputs
    if len(reference_fluxes) == 0:
        raise ValueError("No reference stars provided")

    if len(reference_fluxes) != len(reference_errs):
        raise ValueError("reference_fluxes and reference_errs must have same length")

    # Check for invalid fluxes
    if target_flux <= 0:
        warnings.warn(f"Target flux non-positive: {target_flux}")
        return np.nan, np.nan

    valid_refs = (reference_fluxes > 0) & (reference_errs > 0)
    if not np.any(valid_refs):
        warnings.warn("No valid reference stars (all fluxes <= 0)")
        return np.nan, np.nan

    ref_fluxes = reference_fluxes[valid_refs]
    ref_errs = reference_errs[valid_refs]

    # Compute weighted reference flux
    if weighting == "inverse_variance":
        weights = 1.0 / ref_errs**2
        weighted_ref_flux = np.sum(ref_fluxes * weights) / np.sum(weights)
        weighted_ref_err = np.sqrt(1.0 / np.sum(weights))

    elif weighting == "equal":
        weighted_ref_flux = np.mean(ref_fluxes)
        weighted_ref_err = np.sqrt(np.sum(ref_errs**2)) / len(ref_errs)

    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    # Compute ratio
    ratio = target_flux / weighted_ref_flux

    # Propagate uncertainties
    # σ_ratio² = ratio² × [(σ_target/F_target)² + (σ_ref/F_ref)²]
    ratio_err = ratio * np.sqrt(
        (target_err / target_flux) ** 2 + (weighted_ref_err / weighted_ref_flux) ** 2
    )

    return float(ratio), float(ratio_err)


class LightCurveBuilder:
    """
    Build a differential photometry light curve from multi-frame data.

    Parameters
    ----------
    target_index : int
        Index of target star in source list
    reference_indices : list of int
        Indices of reference stars
    weighting : str, optional
        Reference combination method (default: 'inverse_variance')

    Examples
    --------
    >>> builder = LightCurveBuilder(
    ...     target_index=2,
    ...     reference_indices=[0, 1],
    ...     weighting='inverse_variance'
    ... )
    >>> lc = builder.build(calibrated_images, headers, sources_list, phot_config)
    """

    def __init__(
        self, target_index: int, reference_indices: List[int], weighting: str = "inverse_variance"
    ):
        self.target_index = target_index
        self.reference_indices = reference_indices
        self.weighting = weighting

        # Validate
        if target_index in reference_indices:
            raise ValueError("Target cannot be in reference list")

        if len(reference_indices) == 0:
            raise ValueError("At least one reference star required")

        print("✓ Light curve builder initialized")
        print(f"  Target: star #{target_index}")
        print(f"  References: stars {reference_indices}")
        print(f"  Weighting: {weighting}")

    def build(
        self,
        images: np.ndarray,
        sources_per_frame: List,
        photometry_func,
        time_extractor,
        verbose: bool = True,
    ) -> dict:
        """
        Build light curve from image sequence.

        Parameters
        ----------
        images : np.ndarray
            3D array of calibrated images (n_frames, height, width)
        sources_per_frame : list
            List of source tables, one per frame
        photometry_func : callable
            Function(image, source_idx) -> flux_dict
        time_extractor : callable
            Function(frame_idx) -> time
        verbose : bool, optional
            Print progress (default: True)

        Returns
        -------
        lightcurve : dict
            Dictionary containing:
            - times: observation times
            - fluxes: differential flux ratios
            - errors: flux ratio uncertainties
            - target_fluxes: raw target fluxes
            - reference_fluxes: combined reference fluxes
            - centroids: target centroids per frame
            - valid_frames: boolean mask of successfully processed frames

        Notes
        -----
        Frames are skipped if:
        - Target or reference stars not detected
        - Photometry fails
        - Invalid flux values (negative, NaN)
        """
        n_frames = len(images)

        times = []
        ratios = []
        ratio_errs = []
        target_fluxes_list = []
        reference_fluxes_list = []
        centroids = []
        valid_frames = []

        for i in range(n_frames):
            try:
                image = images[i]
                sources = sources_per_frame[i]
                time = time_extractor(i)

                # Check if all required stars are detected
                n_detected = len(sources)
                required_indices = [self.target_index] + self.reference_indices

                if max(required_indices) >= n_detected:
                    if verbose:
                        print(
                            f"⚠  Frame {i+1}/{n_frames}: Missing stars "
                            f"(only {n_detected} detected), skipping"
                        )
                    valid_frames.append(False)
                    continue

                # Measure target
                target_result = photometry_func(image, self.target_index)
                target_flux = target_result["flux"]
                target_err = target_result["flux_err"]
                target_centroid = target_result["centroid"]

                # Measure references
                ref_fluxes = []
                ref_errs = []
                for ref_idx in self.reference_indices:
                    try:
                        ref_result = photometry_func(image, ref_idx)
                        ref_fluxes.append(ref_result["flux"])
                        ref_errs.append(ref_result["flux_err"])
                    except Exception as e:
                        if verbose:
                            print(
                                f"⚠  Frame {i+1}/{n_frames}: Reference star {ref_idx} failed: {e}"
                            )
                        continue

                if len(ref_fluxes) == 0:
                    if verbose:
                        print(f"⚠  Frame {i+1}/{n_frames}: No valid references, skipping")
                    valid_frames.append(False)
                    continue

                # Compute differential photometry
                ratio, ratio_err = differential_photometry(
                    target_flux,
                    target_err,
                    np.array(ref_fluxes),
                    np.array(ref_errs),
                    weighting=self.weighting,
                )

                # Check for valid result
                if not np.isfinite(ratio) or not np.isfinite(ratio_err):
                    if verbose:
                        print(f"⚠  Frame {i+1}/{n_frames}: Invalid ratio, skipping")
                    valid_frames.append(False)
                    continue

                # Store results
                times.append(time)
                ratios.append(ratio)
                ratio_errs.append(ratio_err)
                target_fluxes_list.append(target_flux)
                reference_fluxes_list.append(np.mean(ref_fluxes))
                centroids.append(target_centroid)
                valid_frames.append(True)

                if verbose and (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{n_frames} frames...")

            except Exception as e:
                if verbose:
                    print(f"⚠  Frame {i+1}/{n_frames}: Error: {e}, skipping")
                valid_frames.append(False)
                continue

        if len(times) == 0:
            raise RuntimeError("No valid frames processed. Check your data and parameters.")

        print(f"\n✓ Light curve built: {len(times)}/{n_frames} frames valid")
        print(f"  Mean flux ratio: {np.mean(ratios):.6f}")
        print(f"  RMS scatter: {np.std(ratios):.6f} ({np.std(ratios)/np.mean(ratios)*100:.2f}%)")

        return {
            "times": np.array(times),
            "fluxes": np.array(ratios),
            "errors": np.array(ratio_errs),
            "target_fluxes": np.array(target_fluxes_list),
            "reference_fluxes": np.array(reference_fluxes_list),
            "centroids": np.array(centroids),
            "valid_frames": np.array(valid_frames),
        }


def normalize_lightcurve(
    fluxes: np.ndarray, errors: np.ndarray, method: str = "median"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize light curve to unity baseline.

    Parameters
    ----------
    fluxes : np.ndarray
        Flux values
    errors : np.ndarray
        Flux uncertainties
    method : str, optional
        'median' (default) or 'mean'

    Returns
    -------
    normalized_fluxes : np.ndarray
        Fluxes divided by baseline
    normalized_errors : np.ndarray
        Scaled uncertainties

    Notes
    -----
    Normalizing to 1.0 makes transit depth directly readable as
    fractional change (e.g., 0.99 = 1% dip).
    """
    if method == "median":
        baseline = np.median(fluxes)
    elif method == "mean":
        baseline = np.mean(fluxes)
    else:
        raise ValueError(f"Unknown method: {method}")

    normalized_fluxes = fluxes / baseline
    normalized_errors = errors / baseline

    return normalized_fluxes, normalized_errors
