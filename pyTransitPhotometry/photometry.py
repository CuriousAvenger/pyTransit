"""Aperture photometry routines."""

import warnings
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.centroids import centroid_2dg, centroid_com, centroid_sources
from photutils.utils import calc_total_error



def refine_centroid(
    image: NDArray[np.float32],
    initial_position: Tuple[float, float],
    box_size: int = 51,
    centroid_func=centroid_2dg,
) -> Tuple[float, float]:
    """
    Refine a star centroid; falls back to COM then initial position if fitting fails.

    Suppresses photutils convergence warnings internally.
    """
    x_init, y_init = initial_position
    if box_size % 2 == 0:
        box_size += 1

    h, w = image.shape
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*fit may not have converged.*",
                category=Warning,
            )
            x_refined, y_refined = centroid_sources(
                image, [x_init], [y_init], box_size=box_size, centroid_func=centroid_func
            )
        x_ref, y_ref = float(x_refined[0]), float(y_refined[0])

        if np.isnan(x_ref) or np.isnan(y_ref) or not (0 <= x_ref < w) or not (0 <= y_ref < h):
            # Fall back to centre-of-mass — always converges
            x_com, y_com = centroid_sources(
                image, [x_init], [y_init], box_size=box_size, centroid_func=centroid_com
            )
            x_ref, y_ref = float(x_com[0]), float(y_com[0])
            if np.isnan(x_ref) or np.isnan(y_ref):
                return x_init, y_init

        return x_ref, y_ref

    except Exception as exc:
        warnings.warn(f"Centroid refinement failed: {exc}. Using initial position.")
        return x_init, y_init



def optimize_aperture_radius(
    image: NDArray[np.float32],
    position: Tuple[float, float],
    radii: NDArray[np.float64],
    annulus_inner: float,
    annulus_outer: float,
    ccd_gain: float = 1.0,
    return_snr_curve: bool = False,
) -> float:
    """
    Find the aperture radius that maximises SNR for a given star.

    Parameters
    ----------
    return_snr_curve : bool
        If True, also return ``(radii, snr_values)``.
    """
    bkg_annulus = CircularAnnulus(position, r_in=annulus_inner, r_out=annulus_outer)
    annulus_mask = bkg_annulus.to_mask(method="center")
    if isinstance(annulus_mask, list):
        annulus_mask = annulus_mask[0]
    annulus_data_1d = annulus_mask.multiply(image)[annulus_mask.data > 0]
    if len(annulus_data_1d) == 0:
        raise ValueError("Background annulus contains no valid pixels")
    bkg_std = float(np.std(annulus_data_1d))

    snr_list = []
    for r in radii:
        aperture = CircularAperture(position, r=r)
        phot_table = aperture_photometry(image, [aperture, bkg_annulus])
        aperture_sum = float(phot_table["aperture_sum_0"][0])
        bkg_sum = float(phot_table["aperture_sum_1"][0])
        bkg_per_pixel = bkg_sum / bkg_annulus.area
        signal = aperture_sum - bkg_per_pixel * aperture.area
        noise_sq = np.abs(aperture_sum) * ccd_gain + aperture.area * bkg_std**2
        noise = np.sqrt(noise_sq)
        snr_list.append(signal / noise if noise > 0 else 0.0)

    snr_array = np.array(snr_list)
    best_idx = int(np.argmax(snr_array))
    optimal_radius = float(radii[best_idx])
    print(f"✓ Optimal aperture radius: {optimal_radius:.1f} px (SNR = {snr_array[best_idx]:.1f})")

    if return_snr_curve:
        return optimal_radius, (radii, snr_array)
    return optimal_radius



def measure_flux(
    image: NDArray[np.float32],
    position: Tuple[float, float],
    aperture_radius: float,
    annulus_inner: float,
    annulus_outer: float,
    ccd_gain: float = 1.0,
    error_map: Optional[NDArray[np.float64]] = None,
) -> dict:
    """
    Measure background-subtracted flux with uncertainty propagation.

    Returns
    -------
    result : dict
        Keys: ``'flux'``, ``'flux_err'``, ``'background_mean'``,
        ``'background_std'``, ``'snr'``, ``'aperture_sum'``, ``'centroid'``.

    Raises
    ------
    ValueError
        If the background annulus at *position* is empty.
    """
    x_center, y_center = refine_centroid(image, position)
    refined_position = (x_center, y_center)

    aperture = CircularAperture(refined_position, r=aperture_radius)
    annulus = CircularAnnulus(refined_position, r_in=annulus_inner, r_out=annulus_outer)

    annulus_mask = annulus.to_mask(method="center")
    if isinstance(annulus_mask, list):
        annulus_mask = annulus_mask[0]
    annulus_data_1d = annulus_mask.multiply(image)[annulus_mask.data > 0]

    if len(annulus_data_1d) == 0:
        raise ValueError(f"Background annulus empty at position {position}")

    bkg_mean = float(np.median(annulus_data_1d))  # robust against hot pixels
    bkg_std = float(np.std(annulus_data_1d))

    phot_table = aperture_photometry(image, [aperture, annulus])
    aperture_sum = float(phot_table["aperture_sum_0"][0])

    bkg_in_aperture = bkg_mean * aperture.area
    flux = aperture_sum - bkg_in_aperture

    if error_map is None:
        bkg_error_map = np.full_like(image, bkg_std, dtype=float)
        error_map = calc_total_error(image, bkg_error_map, effective_gain=ccd_gain)

    phot_with_err = aperture_photometry(image, aperture, error=error_map)
    flux_err = float(phot_with_err["aperture_sum_err"][0])

    noise_sq = np.abs(aperture_sum) * ccd_gain + aperture.area * bkg_std**2
    noise = float(np.sqrt(noise_sq))
    snr = flux / noise if noise > 0 else 0.0

    return {
        "flux": float(flux),
        "flux_err": float(flux_err),
        "background_mean": bkg_mean,
        "background_std": bkg_std,
        "snr": snr,
        "aperture_sum": aperture_sum,
        "centroid": refined_position,
    }



class ApertureConfig:
    """Bundles aperture geometry for repeated :func:`measure_flux` calls."""

    def __init__(
        self,
        aperture_radius: float,
        annulus_inner: float,
        annulus_outer: float,
        ccd_gain: float = 1.0,
        centroid_box_size: int = 51,
    ) -> None:
        if aperture_radius <= 0:
            raise ValueError("aperture_radius must be positive")
        if annulus_inner <= aperture_radius:
            raise ValueError("annulus_inner must be > aperture_radius")
        if annulus_outer <= annulus_inner:
            raise ValueError("annulus_outer must be > annulus_inner")
        if ccd_gain <= 0:
            raise ValueError("ccd_gain must be positive")

        self.aperture_radius = aperture_radius
        self.annulus_inner = annulus_inner
        self.annulus_outer = annulus_outer
        self.ccd_gain = ccd_gain
        self.centroid_box_size = centroid_box_size

    def measure_flux(
        self, image: NDArray[np.float32], position: Tuple[float, float]
    ) -> dict:
        """Measure flux at *position* using this configuration."""
        return measure_flux(
            image,
            position,
            self.aperture_radius,
            self.annulus_inner,
            self.annulus_outer,
            self.ccd_gain,
        )

    def __repr__(self) -> str:
        return (
            f"ApertureConfig(aperture_r={self.aperture_radius:.1f}, "
            f"annulus={self.annulus_inner:.1f}–{self.annulus_outer:.1f} px, "
            f"gain={self.ccd_gain:.2f})"
        )


