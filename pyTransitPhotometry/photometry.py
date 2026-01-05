"""
Aperture photometry routines.

Implements:
- Centroid refinement using 2D Gaussian fitting
- SNR optimization for aperture size selection
- Background-subtracted flux extraction with error propagation
"""

import numpy as np
from photutils.centroids import centroid_sources, centroid_2dg
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.aperture import aperture_photometry
from photutils.utils import calc_total_error
from typing import Tuple, Optional
import warnings


def refine_centroid(
    image: np.ndarray,
    initial_position: Tuple[float, float],
    box_size: int = 51,
    centroid_func=centroid_2dg
) -> Tuple[float, float]:
    """
    Refine star centroid using 2D Gaussian fitting.
    
    Parameters
    ----------
    image : np.ndarray
        2D image
    initial_position : tuple of float
        Initial (x, y) guess for centroid
    box_size : int, optional
        Size of cutout box for centroid fitting (default: 51)
        Should be odd number
    centroid_func : callable, optional
        Centroid function (default: centroid_2dg for 2D Gaussian)
    
    Returns
    -------
    x_centroid : float
        Refined x position
    y_centroid : float
        Refined y position
    
    Notes
    -----
    Uses iterative 2D Gaussian fitting to achieve sub-pixel accuracy.
    Typical precision: 0.01-0.1 pixels for bright, isolated stars.
    """
    x_init, y_init = initial_position
    
    # Ensure box_size is odd
    if box_size % 2 == 0:
        box_size += 1
    
    try:
        x_refined, y_refined = centroid_sources(
            image, [x_init], [y_init],
            box_size=box_size,
            centroid_func=centroid_func
        )
        return float(x_refined[0]), float(y_refined[0])
    
    except Exception as e:
        warnings.warn(f"Centroid refinement failed: {e}. Using initial position.")
        return x_init, y_init


def optimize_aperture_radius(
    image: np.ndarray,
    position: Tuple[float, float],
    radii: np.ndarray,
    annulus_inner: float,
    annulus_outer: float,
    ccd_gain: float = 1.0,
    return_snr_curve: bool = False
) -> float:
    """
    Find optimal aperture radius that maximizes SNR.
    
    Parameters
    ----------
    image : np.ndarray
        2D image
    position : tuple of float
        Star centroid (x, y)
    radii : np.ndarray
        Array of radii to test (e.g., np.arange(3, 20, 1))
    annulus_inner : float
        Inner radius of background annulus
    annulus_outer : float
        Outer radius of background annulus
    ccd_gain : float, optional
        CCD gain in e-/ADU (default: 1.0)
    return_snr_curve : bool, optional
        If True, also return (radii, snr_values)
    
    Returns
    -------
    optimal_radius : float
        Radius that maximizes SNR
    (radii, snr_values) : tuple, optional
        SNR curve if return_snr_curve=True
    
    Notes
    -----
    SNR calculation:
        Signal = aperture_flux - background_per_pixel * n_pixels
        Noise² = Signal*gain + n_pixels*σ_background²
        SNR = Signal / Noise
    
    Optimal radius typically 1-2× FWHM, balancing:
    - Larger aperture: collects more photons
    - Smaller aperture: less sky noise
    
    Examples
    --------
    >>> radii_test = np.arange(3, 20, 1)
    >>> optimal_r = optimize_aperture_radius(
    ...     image, (x, y), radii_test,
    ...     annulus_inner=40, annulus_outer=60,
    ...     ccd_gain=1.5
    ... )
    """
    bkg_annulus = CircularAnnulus(position, r_in=annulus_inner, r_out=annulus_outer)
    
    # Compute background statistics
    annulus_masks = bkg_annulus.to_mask(method='center')
    annulus_mask = annulus_masks[0] if isinstance(annulus_masks, list) else annulus_masks
    annulus_data = annulus_mask.multiply(image)
    annulus_data_1d = annulus_data[annulus_mask.data > 0]
    
    if len(annulus_data_1d) == 0:
        raise ValueError("Background annulus contains no valid pixels")
    
    bkg_mean = np.mean(annulus_data_1d)
    bkg_std = np.std(annulus_data_1d)
    
    snr_list = []
    
    for r in radii:
        aperture = CircularAperture(position, r=r)
        phot_table = aperture_photometry(image, [aperture, bkg_annulus])
        
        # Background-subtracted signal
        aperture_sum = phot_table['aperture_sum_0'][0]
        bkg_sum = phot_table['aperture_sum_1'][0]
        bkg_per_pixel = bkg_sum / bkg_annulus.area
        bkg_in_aperture = bkg_per_pixel * aperture.area
        signal = aperture_sum - bkg_in_aperture
        
        # Noise calculation (CCD equation)
        # Noise² = Poisson(signal+sky) + Npix × σ_sky²
        noise_squared = (
            np.abs(aperture_sum) * ccd_gain +  # Poisson from star + sky
            aperture.area * bkg_std**2  # Background uncertainty
        )
        noise = np.sqrt(noise_squared)
        
        snr = signal / noise if noise > 0 else 0
        snr_list.append(snr)
    
    snr_array = np.array(snr_list)
    best_idx = np.argmax(snr_array)
    optimal_radius = radii[best_idx]
    
    print(f"✓ Optimal aperture radius: {optimal_radius:.1f} px (SNR = {snr_array[best_idx]:.1f})")
    
    if return_snr_curve:
        return optimal_radius, (radii, snr_array)
    else:
        return optimal_radius


def measure_flux(
    image: np.ndarray,
    position: Tuple[float, float],
    aperture_radius: float,
    annulus_inner: float,
    annulus_outer: float,
    ccd_gain: float = 1.0,
    error_map: Optional[np.ndarray] = None
) -> dict:
    """
    Measure background-subtracted flux with uncertainties.
    
    Parameters
    ----------
    image : np.ndarray
        2D image
    position : tuple of float
        Star centroid (x, y)
    aperture_radius : float
        Photometry aperture radius
    annulus_inner : float
        Inner radius of background annulus
    annulus_outer : float
        Outer radius of background annulus
    ccd_gain : float, optional
        CCD gain in e-/ADU (default: 1.0)
    error_map : np.ndarray, optional
        Pre-computed error map from calc_total_error()
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - flux: background-subtracted flux
        - flux_err: flux uncertainty
        - background_mean: background per pixel
        - background_std: background standard deviation
        - snr: signal-to-noise ratio
        - aperture_sum: raw aperture sum
        - centroid: refined (x, y) position
    
    Notes
    -----
    This is the core photometry function. It:
    1. Refines centroid position
    2. Measures raw aperture flux
    3. Estimates local background from annulus
    4. Subtracts background
    5. Computes uncertainties using CCD noise model
    
    Examples
    --------
    >>> result = measure_flux(
    ...     image, (x, y), aperture_radius=8.0,
    ...     annulus_inner=40, annulus_outer=60,
    ...     ccd_gain=1.5
    ... )
    >>> print(f"Flux: {result['flux']:.1f} ± {result['flux_err']:.1f}")
    """
    # Refine centroid
    x_center, y_center = refine_centroid(image, position)
    refined_position = (x_center, y_center)
    
    # Define apertures
    aperture = CircularAperture(refined_position, r=aperture_radius)
    annulus = CircularAnnulus(refined_position, r_in=annulus_inner, r_out=annulus_outer)
    
    # Get background statistics
    annulus_masks = annulus.to_mask(method='center')
    annulus_mask = annulus_masks[0] if isinstance(annulus_masks, list) else annulus_masks
    annulus_data = annulus_mask.multiply(image)
    annulus_data_1d = annulus_data[annulus_mask.data > 0]
    
    if len(annulus_data_1d) == 0:
        raise ValueError(f"Background annulus empty at position {position}")
    
    bkg_mean = np.mean(annulus_data_1d)
    bkg_std = np.std(annulus_data_1d)
    
    # Perform aperture photometry
    phot_table = aperture_photometry(image, [aperture, annulus])
    
    aperture_sum = phot_table['aperture_sum_0'][0]
    bkg_sum = phot_table['aperture_sum_1'][0]
    
    # Background-subtracted flux
    bkg_per_pixel = bkg_sum / annulus.area
    bkg_in_aperture = bkg_per_pixel * aperture.area
    flux = aperture_sum - bkg_in_aperture
    
    # Compute flux error
    if error_map is None:
        error_map = calc_total_error(image, bkg_std, effective_gain=ccd_gain)
    
    phot_with_err = aperture_photometry(image, aperture, error=error_map)
    flux_err = phot_with_err['aperture_sum_err'][0]
    
    # Compute SNR
    noise_squared = (
        np.abs(aperture_sum) * ccd_gain +
        aperture.area * bkg_std**2
    )
    noise = np.sqrt(noise_squared)
    snr = flux / noise if noise > 0 else 0
    
    return {
        'flux': float(flux),
        'flux_err': float(flux_err),
        'background_mean': float(bkg_mean),
        'background_std': float(bkg_std),
        'snr': float(snr),
        'aperture_sum': float(aperture_sum),
        'centroid': refined_position
    }


class PhotometryConfig:
    """
    Configuration container for aperture photometry.
    
    Parameters
    ----------
    aperture_radius : float
        Photometry aperture radius in pixels
    annulus_inner : float
        Inner radius of background annulus
    annulus_outer : float
        Outer radius of background annulus
    ccd_gain : float, optional
        CCD gain in e-/ADU (default: 1.0)
    centroid_box_size : int, optional
        Box size for centroid refinement (default: 51)
    
    Examples
    --------
    >>> config = PhotometryConfig(
    ...     aperture_radius=8.0,
    ...     annulus_inner=40.0,
    ...     annulus_outer=60.0,
    ...     ccd_gain=1.5
    ... )
    """
    
    def __init__(
        self,
        aperture_radius: float,
        annulus_inner: float,
        annulus_outer: float,
        ccd_gain: float = 1.0,
        centroid_box_size: int = 51
    ):
        # Validation
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
    
    def measure_flux(self, image: np.ndarray, position: Tuple[float, float]) -> dict:
        """Convenience method to measure flux with this configuration."""
        return measure_flux(
            image, position,
            self.aperture_radius, self.annulus_inner, self.annulus_outer,
            self.ccd_gain
        )
    
    def __repr__(self):
        return (
            f"PhotometryConfig(aperture_r={self.aperture_radius:.1f}, "
            f"annulus={self.annulus_inner:.1f}-{self.annulus_outer:.1f}, "
            f"gain={self.ccd_gain:.2f})"
        )
