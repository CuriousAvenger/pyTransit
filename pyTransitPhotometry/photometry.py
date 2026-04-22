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
    centroid_func=centroid_2dg,
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

    h, w = image.shape
    try:
        x_refined, y_refined = centroid_sources(
            image, [x_init], [y_init], box_size=box_size, centroid_func=centroid_func
        )
        x_ref, y_ref = float(x_refined[0]), float(y_refined[0])
        # Guard against NaN or out-of-bounds divergence (e.g. featureless backgrounds)
        if np.isnan(x_ref) or np.isnan(y_ref) or not (0 <= x_ref < w) or not (0 <= y_ref < h):
            return x_init, y_init
        return x_ref, y_ref

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
    return_snr_curve: bool = False,
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
    annulus_masks = bkg_annulus.to_mask(method="center")
    annulus_mask = annulus_masks[0] if isinstance(annulus_masks, list) else annulus_masks
    annulus_data = annulus_mask.multiply(image)
    annulus_data_1d = annulus_data[annulus_mask.data > 0]

    if len(annulus_data_1d) == 0:
        raise ValueError("Background annulus contains no valid pixels")

    bkg_std = np.std(annulus_data_1d)

    snr_list = []

    for r in radii:
        aperture = CircularAperture(position, r=r)
        phot_table = aperture_photometry(image, [aperture, bkg_annulus])

        # Background-subtracted signal
        aperture_sum = phot_table["aperture_sum_0"][0]
        bkg_sum = phot_table["aperture_sum_1"][0]
        bkg_per_pixel = bkg_sum / bkg_annulus.area
        bkg_in_aperture = bkg_per_pixel * aperture.area
        signal = aperture_sum - bkg_in_aperture

        # Noise calculation (CCD equation)
        # Noise² = Poisson(signal+sky) + Npix × σ_sky²
        noise_squared = (
            np.abs(aperture_sum) * ccd_gain  # Poisson from star + sky
            + aperture.area * bkg_std**2  # Background uncertainty
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
    error_map: Optional[np.ndarray] = None,
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
    annulus_masks = annulus.to_mask(method="center")
    annulus_mask = annulus_masks[0] if isinstance(annulus_masks, list) else annulus_masks
    annulus_data = annulus_mask.multiply(image)
    annulus_data_1d = annulus_data[annulus_mask.data > 0]

    if len(annulus_data_1d) == 0:
        raise ValueError(f"Background annulus empty at position {position}")

    bkg_mean = np.mean(annulus_data_1d)
    bkg_std = np.std(annulus_data_1d)

    # Perform aperture photometry
    phot_table = aperture_photometry(image, [aperture, annulus])

    aperture_sum = phot_table["aperture_sum_0"][0]
    bkg_sum = phot_table["aperture_sum_1"][0]

    # Background-subtracted flux
    bkg_per_pixel = bkg_sum / annulus.area
    bkg_in_aperture = bkg_per_pixel * aperture.area
    flux = aperture_sum - bkg_in_aperture

    # Compute flux error
    # photutils >=3.0 requires bkg_error to be a 2D array matching image shape
    if error_map is None:
        bkg_error_map = np.full_like(image, bkg_std, dtype=float)
        error_map = calc_total_error(image, bkg_error_map, effective_gain=ccd_gain)

    phot_with_err = aperture_photometry(image, aperture, error=error_map)
    flux_err = phot_with_err["aperture_sum_err"][0]

    # Compute SNR
    noise_squared = np.abs(aperture_sum) * ccd_gain + aperture.area * bkg_std**2
    noise = np.sqrt(noise_squared)
    snr = flux / noise if noise > 0 else 0

    return {
        "flux": float(flux),
        "flux_err": float(flux_err),
        "background_mean": float(bkg_mean),
        "background_std": float(bkg_std),
        "snr": float(snr),
        "aperture_sum": float(aperture_sum),
        "centroid": refined_position,
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
        centroid_box_size: int = 51,
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
            image,
            position,
            self.aperture_radius,
            self.annulus_inner,
            self.annulus_outer,
            self.ccd_gain,
        )

    def __repr__(self):
        return (
            f"PhotometryConfig(aperture_r={self.aperture_radius:.1f}, "
            f"annulus={self.annulus_inner:.1f}-{self.annulus_outer:.1f}, "
            f"gain={self.ccd_gain:.2f})"
        )


# ============================================================================
# 2D BACKGROUND ESTIMATION
# ============================================================================


def estimate_2d_background(
    image: np.ndarray,
    box_size: int = 64,
    filter_size: int = 3,
    sigma_clip_val: float = 3.0,
    method: str = "background2d",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate a spatially varying 2D background to correct uneven illumination
    and atmospheric intensity gradients across the detector.

    Parameters
    ----------
    image : np.ndarray
        2D science image.
    box_size : int, optional
        Tile size (pixels) for the mesh-based ``background2d`` method (default: 64).
    filter_size : int, optional
        Median filter window applied to the background mesh (default: 3).
    sigma_clip_val : float, optional
        Sigma threshold for masking stellar sources before background estimation
        (default: 3.0).
    method : str, optional
        ``'background2d'`` — photutils mesh-based Background2D (default, recommended).
        ``'polynomial'`` — global astropy Polynomial2D fit (degree 3); useful for
        severe large-scale illumination gradients.

    Returns
    -------
    background : np.ndarray
        2D background map, same shape as *image*.
    background_rms : np.ndarray
        2D map of background RMS noise.

    Notes
    -----
    The mesh-based ``background2d`` tiles the image into *box_size* × *box_size*
    cells, estimates the sky in each cell with a sigma-clipped median, and
    interpolates to produce a smooth map.  The polynomial method fits a
    third-order 2D polynomial to sigma-clipped background pixels—best for
    frames with severe atmospheric gradients that vary on scales larger than
    the tile size.
    """
    if method == "background2d":
        from photutils.background import Background2D, MedianBackground
        from astropy.stats import SigmaClip

        sigma_clip = SigmaClip(sigma=sigma_clip_val, maxiters=5)
        bkg_estimator = MedianBackground()
        bkg = Background2D(
            image,
            box_size=(box_size, box_size),
            filter_size=(filter_size, filter_size),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
        )
        return bkg.background.astype(np.float32), bkg.background_rms.astype(np.float32)

    elif method == "polynomial":
        from astropy.modeling.models import Polynomial2D
        from astropy.modeling.fitting import LevMarLSQFitter
        from astropy.stats import sigma_clip as astropy_sigma_clip

        y_idx, x_idx = np.mgrid[: image.shape[0], : image.shape[1]]

        # Sigma-clip stellar sources so they do not bias the polynomial fit
        clipped = astropy_sigma_clip(image, sigma=sigma_clip_val, maxiters=5, masked=True)
        valid = ~clipped.mask

        # Downsample for efficiency
        step = max(1, min(image.shape) // 64)
        x_ds = x_idx[::step, ::step][valid[::step, ::step]].ravel()
        y_ds = y_idx[::step, ::step][valid[::step, ::step]].ravel()
        z_ds = image[::step, ::step][valid[::step, ::step]].ravel()

        poly_init = Polynomial2D(degree=3)
        fitter = LevMarLSQFitter()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            poly_fit = fitter(poly_init, x_ds, y_ds, z_ds)

        background = poly_fit(x_idx, y_idx).astype(np.float32)
        residuals = image - background
        bkg_rms_val = float(np.std(residuals[valid]))
        background_rms = np.full_like(background, bkg_rms_val)

        print(
            f"✓ Polynomial 2D background: mean = {background.mean():.1f}, "
            f"RMS = {bkg_rms_val:.2f}"
        )
        return background, background_rms

    else:
        raise ValueError(
            f"Unknown background method '{method}'. " "Choose 'background2d' or 'polynomial'."
        )


# ============================================================================
# EMPIRICAL PSF CONSTRUCTION
# ============================================================================


def build_epsf(
    image: np.ndarray,
    positions: list,
    size: int = 25,
    oversampling: int = 4,
    maxiters: int = 10,
    sigma_clip_val: float = 3.0,
) -> object:
    """
    Build an empirical PSF (ePSF) from isolated bright stars.

    Parameters
    ----------
    image : np.ndarray
        2D science image (background-subtracted recommended).
    positions : list of (float, float)
        (x, y) centroids of stars used to construct the ePSF.
    size : int, optional
        Side length (pixels) of each star cutout; forced to odd (default: 25).
    oversampling : int, optional
        Pixel oversampling factor for the ePSF grid (default: 4).
    maxiters : int, optional
        Maximum ePSF building iterations (default: 10).
    sigma_clip_val : float, optional
        Sigma clipping during ePSF construction (default: 3.0).

    Returns
    -------
    epsf : photutils.psf.EPSFModel
        Oversampled empirical PSF model ready for :func:`run_psf_photometry`.

    Notes
    -----
    Implements the Anderson & King (2000) ePSF algorithm via photutils.
    Pass 10–50 well-isolated, unsaturated stars for best results.
    """
    from photutils.psf import EPSFBuilder, extract_stars
    from astropy.nddata import NDData
    from astropy.table import Table
    from astropy.stats import SigmaClip

    if size % 2 == 0:
        size += 1

    nddata = NDData(data=image.astype(float))
    stars_tbl = Table()
    stars_tbl["x"] = [float(p[0]) for p in positions]
    stars_tbl["y"] = [float(p[1]) for p in positions]
    stars = extract_stars(nddata, stars_tbl, size=size)

    if len(stars) == 0:
        raise RuntimeError(
            "No valid star cutouts extracted. "
            "Verify that positions fall within the image bounds."
        )

    sigma_clip = SigmaClip(sigma=sigma_clip_val)
    builder = EPSFBuilder(
        oversampling=oversampling,
        maxiters=maxiters,
        progress_bar=False,
        sigma_clip=sigma_clip,
    )

    epsf, fitted_stars = builder(stars)
    print(
        f"✓ Built ePSF from {len(fitted_stars)} stars "
        f"({oversampling}× oversampling, {maxiters} iterations)"
    )
    return epsf


# ============================================================================
# PSF FITTING PHOTOMETRY
# ============================================================================


def run_psf_photometry(
    image: np.ndarray,
    positions: list,
    epsf,
    fwhm: float = 5.0,
    fit_shape: int = 11,
    background_2d: Optional[np.ndarray] = None,
    ccd_gain: float = 1.0,
) -> list:
    """
    Measure stellar fluxes via PSF fitting, accurately separating blended sources.

    Unlike circular aperture photometry, PSF fitting decomposes the observed
    image into individual stellar profiles, preventing flux dilution from
    unresolved background stars within the photometric aperture.

    Parameters
    ----------
    image : np.ndarray
        2D science image.
    positions : list of (float, float)
        (x, y) centroids for each star to measure.
    epsf : EPSFModel
        Empirical PSF model from :func:`build_epsf`.
    fwhm : float, optional
        Approximate PSF FWHM in pixels (sets aperture_radius; default: 5.0).
    fit_shape : int, optional
        Side length (pixels) of the fitting region per source; forced to odd
        (default: 11).
    background_2d : np.ndarray, optional
        2D background map to subtract before PSF fitting.
    ccd_gain : float, optional
        CCD gain in e⁻/ADU for Poisson error estimation (default: 1.0).

    Returns
    -------
    results : list of dict
        One dict per input position with keys:
        ``'flux'``, ``'flux_err'``, ``'x_fit'``, ``'y_fit'``.

    Notes
    -----
    Subtracts *background_2d* before PSF fitting to isolate stellar signal.
    This mitigates the flux dilution that caused the χ²_red = 2.52 from the
    simple circular aperture approach.
    """
    try:
        from photutils.psf import PSFPhotometry  # photutils >= 1.8
    except ImportError:
        try:
            from photutils.psf import BasicPSFPhotometry as PSFPhotometry  # type: ignore
        except ImportError:
            raise ImportError(
                "photutils >= 1.8 is required for PSF photometry. "
                "Install with: pip install 'photutils>=1.8'"
            )

    from astropy.table import Table

    if fit_shape % 2 == 0:
        fit_shape += 1

    image_work = (image - background_2d) if background_2d is not None else image.copy()
    image_work = image_work.astype(float)

    init_params = Table()
    init_params["x"] = [float(p[0]) for p in positions]
    init_params["y"] = [float(p[1]) for p in positions]

    try:
        psfphot = PSFPhotometry(
            psf_model=epsf,
            fit_shape=(fit_shape, fit_shape),
            aperture_radius=max(3, int(fwhm * 1.5)),
        )
        phot_table = psfphot(image_work, init_params=init_params)
    except Exception as exc:
        raise RuntimeError(f"PSF photometry failed: {exc}") from exc

    results = []
    for i, pos in enumerate(positions):
        if i < len(phot_table):
            row = phot_table[i]
            # Column names differ across photutils versions
            for flux_col in ("flux_fit", "flux", "aperture_sum"):
                if flux_col in phot_table.colnames:
                    flux = float(row[flux_col])
                    break
            else:
                flux = np.nan

            flux_err = np.nan
            for err_col in ("flux_err", "flux_unc"):
                if err_col in phot_table.colnames:
                    flux_err = float(row[err_col])
                    break
            if not np.isfinite(flux_err):
                flux_err = float(np.sqrt(max(flux, 0) / ccd_gain + 1.0))

            x_fit = float(row["x_fit"]) if "x_fit" in phot_table.colnames else pos[0]
            y_fit = float(row["y_fit"]) if "y_fit" in phot_table.colnames else pos[1]

            results.append({"flux": flux, "flux_err": flux_err, "x_fit": x_fit, "y_fit": y_fit})
        else:
            results.append({"flux": np.nan, "flux_err": np.nan, "x_fit": pos[0], "y_fit": pos[1]})

    return results
