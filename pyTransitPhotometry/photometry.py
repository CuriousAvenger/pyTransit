"""
Aperture photometry routines.

REFACTOR:
  - ``estimate_2d_background`` and ``build_epsf`` / ``run_psf_photometry``
    have been extracted to the dedicated ``background`` and ``psf`` modules.
    This module now has a single responsibility: aperture-based flux
    measurement including centroid refinement.
  - ``PhotometryConfig`` renamed to ``ApertureConfig`` to eliminate the name
    collision with the ``PhotometryConfig`` dataclass in ``config.py``.
    The pipeline imports it under the alias ``PhotConfig``, so the rename is
    transparent to pipeline code.  External callers who imported
    ``PhotometryConfig`` from this module should update to ``ApertureConfig``.
  - Full PEP 484 / ``numpy.typing.NDArray`` type annotations.
  - NumPy docstring format on every public symbol.

Public API
----------
refine_centroid(image, initial_position, box_size, centroid_func)
optimize_aperture_radius(image, position, radii, annulus_inner, annulus_outer, ...)
measure_flux(image, position, aperture_radius, annulus_inner, annulus_outer, ...)
ApertureConfig
"""

import warnings
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.centroids import centroid_2dg, centroid_com, centroid_sources
from photutils.utils import calc_total_error


# ── Centroid refinement ────────────────────────────────────────────────────────


def refine_centroid(
    image: NDArray[np.float32],
    initial_position: Tuple[float, float],
    box_size: int = 51,
    centroid_func=centroid_2dg,
) -> Tuple[float, float]:
    """
    Refine a star centroid using 2-D Gaussian fitting with a COM fallback.

    Parameters
    ----------
    image : NDArray[np.float32]
        2-D science image.
    initial_position : tuple of float
        Initial (x, y) guess for the centroid position.
    box_size : int, optional
        Size of the cutout used for centroid fitting (default: 51).
        Forced to the nearest odd integer if even.
    centroid_func : callable, optional
        photutils centroid function (default: ``centroid_2dg`` for 2-D
        Gaussian).

    Returns
    -------
    x_centroid : float
        Refined x position.
    y_centroid : float
        Refined y position.

    Notes
    -----
    The photutils ``centroid_2dg`` fitter emits an
    ``AstropyUserWarning("fit may not have converged")`` for some frames.
    This warning is suppressed inside the function; if the fitted position
    is NaN or out-of-bounds the function falls back to centre-of-mass
    (``centroid_com``), which always converges.  If that also fails the
    original *initial_position* is returned unchanged.

    Typical centroid precision: 0.01–0.1 pixels for bright, isolated stars.

    Examples
    --------
    >>> x, y = refine_centroid(frame, (1028.3, 876.1), box_size=51)
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


# ── Aperture optimisation ──────────────────────────────────────────────────────


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
    image : NDArray[np.float32]
        2-D science image.
    position : tuple of float
        Star centroid (x, y).
    radii : NDArray[np.float64]
        Candidate aperture radii to evaluate (e.g., ``np.arange(3, 20)``).
    annulus_inner : float
        Inner radius of the background annulus (pixels).
    annulus_outer : float
        Outer radius of the background annulus (pixels).
    ccd_gain : float, optional
        CCD gain in e⁻/ADU (default: 1.0).
    return_snr_curve : bool, optional
        If True, also return ``(radii, snr_values)`` (default: False).

    Returns
    -------
    optimal_radius : float
        Radius that maximises SNR.
    (radii, snr_values) : tuple, optional
        Full SNR curve if *return_snr_curve* is True.

    Notes
    -----
    SNR model (CCD equation):

    .. math::
        \\text{SNR} = \\frac{S}{\\sqrt{S/g + N_{pix}\\,σ_{sky}^2}}

    where :math:`S` is the net stellar signal, :math:`g` is the CCD gain,
    and :math:`σ_{sky}` is the background standard deviation.

    Typical optimal radius: 1–2 × FWHM.

    Examples
    --------
    >>> r_opt = optimize_aperture_radius(
    ...     frame, (1028, 876), np.arange(3, 20),
    ...     annulus_inner=40, annulus_outer=60
    ... )
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


# ── Core flux measurement ──────────────────────────────────────────────────────


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
    Measure background-subtracted flux with full uncertainty propagation.

    Parameters
    ----------
    image : NDArray[np.float32]
        2-D science image.
    position : tuple of float
        Initial star centroid (x, y); refined internally via
        :func:`refine_centroid`.
    aperture_radius : float
        Photometry aperture radius in pixels.
    annulus_inner : float
        Inner radius of the background annulus in pixels.
    annulus_outer : float
        Outer radius of the background annulus in pixels.
    ccd_gain : float, optional
        CCD gain in e⁻/ADU (default: 1.0).
    error_map : NDArray[np.float64], optional
        Pre-computed total error map from ``calc_total_error``.  Computed
        internally if not supplied.

    Returns
    -------
    result : dict
        Keys:

        - ``'flux'`` — background-subtracted flux (ADU)
        - ``'flux_err'`` — 1-σ flux uncertainty (ADU)
        - ``'background_mean'`` — median sky background per pixel (ADU)
        - ``'background_std'`` — standard deviation of sky background (ADU)
        - ``'snr'`` — signal-to-noise ratio
        - ``'aperture_sum'`` — raw aperture sum before background subtraction
        - ``'centroid'`` — refined (x, y) centroid tuple

    Raises
    ------
    ValueError
        If the background annulus at *position* is empty.

    Notes
    -----
    Background is estimated with the robust median (resistant to cosmic rays
    and hot pixels within the annulus).

    Examples
    --------
    >>> result = measure_flux(
    ...     frame, (1028.3, 876.1),
    ...     aperture_radius=6.0, annulus_inner=40, annulus_outer=60
    ... )
    >>> print(f"Flux: {result['flux']:.0f} ± {result['flux_err']:.0f}")
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


# ── Convenience wrapper ────────────────────────────────────────────────────────


class ApertureConfig:
    """
    Convenience wrapper that bundles aperture geometry for repeated calls.

    Previously named ``PhotometryConfig`` in this module.  Renamed to
    ``ApertureConfig`` to avoid the name collision with the
    ``PhotometryConfig`` dataclass in ``config.py``.  The pipeline already
    imported this under the alias ``PhotConfig`` so the rename is
    transparent there.

    Parameters
    ----------
    aperture_radius : float
        Photometry aperture radius in pixels.
    annulus_inner : float
        Inner radius of background annulus (must be > *aperture_radius*).
    annulus_outer : float
        Outer radius of background annulus (must be > *annulus_inner*).
    ccd_gain : float, optional
        CCD gain in e⁻/ADU (default: 1.0).
    centroid_box_size : int, optional
        Box size for centroid refinement (default: 51).

    Raises
    ------
    ValueError
        If the aperture / annulus geometry is invalid.

    Examples
    --------
    >>> cfg = ApertureConfig(
    ...     aperture_radius=6.0, annulus_inner=40.0, annulus_outer=60.0
    ... )
    >>> result = cfg.measure_flux(frame, (1028, 876))
    """

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
        """
        Measure flux at *position* using this configuration.

        Parameters
        ----------
        image : NDArray[np.float32]
            2-D science image.
        position : tuple of float
            Initial (x, y) centroid guess.

        Returns
        -------
        result : dict
            See :func:`measure_flux`.
        """
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


# Backward-compatibility alias — old name still importable but deprecated.
# DEAD CODE: remove in v2.0 once all callers have been updated.
PhotometryConfig = ApertureConfig
