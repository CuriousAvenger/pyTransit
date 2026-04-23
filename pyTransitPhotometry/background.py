"""
Background estimation routines for CCD science images.

Public API
----------
estimate_background(image, sample_size, method)
estimate_2d_background(image, box_size, filter_size, sigma_clip_val, method)
"""

import warnings
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


# ── Simple corner / MAD estimator ─────────────────────────────────────────────


def estimate_background(
    image: NDArray[np.float32],
    sample_size: int = 100,
    method: str = "corners",
) -> Tuple[float, float]:
    """
    Estimate background level and noise from a 2-D science image.

    A fast, assumption-light estimator suited for computing the background
    standard deviation used by :func:`~detection.detect_sources` in
    sigma-based threshold mode.  For spatially varying backgrounds use
    :func:`estimate_2d_background` instead.

    Parameters
    ----------
    image : NDArray[np.float32]
        2-D science image array.
    sample_size : int, optional
        Number of pixels to sample along each edge for the ``'corners'``
        method (default: 100).
    method : str, optional
        ``'corners'`` — samples four ``sample_size × sample_size`` corner
        patches (fast, avoids stellar PSF halos near image centre).
        ``'median'`` — uses global median absolute deviation, robust to
        bright stars that cover a small fraction of the frame.

    Returns
    -------
    background_mean : float
        Estimated sky background level (ADU).
    background_std : float
        Estimated background standard deviation (ADU).

    Raises
    ------
    ValueError
        If *method* is not ``'corners'`` or ``'median'``.

    Examples
    --------
    >>> bg_mean, bg_std = estimate_background(image, sample_size=100)
    >>> sources = detect_sources(image, threshold=5, threshold_type='sigma',
    ...                          background_std=bg_std)
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")

    if method == "corners":
        h, w = image.shape
        s = sample_size
        corners = [
            image[:s, :s],
            image[:s, -s:],
            image[-s:, :s],
            image[-s:, -s:],
        ]
        combined = np.concatenate([c.ravel() for c in corners])
        return float(np.mean(combined)), float(np.std(combined))

    elif method == "median":
        median = np.median(image)
        mad = np.median(np.abs(image - median))
        std = 1.4826 * mad  # consistent with Gaussian σ
        return float(median), float(std)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'corners' or 'median'.")


# ── Spatially varying 2-D background ──────────────────────────────────────────


def estimate_2d_background(
    image: NDArray[np.float32],
    box_size: int = 64,
    filter_size: int = 3,
    sigma_clip_val: float = 3.0,
    method: str = "background2d",
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Estimate a spatially varying 2-D background map.

    Corrects for uneven illumination and atmospheric intensity gradients
    across the detector—effects that a single scalar sky level cannot
    capture.

    Parameters
    ----------
    image : NDArray[np.float32]
        2-D science image.
    box_size : int, optional
        Tile size (pixels) for the mesh-based ``'background2d'`` method
        (default: 64).
    filter_size : int, optional
        Median filter window applied to the background mesh (default: 3).
    sigma_clip_val : float, optional
        Sigma threshold for masking stellar sources before background
        estimation (default: 3.0).
    method : str, optional
        ``'background2d'`` — photutils mesh-based Background2D (default,
        recommended).
        ``'polynomial'`` — global third-order Polynomial2D fit; useful for
        severe large-scale illumination gradients that exceed the tile size.

    Returns
    -------
    background : NDArray[np.float32]
        2-D background map, same shape as *image*.
    background_rms : NDArray[np.float32]
        2-D map of background RMS noise.

    Raises
    ------
    ValueError
        If *method* is not ``'background2d'`` or ``'polynomial'``.

    Notes
    -----
    The mesh-based ``'background2d'`` method tiles the image into
    ``box_size × box_size`` cells, estimates the sky in each cell with a
    sigma-clipped median, and interpolates to produce a smooth map.  The
    polynomial method fits a degree-3 2-D polynomial to sigma-clipped
    background pixels — best for frames with severe atmospheric gradients.

    Examples
    --------
    >>> bkg, bkg_rms = estimate_2d_background(frame, box_size=64)
    >>> sky_subtracted = frame - bkg
    """
    if method == "background2d":
        from astropy.stats import SigmaClip
        from photutils.background import Background2D, MedianBackground

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
        from astropy.modeling.fitting import LevMarLSQFitter
        from astropy.modeling.models import Polynomial2D
        from astropy.stats import sigma_clip as astropy_sigma_clip

        y_idx, x_idx = np.mgrid[: image.shape[0], : image.shape[1]]
        clipped = astropy_sigma_clip(image, sigma=sigma_clip_val, maxiters=5, masked=True)
        valid = ~clipped.mask

        # PERF: downsample to reduce the number of points passed to the fitter
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
            f"Unknown background method '{method}'. "
            "Choose 'background2d' or 'polynomial'."
        )
