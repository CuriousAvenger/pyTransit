"""
Star detection routines using photutils.

REFACTOR:
  - ``estimate_background`` moved to the ``background`` module where it sits
    alongside ``estimate_2d_background``.  A deprecation shim is provided
    here so existing call sites import without errors, but they should
    migrate to ``from pyTransitPhotometry.background import estimate_background``.
  - Full PEP 484 / ``numpy.typing.NDArray`` type annotations.
  - NumPy-format docstrings on every public function.

Public API
----------
detect_sources(image, fwhm, threshold, threshold_type, background_std, ...)
filter_sources(sources, min_sharpness, max_sharpness, max_roundness, min_flux)
select_reference_stars(sources, target_index, n_references, max_separation)
"""

import warnings
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from astropy.table import Table
from photutils.detection import DAOStarFinder


def detect_sources(
    image: NDArray[np.float32],
    fwhm: float = 5.0,
    threshold: float = 10.0,
    threshold_type: str = "absolute",
    background_std: Optional[float] = None,
    exclude_border: bool = True,
    sort_by: str = "flux",
) -> Table:
    """
    Detect point sources in an image using DAOStarFinder.

    Parameters
    ----------
    image : NDArray[np.float32]
        2-D image array.
    fwhm : float, optional
        Full-width at half-maximum of the stellar PSF in pixels (default: 5.0).
    threshold : float, optional
        Detection threshold (default: 10.0).  Interpretation depends on
        *threshold_type*.
    threshold_type : str, optional
        ``'absolute'`` — threshold is in raw counts (ADU).
        ``'sigma'`` — threshold is a multiple of *background_std*.
    background_std : float, optional
        Background standard deviation.  Required when
        *threshold_type* is ``'sigma'``.
    exclude_border : bool, optional
        Exclude sources whose PSF overlaps the image border (default: True).
    sort_by : str, optional
        Sort sources by ``'flux'`` (default, brightest first) or
        ``'sharpness'``.

    Returns
    -------
    sources : astropy.table.Table
        Detected sources with columns:
        ``id``, ``x_centroid``, ``y_centroid``, ``flux``, ``peak``,
        ``sharpness``, ``roundness``, ``npix``.

    Raises
    ------
    ValueError
        If *image* is not 2-D, or *threshold_type* is ``'sigma'`` but
        *background_std* is not provided, or *threshold_type* is unknown.
    RuntimeError
        If no sources are detected above *threshold*.

    Notes
    -----
    DAOStarFinder algorithm:
    1. Convolves the image with a Gaussian kernel (FWHM).
    2. Finds local maxima above *threshold*.
    3. Fits 1-D Gaussian profiles in x / y to refine centroids.
    4. Computes sharpness and roundness for quality filtering.

    Examples
    --------
    >>> sources = detect_sources(image, fwhm=5.0, threshold=10000.0)
    >>> x, y = sources['x_centroid'][0], sources['y_centroid'][0]
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")

    if threshold_type == "sigma":
        if background_std is None:
            raise ValueError("background_std required when threshold_type='sigma'")
        actual_threshold = threshold * background_std
    elif threshold_type == "absolute":
        actual_threshold = threshold
    else:
        raise ValueError(
            f"Unknown threshold_type '{threshold_type}'. "
            "Choose 'absolute' or 'sigma'."
        )

    daofind = DAOStarFinder(
        fwhm=fwhm, threshold=actual_threshold, exclude_border=exclude_border
    )
    sources = daofind(image)

    if sources is None or len(sources) == 0:
        raise RuntimeError(
            f"No sources detected (threshold={actual_threshold:.1f}, fwhm={fwhm}). "
            "Try lowering the threshold or adjusting FWHM."
        )

    # photutils >= 3.0 renamed xcentroid → x_centroid
    if "xcentroid" in sources.colnames:
        sources.rename_column("xcentroid", "x_centroid")
    if "ycentroid" in sources.colnames:
        sources.rename_column("ycentroid", "y_centroid")

    if sort_by == "flux":
        sources.sort("flux", reverse=True)
    elif sort_by == "sharpness":
        sources.sort("sharpness")
    else:
        warnings.warn(f"Unknown sort_by '{sort_by}', using 'flux'")
        sources.sort("flux", reverse=True)

    for col in sources.colnames:
        if col not in ("id", "npix"):
            sources[col].info.format = "%.2f"

    print(f"✓ Detected {len(sources)} sources")
    print(
        f"  Brightest: flux={sources['flux'][0]:.0f}, "
        f"position=({sources['x_centroid'][0]:.1f}, {sources['y_centroid'][0]:.1f})"
    )
    return sources


def filter_sources(
    sources: Table,
    min_sharpness: float = 0.3,
    max_sharpness: float = 1.0,
    max_roundness: float = 0.5,
    min_flux: Optional[float] = None,
) -> Table:
    """
    Filter detected sources by morphological quality metrics.

    Parameters
    ----------
    sources : astropy.table.Table
        Source table from :func:`detect_sources`.
    min_sharpness : float, optional
        Minimum sharpness (default: 0.3).  Low values indicate extended or
        blended sources.
    max_sharpness : float, optional
        Maximum sharpness (default: 1.0).  High values indicate cosmic rays.
    max_roundness : float, optional
        Maximum absolute roundness (default: 0.5).  Measures elongation;
        0 = perfectly round.
    min_flux : float, optional
        Minimum flux threshold; sources below this are rejected.

    Returns
    -------
    filtered : astropy.table.Table
        Filtered source table, sorted identically to *sources*.

    Notes
    -----
    Sharpness: ratio of central pixel to surrounding pixels.
    Roundness: :math:`(2σ_x − 2σ_y)/(2σ_x + 2σ_y)` where σ are Gaussian widths.

    Good stellar sources typically have sharpness 0.4–0.8 and
    |roundness| < 0.3.
    """
    mask = np.ones(len(sources), dtype=bool)

    if "sharpness" in sources.colnames:
        mask &= sources["sharpness"] >= min_sharpness
        mask &= sources["sharpness"] <= max_sharpness

    if "roundness" in sources.colnames:
        mask &= np.abs(sources["roundness"]) <= max_roundness

    if min_flux is not None and "flux" in sources.colnames:
        mask &= sources["flux"] >= min_flux

    filtered = sources[mask]
    n_removed = len(sources) - len(filtered)
    if n_removed > 0:
        print(f"✓ Filtered out {n_removed} sources (quality cuts); {len(filtered)} remaining")

    return filtered


def select_reference_stars(
    sources: Table,
    target_index: int,
    n_references: int = 3,
    max_separation: Optional[float] = None,
) -> Tuple[Tuple[float, float], List[Tuple[float, float]], List[int]]:
    """
    Select reference stars for differential photometry.

    Parameters
    ----------
    sources : astropy.table.Table
        Source table sorted by brightness (brightest first).
    target_index : int
        Index of the target star in *sources*.
    n_references : int, optional
        Number of reference stars to select (default: 3).
    max_separation : float, optional
        Maximum allowed separation from target in pixels.  No constraint
        applied if None.

    Returns
    -------
    target_position : tuple of float
        (x, y) centroid of the target star.
    reference_positions : list of tuple of float
        (x, y) centroids of the selected reference stars.
    reference_indices : list of int
        Indices of the selected reference stars in *sources*.

    Raises
    ------
    ValueError
        If *target_index* is out of range.

    Notes
    -----
    Selects the brightest available stars (excluding the target) as
    references, subject to the optional distance constraint.

    Examples
    --------
    >>> tgt, refs, ref_idx = select_reference_stars(
    ...     sources, target_index=1, n_references=3
    ... )
    """
    if target_index >= len(sources):
        raise ValueError(
            f"target_index {target_index} out of range (table has {len(sources)} rows)"
        )

    target_x = float(sources["x_centroid"][target_index])
    target_y = float(sources["y_centroid"][target_index])
    target_position = (target_x, target_y)

    candidate_indices = [i for i in range(len(sources)) if i != target_index]

    if max_separation is not None:
        candidate_indices = [
            idx for idx in candidate_indices
            if np.hypot(
                float(sources["x_centroid"][idx]) - target_x,
                float(sources["y_centroid"][idx]) - target_y,
            ) <= max_separation
        ]

    if len(candidate_indices) < n_references:
        warnings.warn(
            f"Only {len(candidate_indices)} reference stars available "
            f"(requested {n_references})"
        )
        n_references = len(candidate_indices)

    reference_indices = candidate_indices[:n_references]
    reference_positions = [
        (float(sources["x_centroid"][idx]), float(sources["y_centroid"][idx]))
        for idx in reference_indices
    ]

    print(f"✓ Target star #{target_index}: ({target_x:.1f}, {target_y:.1f})")
    print(f"✓ Selected {len(reference_positions)} reference stars:")
    for i, (idx, pos) in enumerate(zip(reference_indices, reference_positions)):
        print(
            f"    Ref {i + 1} (star #{idx}): ({pos[0]:.1f}, {pos[1]:.1f}), "
            f"flux={float(sources['flux'][idx]):.0f}"
        )

    return target_position, reference_positions, reference_indices


# ── Deprecation shim ───────────────────────────────────────────────────────────

def estimate_background(
    image: NDArray[np.float32],
    sample_size: int = 100,
    method: str = "corners",
):
    """
    .. deprecated::
        Moved to :mod:`pyTransitPhotometry.background`.
        This shim will be removed in v2.0.

    Parameters
    ----------
    image, sample_size, method
        See :func:`background.estimate_background`.
    """
    warnings.warn(
        "estimate_background has moved to pyTransitPhotometry.background. "
        "Update your import to: from pyTransitPhotometry.background import estimate_background",
        DeprecationWarning,
        stacklevel=2,
    )
    from .background import estimate_background as _impl
    return _impl(image, sample_size=sample_size, method=method)
