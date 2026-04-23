"""Star detection and reference-star selection using photutils."""

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
    Detect point sources using DAOStarFinder.

    Parameters
    ----------
    threshold_type : str
        ``'absolute'`` (raw counts) or ``'sigma'`` (requires *background_std*).
    sort_by : str
        ``'flux'`` (brightest first, default) or ``'sharpness'``.

    Raises
    ------
    ValueError
        If image is not 2-D or *threshold_type* is ``'sigma'`` without *background_std*.
    RuntimeError
        If no sources are detected.
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
    """Filter source table by sharpness, roundness, and minimum flux."""
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

    Selects the *n_references* brightest available stars (excluding target),
    optionally constrained to within *max_separation* pixels of the target.

    Returns
    -------
    target_position, reference_positions, reference_indices

    Raises
    ------
    ValueError
        If *target_index* is out of range.
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



def estimate_background(
    image: NDArray[np.float32],
    sample_size: int = 100,
    method: str = "corners",
):
    """Deprecated: use ``pyTransitPhotometry.background.estimate_background`` instead."""
    warnings.warn(
        "estimate_background has moved to pyTransitPhotometry.background. "
        "Update your import to: from pyTransitPhotometry.background import estimate_background",
        DeprecationWarning,
        stacklevel=2,
    )
    from .background import estimate_background as _impl
    return _impl(image, sample_size=sample_size, method=method)
