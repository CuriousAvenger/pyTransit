"""
Star detection routines using photutils.

Wraps DAOStarFinder for consistent, configurable source detection.
"""

import numpy as np
from photutils.detection import DAOStarFinder
from astropy.table import Table
from typing import Optional
import warnings


def detect_sources(
    image: np.ndarray,
    fwhm: float = 5.0,
    threshold: float = 10.0,
    threshold_type: str = "absolute",
    background_std: Optional[float] = None,
    exclude_border: bool = True,
    sort_by: str = "flux"
) -> Table:
    """
    Detect point sources in an image using DAOStarFinder.
    
    Parameters
    ----------
    image : np.ndarray
        2D image array
    fwhm : float, optional
        Full-width at half-maximum of stellar PSF in pixels (default: 5.0)
    threshold : float, optional
        Detection threshold (default: 10.0)
    threshold_type : str, optional
        'absolute' for counts, 'sigma' for SNR (default: 'absolute')
    background_std : float, optional
        Background standard deviation for sigma-based threshold
        Required if threshold_type='sigma'
    exclude_border : bool, optional
        Exclude sources near image borders (default: True)
    sort_by : str, optional
        Sort sources by 'flux' (default) or 'sharpness'
    
    Returns
    -------
    sources : astropy.table.Table
        Detected sources with columns:
        - id, xcentroid, ycentroid, flux, peak, sharpness, roundness, npix
    
    Raises
    ------
    ValueError
        If threshold_type='sigma' but background_std not provided
    RuntimeError
        If no sources detected
    
    Notes
    -----
    DAOStarFinder algorithm:
    1. Convolves image with Gaussian kernel (FWHM)
    2. Finds local maxima above threshold
    3. Fits 1D Gaussian profiles in x/y to refine centroids
    4. Computes sharpness and roundness for quality filtering
    
    Examples
    --------
    >>> sources = detect_sources(image, fwhm=5.0, threshold=10000.0)
    >>> print(f"Found {len(sources)} sources")
    >>> target_x, target_y = sources['xcentroid'][0], sources['ycentroid'][0]
    """
    # Validate inputs
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    
    if threshold_type == "sigma":
        if background_std is None:
            raise ValueError("background_std required for sigma-based threshold")
        actual_threshold = threshold * background_std
    elif threshold_type == "absolute":
        actual_threshold = threshold
    else:
        raise ValueError(f"Unknown threshold_type: {threshold_type}")
    
    # Initialize finder
    daofind = DAOStarFinder(
        fwhm=fwhm,
        threshold=actual_threshold,
        exclude_border=exclude_border
    )
    
    # Detect sources
    sources = daofind(image)
    
    if sources is None or len(sources) == 0:
        raise RuntimeError(
            f"No sources detected with threshold={actual_threshold:.1f}, fwhm={fwhm}. "
            "Try lowering the threshold or adjusting FWHM."
        )
    
    # Sort sources
    if sort_by == "flux":
        sources.sort("flux", reverse=True)
    elif sort_by == "sharpness":
        sources.sort("sharpness")
    else:
        warnings.warn(f"Unknown sort_by: {sort_by}, using flux")
        sources.sort("flux", reverse=True)
    
    # Format output
    for col in sources.colnames:
        if col not in ('id', 'npix'):
            sources[col].info.format = '%.2f'
    
    print(f"✓ Detected {len(sources)} sources")
    print(f"  Brightest source: flux={sources['flux'][0]:.0f}, "
          f"position=({sources['xcentroid'][0]:.1f}, {sources['ycentroid'][0]:.1f})")
    
    return sources


def filter_sources(
    sources: Table,
    min_sharpness: float = 0.3,
    max_sharpness: float = 1.0,
    max_roundness: float = 0.5,
    min_flux: Optional[float] = None
) -> Table:
    """
    Filter sources by quality metrics.
    
    Parameters
    ----------
    sources : Table
        Source table from detect_sources()
    min_sharpness : float, optional
        Minimum sharpness (default: 0.3)
        Low values indicate extended/blended sources
    max_sharpness : float, optional
        Maximum sharpness (default: 1.0)
        High values indicate cosmic rays
    max_roundness : float, optional
        Maximum absolute roundness (default: 0.5)
        Measures elongation; 0 = perfectly round
    min_flux : float, optional
        Minimum flux threshold
    
    Returns
    -------
    filtered : Table
        Filtered source table
    
    Notes
    -----
    Sharpness: ratio of central pixel to surrounding pixels
    Roundness: (2σ_x - 2σ_y) / (2σ_x + 2σ_y) where σ are Gaussian widths
    
    Good stellar sources typically have:
    - sharpness: 0.4 - 0.8
    - |roundness|: < 0.3
    """
    mask = np.ones(len(sources), dtype=bool)
    
    if 'sharpness' in sources.colnames:
        mask &= (sources['sharpness'] >= min_sharpness)
        mask &= (sources['sharpness'] <= max_sharpness)
    
    if 'roundness' in sources.colnames:
        mask &= np.abs(sources['roundness']) <= max_roundness
    
    if min_flux is not None and 'flux' in sources.colnames:
        mask &= sources['flux'] >= min_flux
    
    filtered = sources[mask]
    
    n_removed = len(sources) - len(filtered)
    if n_removed > 0:
        print(f"✓ Filtered out {n_removed} sources (quality cuts)")
        print(f"  Remaining: {len(filtered)} sources")
    
    return filtered


def select_reference_stars(
    sources: Table,
    target_index: int,
    n_references: int = 3,
    max_separation: Optional[float] = None
) -> tuple:
    """
    Select reference stars for differential photometry.
    
    Parameters
    ----------
    sources : Table
        Source table sorted by brightness
    target_index : int
        Index of target star in sources table
    n_references : int, optional
        Number of reference stars to select (default: 3)
    max_separation : float, optional
        Maximum allowed separation from target in pixels
        If None, no distance constraint
    
    Returns
    -------
    target_position : tuple
        (x, y) position of target star
    reference_positions : list of tuples
        List of (x, y) positions for reference stars
    reference_indices : list of int
        Indices of reference stars in sources table
    
    Notes
    -----
    Selects the brightest stars (excluding target) as references.
    Ideal reference stars are:
    - Bright (high SNR)
    - Non-variable
    - Nearby (minimize differential atmospheric effects)
    - Not saturated
    
    Examples
    --------
    >>> target_pos, ref_positions, ref_indices = select_reference_stars(
    ...     sources, target_index=2, n_references=2
    ... )
    """
    if target_index >= len(sources):
        raise ValueError(f"target_index {target_index} out of range (max: {len(sources)-1})")
    
    target_x = sources['xcentroid'][target_index]
    target_y = sources['ycentroid'][target_index]
    target_position = (target_x, target_y)
    
    # Get all potential references (excluding target)
    reference_indices = [i for i in range(len(sources)) if i != target_index]
    
    # Apply distance constraint if specified
    if max_separation is not None:
        filtered_refs = []
        for idx in reference_indices:
            dx = sources['xcentroid'][idx] - target_x
            dy = sources['ycentroid'][idx] - target_y
            separation = np.sqrt(dx**2 + dy**2)
            if separation <= max_separation:
                filtered_refs.append(idx)
        reference_indices = filtered_refs
    
    # Select top n_references by brightness
    if len(reference_indices) < n_references:
        warnings.warn(
            f"Only {len(reference_indices)} reference stars available "
            f"(requested {n_references})"
        )
        n_references = len(reference_indices)
    
    reference_indices = reference_indices[:n_references]
    reference_positions = [
        (sources['xcentroid'][idx], sources['ycentroid'][idx])
        for idx in reference_indices
    ]
    
    print(f"✓ Selected target star #{target_index}: position=({target_x:.1f}, {target_y:.1f})")
    print(f"✓ Selected {len(reference_positions)} reference stars:")
    for i, (idx, pos) in enumerate(zip(reference_indices, reference_positions)):
        print(f"    Ref {i+1} (star #{idx}): position=({pos[0]:.1f}, {pos[1]:.1f}), "
              f"flux={sources['flux'][idx]:.0f}")
    
    return target_position, reference_positions, reference_indices


def estimate_background(
    image: np.ndarray,
    sample_size: int = 100,
    method: str = "corners"
) -> tuple:
    """
    Estimate background level and noise from image.
    
    Parameters
    ----------
    image : np.ndarray
        2D image
    sample_size : int, optional
        Size of sample region in pixels (default: 100)
    method : str, optional
        'corners' (default) samples four corners
        'median' uses global median absolute deviation
    
    Returns
    -------
    background_mean : float
        Mean background level
    background_std : float
        Background standard deviation
    
    Examples
    --------
    >>> bg_mean, bg_std = estimate_background(image, sample_size=100)
    >>> sources = detect_sources(image, threshold=5, threshold_type='sigma',
    ...                          background_std=bg_std)
    """
    if method == "corners":
        # Sample four corners
        h, w = image.shape
        corners = [
            image[:sample_size, :sample_size],
            image[:sample_size, -sample_size:],
            image[-sample_size:, :sample_size],
            image[-sample_size:, -sample_size:]
        ]
        combined = np.concatenate([c.flatten() for c in corners])
        return np.mean(combined), np.std(combined)
    
    elif method == "median":
        # Use median absolute deviation (robust to outliers)
        median = np.median(image)
        mad = np.median(np.abs(image - median))
        # Convert MAD to std (for normal distribution)
        std = 1.4826 * mad
        return float(median), float(std)
    
    else:
        raise ValueError(f"Unknown method: {method}")
