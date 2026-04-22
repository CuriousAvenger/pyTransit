"""
I/O utilities for FITS file loading and data management.

Handles:
- FITS file discovery and loading
- Header extraction and validation
- Results export (CSV, JSON)
"""

import numpy as np
from glob import glob
import os
from astropy.io import fits
from typing import Tuple, List, Dict, Any
import warnings


def load_fits_files(
    directory: str, pattern: str = "*.fits", dtype: np.dtype = np.float32, verbose: bool = True
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Load FITS files from a directory.

    Parameters
    ----------
    directory : str
        Path to directory containing FITS files
    pattern : str, optional
        Glob pattern for file matching (default: "*.fits")
    dtype : np.dtype, optional
        Data type for image arrays (default: float32 for memory efficiency)
    verbose : bool, optional
        Print loading progress (default: True)

    Returns
    -------
    data : np.ndarray
        3D array of shape (n_frames, height, width)
    headers : list of dict
        FITS headers as dictionaries

    Raises
    ------
    FileNotFoundError
        If directory doesn't exist or no files match pattern
    ValueError
        If FITS files have inconsistent dimensions

    Notes
    -----
    Files are sorted alphabetically to ensure consistent ordering.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(glob(os.path.join(directory, pattern)))

    if not files:
        raise FileNotFoundError(f"No files found in {directory} with pattern '{pattern}'")

    data_list = []
    headers = []

    for i, filepath in enumerate(files):
        try:
            with fits.open(filepath) as hdul:
                # Get primary HDU data
                img_data = hdul[0].data
                header = dict(hdul[0].header)

                if img_data is None:
                    warnings.warn(f"No data in {os.path.basename(filepath)}, skipping")
                    continue

                data_list.append(img_data.astype(dtype))
                headers.append(header)

                if verbose and (i + 1) % 10 == 0:
                    print(f"Loaded {i + 1}/{len(files)} files...")

        except Exception as e:
            warnings.warn(f"Error loading {filepath}: {e}")
            continue

    if not data_list:
        raise ValueError(f"No valid FITS data loaded from {directory}")

    # Verify consistent dimensions
    shapes = [arr.shape for arr in data_list]
    if len(set(shapes)) > 1:
        raise ValueError(
            f"Inconsistent image dimensions found: {set(shapes)}\n"
            "All FITS files must have the same dimensions."
        )

    data = np.array(data_list)

    if verbose:
        print(f"✓ Loaded {len(data)} frames of shape {data[0].shape}")

    return data, headers


def extract_header_value(
    headers: List[Dict[str, Any]], key: str, default: Any = None, fallback_keys: List[str] = None
) -> np.ndarray:
    """
    Extract a header value from multiple FITS headers.

    Parameters
    ----------
    headers : list of dict
        FITS headers
    key : str
        Primary header key to extract
    default : any, optional
        Default value if key not found
    fallback_keys : list of str, optional
        Alternative keys to try if primary key not found

    Returns
    -------
    values : np.ndarray
        Array of extracted values

    Examples
    --------
    >>> times = extract_header_value(headers, 'JD-HELIO', default=0.0)
    >>> airmass = extract_header_value(
    ...     headers, 'AIRMASS',
    ...     fallback_keys=['SECZ', 'AIRMASS-START']
    ... )
    """
    values = []
    all_keys = [key] + (fallback_keys or [])

    for header in headers:
        value = None
        for k in all_keys:
            if k in header:
                value = header[k]
                break

        if value is None:
            value = default

        values.append(value)

    return np.array(values)


def get_ccd_gain(header: Dict[str, Any]) -> float:
    """
    Extract CCD gain from FITS header.

    Tries common gain keywords: GAIN, EGAIN, CCGAIN, CAMGAIN, CCDGAIN.

    Parameters
    ----------
    header : dict
        FITS header

    Returns
    -------
    gain : float
        CCD gain in e-/ADU (defaults to 1.0 if not found)
    """
    gain_keys = ["GAIN", "EGAIN", "CCGAIN", "CAMGAIN", "CCDGAIN"]

    for key in gain_keys:
        if key in header:
            return float(header[key])

    warnings.warn(
        "CCD gain not found in header. Using default gain=1.0 e-/ADU. "
        "This may affect photometric error estimates."
    )
    return 1.0


def export_lightcurve(
    output_path: str, times: np.ndarray, fluxes: np.ndarray, errors: np.ndarray, **metadata
):
    """
    Export light curve to CSV file.

    Parameters
    ----------
    output_path : str
        Path to output CSV file
    times : np.ndarray
        Observation times (MJD or BJD)
    fluxes : np.ndarray
        Relative flux values
    errors : np.ndarray
        Flux uncertainties
    **metadata : dict
        Additional columns to include (e.g., airmass, centroids)

    Examples
    --------
    >>> export_lightcurve(
    ...     'wasp75b_lightcurve.csv',
    ...     times, fluxes, errors,
    ...     airmass=airmass_values,
    ...     x_centroid=x_positions
    ... )
    """
    import pandas as pd

    data = {"time_mjd": times, "flux_ratio": fluxes, "flux_error": errors}

    # Add any extra columns
    data.update(metadata)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format="%.8f")
    print(f"✓ Light curve saved to {output_path}")


def export_fit_results(
    output_path: str, parameters: Dict[str, Tuple[float, float]], metadata: Dict[str, Any] = None
):
    """
    Export transit fit results to JSON.

    Parameters
    ----------
    output_path : str
        Path to output JSON file
    parameters : dict
        Dictionary of {param_name: (value, uncertainty)}
    metadata : dict, optional
        Additional information (e.g., target name, filters)

    Examples
    --------
    >>> export_fit_results(
    ...     'fit_results.json',
    ...     {'rp_rs': (0.103, 0.002), 'a_rs': (7.2, 0.3)},
    ...     metadata={'target': 'WASP-75b', 'filter': 'V'}
    ... )
    """
    import json

    output = {
        "parameters": {
            k: {"value": float(v[0]), "uncertainty": float(v[1])} for k, v in parameters.items()
        }
    }

    if metadata:
        output["metadata"] = metadata

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✓ Fit results saved to {output_path}")
