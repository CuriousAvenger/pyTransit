"""
CCD calibration routines for bias, dark, and flat field correction.

Implements standard CCD reduction:
    calibrated = (raw - bias - dark*scale) / flat_normalized

Includes exposure time scaling and validation checks.
"""

import numpy as np
import warnings
from typing import Tuple, Optional


def create_master_frame(
    frames: np.ndarray,
    method: str = "median",
    sigma_clip: Optional[float] = None
) -> np.ndarray:
    """
    Combine multiple calibration frames into a master frame.
    
    Parameters
    ----------
    frames : np.ndarray
        3D array of shape (n_frames, height, width)
    method : str, optional
        Combination method: 'median' (default) or 'mean'
    sigma_clip : float, optional
        Sigma threshold for outlier rejection (e.g., 3.0)
        If None, no clipping is performed
    
    Returns
    -------
    master : np.ndarray
        2D master calibration frame
    
    Notes
    -----
    Median is preferred for cosmic ray rejection, but mean has better
    SNR for flat fields if cosmic rays are already removed.
    
    Examples
    --------
    >>> master_bias = create_master_frame(bias_frames, method='median')
    >>> master_flat = create_master_frame(flat_frames, method='median', sigma_clip=3.0)
    """
    if frames.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {frames.shape}")
    
    if len(frames) < 3:
        warnings.warn(
            f"Only {len(frames)} frames available for combination. "
            "At least 5 frames recommended for robust statistics."
        )
    
    if sigma_clip is not None:
        # Sigma clipping
        median = np.median(frames, axis=0)
        std = np.std(frames, axis=0)
        
        # Create mask for valid pixels
        mask = np.abs(frames - median) < sigma_clip * std
        
        # Combine with mask
        if method == "median":
            master = np.ma.median(np.ma.array(frames, mask=~mask), axis=0).data
        elif method == "mean":
            master = np.ma.mean(np.ma.array(frames, mask=~mask), axis=0).data
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        # Simple combination
        if method == "median":
            master = np.median(frames, axis=0)
        elif method == "mean":
            master = np.mean(frames, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'median' or 'mean'.")
    
    return master.astype(np.float32)


def scale_dark_frame(
    master_dark: np.ndarray,
    master_bias: np.ndarray,
    dark_exptime: float,
    target_exptime: float
) -> np.ndarray:
    """
    Scale dark current to match target exposure time.
    
    Dark current accumulates linearly with time, so:
        dark(t) = bias + dark_rate * t
    
    Parameters
    ----------
    master_dark : np.ndarray
        Master dark frame
    master_bias : np.ndarray
        Master bias frame
    dark_exptime : float
        Exposure time of master dark (seconds)
    target_exptime : float
        Desired exposure time (seconds)
    
    Returns
    -------
    scaled_dark : np.ndarray
        Bias-subtracted dark scaled to target exposure time
    
    Notes
    -----
    Returns (dark - bias) * (target_exptime / dark_exptime)
    
    For science frames:
        scaled_dark_sci = scale_dark_frame(dark, bias, t_dark, t_sci)
    For flat frames:
        scaled_dark_flat = scale_dark_frame(dark, bias, t_dark, t_flat)
    """
    if dark_exptime <= 0 or target_exptime < 0:
        raise ValueError("Exposure times must be positive")
    
    # Remove bias and scale dark current
    dark_rate = (master_dark - master_bias) / dark_exptime
    scaled_dark = dark_rate * target_exptime
    
    return scaled_dark.astype(np.float32)


def create_normalized_flat(
    master_flat: np.ndarray,
    master_bias: np.ndarray,
    scaled_dark_flat: np.ndarray,
    method: str = "mean"
) -> np.ndarray:
    """
    Create normalized flat field.
    
    Removes bias and dark, then normalizes to unity mean.
    
    Parameters
    ----------
    master_flat : np.ndarray
        Master flat frame
    master_bias : np.ndarray
        Master bias frame
    scaled_dark_flat : np.ndarray
        Dark current scaled to flat exposure time
    method : str, optional
        Normalization method: 'mean' (default) or 'median'
    
    Returns
    -------
    normalized_flat : np.ndarray
        Flat field normalized to have mean=1
    
    Raises
    ------
    ValueError
        If flat field has zero or negative values after correction
    """
    # Subtract bias and dark
    flat_corrected = master_flat - master_bias - scaled_dark_flat
    
    # Check for invalid values
    if np.any(flat_corrected <= 0):
        warnings.warn(
            "Flat field contains zero or negative values after bias/dark subtraction. "
            "Check your calibration frames."
        )
        # Clip to small positive value to avoid division issues
        flat_corrected = np.clip(flat_corrected, 1e-6, None)
    
    # Normalize
    if method == "mean":
        norm_value = np.mean(flat_corrected)
    elif method == "median":
        norm_value = np.median(flat_corrected)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    normalized_flat = flat_corrected / norm_value
    
    # Validate normalization
    if not np.isfinite(normalized_flat).all():
        raise ValueError("Normalized flat contains NaN or Inf values")
    
    return normalized_flat.astype(np.float32)


def calibrate_image(
    raw_image: np.ndarray,
    master_bias: np.ndarray,
    scaled_dark: np.ndarray,
    normalized_flat: np.ndarray
) -> np.ndarray:
    """
    Apply full CCD calibration to a raw science image.
    
    Performs: (raw - bias - dark) / flat
    
    Parameters
    ----------
    raw_image : np.ndarray
        Raw science frame
    master_bias : np.ndarray
        Master bias frame
    scaled_dark : np.ndarray
        Dark frame scaled to science exposure time
    normalized_flat : np.ndarray
        Normalized flat field (mean=1)
    
    Returns
    -------
    calibrated : np.ndarray
        Calibrated science frame
    
    Notes
    -----
    This is the standard CCD reduction equation. The flat field
    corrects for pixel-to-pixel sensitivity variations.
    """
    # Verify shapes match
    shapes = [arr.shape for arr in [raw_image, master_bias, scaled_dark, normalized_flat]]
    if len(set(shapes)) > 1:
        raise ValueError(f"Shape mismatch: {shapes}")
    
    # Apply calibration
    calibrated = (raw_image - master_bias - scaled_dark) / normalized_flat
    
    return calibrated.astype(np.float32)


class CalibrationFrames:
    """
    Container for calibration frames with validation.
    
    Parameters
    ----------
    master_bias : np.ndarray
        Master bias frame
    master_dark : np.ndarray
        Master dark frame
    master_flat : np.ndarray
        Master flat frame (unnormalized)
    dark_exptime : float
        Exposure time of dark frames (seconds)
    flat_exptime : float
        Exposure time of flat frames (seconds)
    
    Examples
    --------
    >>> calib = CalibrationFrames(
    ...     master_bias, master_dark, master_flat,
    ...     dark_exptime=85.0, flat_exptime=1.0
    ... )
    >>> calibrated = calib.calibrate(raw_image, exptime=85.0)
    """
    
    def __init__(
        self,
        master_bias: np.ndarray,
        master_dark: np.ndarray,
        master_flat: np.ndarray,
        dark_exptime: float,
        flat_exptime: float
    ):
        self.master_bias = master_bias
        self.master_dark = master_dark
        self.dark_exptime = dark_exptime
        self.flat_exptime = flat_exptime
        
        # Pre-compute scaled dark for flat and normalized flat
        self.scaled_dark_flat = scale_dark_frame(
            master_dark, master_bias, dark_exptime, flat_exptime
        )
        self.normalized_flat = create_normalized_flat(
            master_flat, master_bias, self.scaled_dark_flat
        )
        
        print("✓ Calibration frames prepared")
        print(f"  Bias level: {np.median(master_bias):.1f} counts")
        print(f"  Dark current: {np.median(master_dark - master_bias):.1f} counts/{dark_exptime}s")
        print(f"  Flat field range: {np.min(self.normalized_flat):.3f} - {np.max(self.normalized_flat):.3f}")
    
    def calibrate(self, raw_image: np.ndarray, exptime: float) -> np.ndarray:
        """
        Calibrate a science image.
        
        Parameters
        ----------
        raw_image : np.ndarray
            Raw science frame
        exptime : float
            Exposure time of science frame (seconds)
        
        Returns
        -------
        calibrated : np.ndarray
            Calibrated image
        """
        scaled_dark = scale_dark_frame(
            self.master_dark, self.master_bias, self.dark_exptime, exptime
        )
        
        return calibrate_image(
            raw_image, self.master_bias, scaled_dark, self.normalized_flat
        )
    
    def calibrate_batch(
        self,
        raw_images: np.ndarray,
        exptimes: np.ndarray
    ) -> np.ndarray:
        """
        Calibrate multiple images with potentially different exposure times.
        
        Parameters
        ----------
        raw_images : np.ndarray
            3D array of raw images
        exptimes : np.ndarray
            Array of exposure times
        
        Returns
        -------
        calibrated_images : np.ndarray
            3D array of calibrated images
        """
        if len(raw_images) != len(exptimes):
            raise ValueError("Number of images must match number of exposure times")
        
        calibrated = []
        for img, exptime in zip(raw_images, exptimes):
            calibrated.append(self.calibrate(img, exptime))
        
        return np.array(calibrated)
