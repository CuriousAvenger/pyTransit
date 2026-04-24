import numpy as np
import warnings
from typing import Optional

def create_master_frame(frames: np.ndarray, method: str='median', sigma_clip: Optional[float]=None) -> np.ndarray:
    if frames.ndim != 3:
        raise ValueError(f'Expected 3D array, got shape {frames.shape}')
    if len(frames) < 3:
        warnings.warn(f'Only {len(frames)} frames available for combination. At least 5 frames recommended for robust statistics.')
    if sigma_clip is not None:
        median = np.median(frames, axis=0)
        std = np.std(frames, axis=0)
        mask = np.abs(frames - median) < sigma_clip * std
        if method == 'median':
            master = np.ma.median(np.ma.array(frames, mask=~mask), axis=0).data
        elif method == 'mean':
            master = np.ma.mean(np.ma.array(frames, mask=~mask), axis=0).data
        else:
            raise ValueError(f'Unknown method: {method}')
    elif method == 'median':
        master = np.median(frames, axis=0)
    elif method == 'mean':
        master = np.mean(frames, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'median' or 'mean'.")
    return master.astype(np.float32)

def scale_dark_frame(master_dark: np.ndarray, master_bias: np.ndarray, dark_exptime: float, target_exptime: float) -> np.ndarray:
    if dark_exptime <= 0 or target_exptime < 0:
        raise ValueError('Exposure times must be positive')
    dark_rate = (master_dark - master_bias) / dark_exptime
    scaled_dark = dark_rate * target_exptime
    return scaled_dark.astype(np.float32)

def create_normalized_flat(master_flat: np.ndarray, master_bias: np.ndarray, scaled_dark_flat: np.ndarray, method: str='mean') -> np.ndarray:
    flat_corrected = master_flat - master_bias - scaled_dark_flat
    if np.any(flat_corrected <= 0):
        warnings.warn('Flat field contains zero or negative values after bias/dark subtraction. Check your calibration frames.')
        flat_corrected = np.clip(flat_corrected, 1e-06, None)
    if method == 'mean':
        norm_value = np.mean(flat_corrected)
    elif method == 'median':
        norm_value = np.median(flat_corrected)
    else:
        raise ValueError(f'Unknown method: {method}')
    normalized_flat = flat_corrected / norm_value
    if not np.isfinite(normalized_flat).all():
        raise ValueError('Normalized flat contains NaN or Inf values')
    return normalized_flat.astype(np.float32)

def calibrate_image(raw_image: np.ndarray, master_bias: np.ndarray, scaled_dark: np.ndarray, normalized_flat: np.ndarray) -> np.ndarray:
    shapes = [arr.shape for arr in [raw_image, master_bias, scaled_dark, normalized_flat]]
    if len(set(shapes)) > 1:
        raise ValueError(f'Shape mismatch: {shapes}')
    calibrated = (raw_image - master_bias - scaled_dark) / normalized_flat
    return calibrated.astype(np.float32)

class CalibrationFrames:

    def __init__(self, master_bias: np.ndarray, master_dark: np.ndarray, master_flat: np.ndarray, dark_exptime: float, flat_exptime: float):
        self.master_bias = master_bias
        self.master_dark = master_dark
        self.dark_exptime = dark_exptime
        self.flat_exptime = flat_exptime
        self.scaled_dark_flat = scale_dark_frame(master_dark, master_bias, dark_exptime, flat_exptime)
        self.normalized_flat = create_normalized_flat(master_flat, master_bias, self.scaled_dark_flat)
        print('✓ Calibration frames prepared')
        print(f'  Bias level: {np.median(master_bias):.1f} counts')
        print(f'  Dark current: {np.median(master_dark - master_bias):.1f} counts/{dark_exptime}s')
        print(f'  Flat field range: {np.min(self.normalized_flat):.3f} - {np.max(self.normalized_flat):.3f}')

    def calibrate(self, raw_image: np.ndarray, exptime: float) -> np.ndarray:
        scaled_dark = scale_dark_frame(self.master_dark, self.master_bias, self.dark_exptime, exptime)
        return calibrate_image(raw_image, self.master_bias, scaled_dark, self.normalized_flat)

    def calibrate_batch(self, raw_images: np.ndarray, exptimes: np.ndarray) -> np.ndarray:
        if len(raw_images) != len(exptimes):
            raise ValueError('Number of images must match number of exposure times')
        calibrated = []
        for img, exptime in zip(raw_images, exptimes):
            calibrated.append(self.calibrate(img, exptime))
        return np.array(calibrated)
