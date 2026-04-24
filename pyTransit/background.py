import warnings
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

def estimate_background(image: NDArray[np.float32], sample_size: int=100, method: str='corners') -> Tuple[float, float]:
    if image.ndim != 2:
        raise ValueError(f'Expected 2-D image, got shape {image.shape}')
    if method == 'corners':
        h, w = image.shape
        s = sample_size
        corners = [image[:s, :s], image[:s, -s:], image[-s:, :s], image[-s:, -s:]]
        combined = np.concatenate([c.ravel() for c in corners])
        return (float(np.mean(combined)), float(np.std(combined)))
    elif method == 'median':
        median = np.median(image)
        mad = np.median(np.abs(image - median))
        std = 1.4826 * mad
        return (float(median), float(std))
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'corners' or 'median'.")

def estimate_2d_background(image: NDArray[np.float32], box_size: int=64, filter_size: int=3, sigma_clip_val: float=3.0, method: str='background2d') -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    if method == 'background2d':
        from astropy.stats import SigmaClip
        from photutils.background import Background2D, MedianBackground
        sigma_clip = SigmaClip(sigma=sigma_clip_val, maxiters=5)
        bkg_estimator = MedianBackground()
        bkg = Background2D(image, box_size=(box_size, box_size), filter_size=(filter_size, filter_size), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        return (bkg.background.astype(np.float32), bkg.background_rms.astype(np.float32))
    elif method == 'polynomial':
        from astropy.modeling.fitting import LevMarLSQFitter
        from astropy.modeling.models import Polynomial2D
        from astropy.stats import sigma_clip as astropy_sigma_clip
        y_idx, x_idx = np.mgrid[:image.shape[0], :image.shape[1]]
        clipped = astropy_sigma_clip(image, sigma=sigma_clip_val, maxiters=5, masked=True)
        valid = ~clipped.mask
        step = max(1, min(image.shape) // 64)
        x_ds = x_idx[::step, ::step][valid[::step, ::step]].ravel()
        y_ds = y_idx[::step, ::step][valid[::step, ::step]].ravel()
        z_ds = image[::step, ::step][valid[::step, ::step]].ravel()
        poly_init = Polynomial2D(degree=3)
        fitter = LevMarLSQFitter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            poly_fit = fitter(poly_init, x_ds, y_ds, z_ds)
        background = poly_fit(x_idx, y_idx).astype(np.float32)
        residuals = image - background
        bkg_rms_val = float(np.std(residuals[valid]))
        background_rms = np.full_like(background, bkg_rms_val)
        print(f'✓ Polynomial 2D background: mean = {background.mean():.1f}, RMS = {bkg_rms_val:.2f}')
        return (background, background_rms)
    else:
        raise ValueError(f"Unknown background method '{method}'. Choose 'background2d' or 'polynomial'.")
