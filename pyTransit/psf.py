from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

def build_epsf(image: NDArray[np.float32], positions: List[Tuple[float, float]], size: int=25, oversampling: int=4, maxiters: int=10, sigma_clip_val: float=3.0) -> object:
    from astropy.nddata import NDData
    from astropy.stats import SigmaClip
    from astropy.table import Table
    from photutils.psf import EPSFBuilder, extract_stars
    if size % 2 == 0:
        size += 1
    nddata = NDData(data=image.astype(float))
    stars_tbl = Table()
    stars_tbl['x'] = [float(p[0]) for p in positions]
    stars_tbl['y'] = [float(p[1]) for p in positions]
    stars = extract_stars(nddata, stars_tbl, size=size)
    if len(stars) == 0:
        raise RuntimeError('No valid star cutouts extracted. Verify that positions fall within the image bounds.')
    sigma_clip = SigmaClip(sigma=sigma_clip_val)
    builder = EPSFBuilder(oversampling=oversampling, maxiters=maxiters, progress_bar=False, sigma_clip=sigma_clip)
    epsf, fitted_stars = builder(stars)
    print(f'✓ Built ePSF from {len(fitted_stars)} stars ({oversampling}× oversampling, {maxiters} iterations)')
    return epsf

def run_psf_photometry(image: NDArray[np.float32], positions: List[Tuple[float, float]], epsf: object, fwhm: float=5.0, fit_shape: int=11, background_2d: Optional[NDArray[np.float32]]=None, ccd_gain: float=1.0) -> List[dict]:
    try:
        from photutils.psf import PSFPhotometry
    except ImportError:
        try:
            from photutils.psf import BasicPSFPhotometry as PSFPhotometry
        except ImportError:
            raise ImportError("photutils >= 1.8 is required for PSF photometry. Install with: pip install 'photutils>=1.8'")
    from astropy.table import Table
    if fit_shape % 2 == 0:
        fit_shape += 1
    image_work = image - background_2d if background_2d is not None else image.copy()
    image_work = image_work.astype(float)
    init_params = Table()
    init_params['x'] = [float(p[0]) for p in positions]
    init_params['y'] = [float(p[1]) for p in positions]
    try:
        psfphot = PSFPhotometry(psf_model=epsf, fit_shape=(fit_shape, fit_shape), aperture_radius=max(3, int(fwhm * 1.5)))
        phot_table = psfphot(image_work, init_params=init_params)
    except Exception as exc:
        raise RuntimeError(f'PSF photometry failed: {exc}') from exc
    results: List[dict] = []
    for i, pos in enumerate(positions):
        if i < len(phot_table):
            row = phot_table[i]
            flux = np.nan
            for flux_col in ('flux_fit', 'flux', 'aperture_sum'):
                if flux_col in phot_table.colnames:
                    flux = float(row[flux_col])
                    break
            flux_err = np.nan
            for err_col in ('flux_err', 'flux_unc'):
                if err_col in phot_table.colnames:
                    flux_err = float(row[err_col])
                    break
            if not np.isfinite(flux_err):
                flux_err = float(np.sqrt(max(flux, 0) / ccd_gain + 1.0))
            x_fit = float(row['x_fit']) if 'x_fit' in phot_table.colnames else pos[0]
            y_fit = float(row['y_fit']) if 'y_fit' in phot_table.colnames else pos[1]
            results.append({'flux': flux, 'flux_err': flux_err, 'x_fit': x_fit, 'y_fit': y_fit})
        else:
            results.append({'flux': np.nan, 'flux_err': np.nan, 'x_fit': pos[0], 'y_fit': pos[1]})
    return results
