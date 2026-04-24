# Development Guide

**Extensibility, Testing, and Contribution Guidelines for pyTransit**

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Code Organization](#code-organization)
3. [Extension Guide](#extension-guide)
4. [Testing Strategy](#testing-strategy)
5. [Known Limitations](#known-limitations)
6. [Technical Debt](#technical-debt)
7. [Future Roadmap](#future-roadmap)
8. [Contribution Guidelines](#contribution-guidelines)

---

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtualenv or conda

### Clone and Install

```bash
# Clone repository
git clone https://github.com/CuriousAvenger/pyTransit.git
cd pytransitphotometry

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Development Dependencies

Installed via `pip install -e ".[dev]"`:
- **pytest** (≥7.0): Testing framework
- **pytest-cov** (≥3.0): Coverage reporting
- **black** (≥22.0): Code formatting
- **flake8** (≥4.0): Linting
- **jupyter** (≥1.0): Interactive development

### Project Structure

```
pytransitphotometry/
├── pyTransit/       # Main package
│   ├── __init__.py
│   ├── pipeline.py
│   ├── config.py
│   ├── calibration.py
│   ├── detection.py
│   ├── photometry.py
│   ├── lightcurve.py
│   ├── detrending.py
│   ├── models.py
│   ├── io.py
│   ├── visualization.py
│   └── cli.py
│
├── tests/                      # Test suite (TO BE CREATED)
│   ├── __init__.py
│   ├── test_calibration.py
│   ├── test_detection.py
│   ├── test_photometry.py
│   └── ...
│
├── docs/                       # Documentation
│   ├── DOCUMENTATION.md
│   ├── architecture.md
│   ├── api.md
│   ├── usage.md
│   └── DEVELOPMENT.md (this file)
│
├── examples/                   # Example scripts (TO BE CREATED)
│   ├── basic_usage.py
│   ├── custom_detrending.py
│   └── batch_processing.py
│
├── setup.py                    # Package configuration
├── requirements.txt            # Dependencies
├── config_example.yaml         # Template configuration
├── README.md                   # Project overview
└── LICENSE                     # MIT License
```

---

## Code Organization

### Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Open/Closed**: Open for extension, closed for modification
3. **Dependency Inversion**: Depend on abstractions, not concrete implementations
4. **Explicit Over Implicit**: No magic, all behavior visible in code/config

### Module Dependencies

**Rule:** Modules at lower levels should not import from higher levels.

```
cli.py  ──────────────────────┐
                              ▼
pipeline.py  ────────────────────────┐
                                     ▼
models.py, lightcurve.py, detrending.py
                                     ▼
photometry.py, detection.py, calibration.py
                                     ▼
io.py, config.py, visualization.py
                                     ▼
numpy, scipy, astropy, photutils, batman
```

**Implication:** You can replace/extend high-level modules without touching low-level algorithms.

### Naming Conventions

- **Functions**: `snake_case` (e.g., `measure_flux`)
- **Classes**: `PascalCase` (e.g., `TransitPipeline`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `BATMAN_AVAILABLE`)
- **Private methods**: Prefix with `_` (e.g., `_validate_shapes`)
- **Config dataclasses**: Suffix with `Config` (e.g., `PhotometryConfig`)

### Docstring Style

**NumPy style** for consistency with scientific Python ecosystem.

```python
def measure_flux(
    image: np.ndarray,
    position: Tuple[float, float],
    aperture_radius: float,
    annulus_inner: float,
    annulus_outer: float,
    ccd_gain: float = 1.0
) -> dict:
    """
    Measure background-subtracted flux with uncertainties.
    
    Parameters
    ----------
    image : np.ndarray
        2D image array
    position : tuple of float
        Star centroid (x, y)
    aperture_radius : float
        Photometry aperture radius in pixels
    annulus_inner : float
        Background annulus inner radius
    annulus_outer : float
        Background annulus outer radius
    ccd_gain : float, optional
        CCD gain in e-/ADU (default: 1.0)
    
    Returns
    -------
    result : dict
        Dictionary with keys: 'flux', 'flux_err', 'snr', 'background_mean',
        'background_std', 'aperture_sum', 'centroid'
    
    Notes
    -----
    Uses photutils aperture photometry with local background estimation.
    Centroid is refined via 2D Gaussian fit before photometry.
    
    Examples
    --------
    >>> result = measure_flux(image, (512.3, 768.9), 8.0, 40.0, 60.0, ccd_gain=1.5)
    >>> print(f"Flux: {result['flux']:.1f} ± {result['flux_err']:.1f}")
    """
    # Implementation...
```

---

## Extension Guide

### Adding a New Detrending Method

**Scenario:** You want to add Gaussian Process detrending.

#### Step 1: Create New Function in `detrending.py`

```python
def gp_detrend(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    kernel_scale: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detrend using Gaussian Process regression.
    
    Parameters
    ----------
    times : np.ndarray
        Observation times
    fluxes : np.ndarray
        Flux measurements
    errors : np.ndarray
        Flux uncertainties
    kernel_scale : float, optional
        GP kernel length scale (default: 0.1 days)
    
    Returns
    -------
    detrended_fluxes : np.ndarray
        Fluxes with GP trend removed
    gp_trend : np.ndarray
        Fitted GP model
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    
    # Define kernel
    kernel = RBF(length_scale=kernel_scale) + WhiteKernel()
    
    # Fit GP
    gp = GaussianProcessRegressor(kernel=kernel, alpha=errors**2)
    gp.fit(times.reshape(-1, 1), fluxes)
    
    # Predict trend
    gp_trend = gp.predict(times.reshape(-1, 1))
    
    # Detrend
    detrended_fluxes = fluxes - gp_trend + np.median(fluxes)
    
    return detrended_fluxes, gp_trend
```

#### Step 2: Add Configuration Option

In `config.py`:

```python
@dataclass
class DetrendingConfig:
    # Existing options...
    sigma_threshold: float = 3.0
    max_iterations: int = 5
    remove_linear_trend: bool = True
    test_airmass: bool = True
    
    # New option
    use_gp_detrending: bool = False
    gp_kernel_scale: float = 0.1
```

#### Step 3: Integrate into Pipeline

In `pipeline.py`:

```python
def run_detrending(self):
    """Stage 4: Remove outliers and systematic trends."""
    from .detrending import detrend_lightcurve, gp_detrend
    
    # Existing detrending
    self.detrended_lc = detrend_lightcurve(
        self.lightcurve,
        self.config.detrending,
        airmass=extract_header_value(self.headers, 'AIRMASS', default=None)
    )
    
    # Optional GP detrending
    if self.config.detrending.use_gp_detrending:
        print("Applying Gaussian Process detrending...")
        detrended_fluxes, gp_trend = gp_detrend(
            self.detrended_lc['times'],
            self.detrended_lc['fluxes'],
            self.detrended_lc['errors'],
            kernel_scale=self.config.detrending.gp_kernel_scale
        )
        self.detrended_lc['fluxes'] = detrended_fluxes
        self.detrended_lc['gp_trend'] = gp_trend
```

#### Step 4: Update Config Example

In `config_example.yaml`:

```yaml
detrending:
  sigma_threshold: 3.0
  max_iterations: 5
  remove_linear_trend: true
  test_airmass: true
  
  # Gaussian Process detrending (advanced)
  use_gp_detrending: false
  gp_kernel_scale: 0.1  # Length scale in days
```

#### Step 5: Document New Feature

Update `usage.md` and `api.md`.

---

### Adding a New Photometry Backend

**Scenario:** Replace photutils with SExtractor for crowded fields.

#### Step 1: Create Adapter Function

In `photometry.py`:

```python
def measure_flux_sextractor(
    image: np.ndarray,
    position: Tuple[float, float],
    aperture_radius: float,
    annulus_inner: float,
    annulus_outer: float,
    **kwargs
) -> dict:
    """
    Measure flux using SExtractor.
    
    Compatible interface with measure_flux().
    """
    # Save image to temp file
    import tempfile
    import subprocess
    from astropy.io import fits
    
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
        fits.writeto(f.name, image, overwrite=True)
        image_path = f.name
    
    # Run SExtractor
    sex_config = f"""
    DETECT_THRESH 5.0
    PHOT_APERTURES {aperture_radius*2}
    """
    
    # ... (SExtractor execution and parsing)
    
    # Return compatible dict
    return {
        'flux': flux_value,
        'flux_err': flux_err,
        'snr': snr,
        'background_mean': bkg_mean,
        'background_std': bkg_std,
        'aperture_sum': aper_sum,
        'centroid': (x_refined, y_refined)
    }
```

#### Step 2: Add Configuration Switch

```python
@dataclass
class PhotometryConfig:
    # ... existing fields
    
    backend: str = "photutils"  # "photutils" or "sextractor"
    sextractor_path: Optional[str] = None
```

#### Step 3: Use in Pipeline

```python
def run_photometry(self):
    from .photometry import measure_flux, measure_flux_sextractor
    
    # Select backend
    if self.config.photometry.backend == "sextractor":
        phot_func = measure_flux_sextractor
    else:
        phot_func = measure_flux
    
    # Rest of photometry code uses phot_func
```

**Key insight:** Strategy pattern allows swapping implementations without changing algorithm logic.

---

### Adding MCMC Fitting

**Scenario:** Replace least-squares with MCMC for uncertainty estimation.

#### Step 1: Create MCMC Fitter

In `models.py`:

```python
class MCMCTransitFitter:
    """
    MCMC transit fitter using emcee.
    
    Compatible interface with TransitFitter.
    """
    
    def __init__(self, period, t0_guess, limb_dark_u1=0.4, limb_dark_u2=0.26):
        self.period = period
        self.t0_guess = t0_guess
        self.limb_dark_u1 = limb_dark_u1
        self.limb_dark_u2 = limb_dark_u2
    
    def fit(self, times, fluxes, errors, nwalkers=50, nsteps=5000):
        """
        Fit using MCMC.
        
        Returns same dict structure as TransitFitter.fit().
        """
        import emcee
        
        def log_likelihood(params, times, fluxes, errors):
            rp, a, inc, baseline, slope = params
            model = self.model_with_detrending(times, rp, a, inc, baseline, slope)
            return -0.5 * np.sum(((fluxes - model) / errors)**2)
        
        def log_prior(params):
            rp, a, inc, baseline, slope = params
            # Uniform priors
            if not (0.05 < rp < 0.2): return -np.inf
            if not (3 < a < 15): return -np.inf
            if not (70 < inc < 90): return -np.inf
            return 0.0
        
        def log_probability(params, times, fluxes, errors):
            lp = log_prior(params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(params, times, fluxes, errors)
        
        # Initial positions
        initial_params = [0.1, 7.0, 82.0, 1.0, 0.0]
        pos = initial_params + 1e-4 * np.random.randn(nwalkers, len(initial_params))
        
        # Run MCMC
        sampler = emcee.EnsembleSampler(
            nwalkers, len(initial_params), log_probability,
            args=(times, fluxes, errors)
        )
        sampler.run_mcmc(pos, nsteps, progress=True)
        
        # Extract results (median + std from chains)
        samples = sampler.get_chain(discard=1000, thin=15, flat=True)
        
        fitted_params = {}
        param_names = ['rp', 'a', 'inc', 'baseline', 'slope']
        for i, name in enumerate(param_names):
            median = np.median(samples[:, i])
            std = np.std(samples[:, i])
            fitted_params[name] = (median, std)
        
        # Compute model, residuals, chi2
        best_params = [fitted_params[p][0] for p in param_names]
        model_flux = self.model_with_detrending(times, *best_params)
        residuals = fluxes - model_flux
        chi_squared = np.sum((residuals / errors)**2)
        
        return {
            'fitted_params': fitted_params,
            't0': self.t0_guess,
            'model_flux': model_flux,
            'residuals': residuals,
            'chi_squared': chi_squared,
            'reduced_chi_squared': chi_squared / (len(times) - len(param_names)),
            'samples': samples  # Extra: full posterior samples
        }
```

#### Usage

```python
from pytransitphotometry.models import MCMCTransitFitter

fitter = MCMCTransitFitter(period=2.4842, t0_guess=2460235.752)
result = fitter.fit(times, fluxes, errors, nwalkers=50, nsteps=5000)

# Plot posterior distributions
import corner
corner.corner(result['samples'], 
              labels=['Rp/Rs', 'a/Rs', 'inc', 'baseline', 'slope'])
```

---

### Custom Visualization

**Scenario:** Add corner plot for MCMC results.

In `visualization.py`:

```python
def plot_mcmc_corner(
    samples: np.ndarray,
    param_names: List[str],
    truths: Optional[List[float]] = None,
    save_path: Optional[str] = None
):
    """
    Corner plot for MCMC posterior samples.
    
    Parameters
    ----------
    samples : np.ndarray
        MCMC samples (n_samples, n_params)
    param_names : list of str
        Parameter names
    truths : list of float, optional
        True values to overplot
    save_path : str, optional
        Save figure to path
    """
    import corner
    
    fig = corner.corner(
        samples,
        labels=param_names,
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()
```

---

## Testing Strategy

### Current State

**⚠️ WARNING:** As of v1.1.0, 36 unit tests cover all pipeline stages.

### Recommended Testing Structure

```
tests/
├── __init__.py
├── conftest.py              # pytest fixtures
│
├── test_calibration.py      # CCD calibration tests
├── test_detection.py        # Source detection tests
├── test_photometry.py       # Aperture photometry tests
├── test_lightcurve.py       # Light curve construction tests
├── test_detrending.py       # Detrending tests
├── test_models.py           # Transit fitting tests
│
├── test_io.py               # I/O utilities tests
├── test_config.py           # Configuration validation tests
├── test_pipeline.py         # End-to-end pipeline tests
│
└── fixtures/                # Test data
    ├── synthetic_bias.fits
    ├── synthetic_dark.fits
    ├── synthetic_flat.fits
    └── synthetic_transit.fits
```

### Example Tests

#### Unit Test: Calibration

`tests/test_calibration.py`:

```python
import numpy as np
import pytest
from pytransitphotometry.calibration import (
    create_master_frame,
    scale_dark_frame,
    calibrate_image
)

def test_create_master_frame_median():
    """Test median combination of calibration frames."""
    # Create synthetic frames with known median
    frames = np.array([
        np.ones((100, 100)) * 100,
        np.ones((100, 100)) * 102,
        np.ones((100, 100)) * 98
    ])
    
    master = create_master_frame(frames, method='median')
    
    assert master.shape == (100, 100)
    assert np.allclose(master, 100.0)

def test_create_master_frame_mean():
    """Test mean combination."""
    frames = np.array([
        np.ones((100, 100)) * 100,
        np.ones((100, 100)) * 102,
        np.ones((100, 100)) * 98
    ])
    
    master = create_master_frame(frames, method='mean')
    
    assert np.allclose(master, 100.0)

def test_scale_dark_frame():
    """Test dark current scaling."""
    bias = np.ones((100, 100)) * 100
    dark = np.ones((100, 100)) * 110  # bias + 10 counts dark
    
    # Scale from 10s to 30s
    scaled = scale_dark_frame(dark, bias, dark_exptime=10.0, target_exptime=30.0)
    
    # Expected: (110 - 100) / 10 * 30 = 30
    assert np.allclose(scaled, 30.0)

def test_calibrate_image():
    """Test full calibration pipeline."""
    raw = np.ones((100, 100)) * 1000
    bias = np.ones((100, 100)) * 100
    dark = np.ones((100, 100)) * 20
    flat = np.ones((100, 100)) * 1.0
    
    calibrated = calibrate_image(raw, bias, dark, flat)
    
    # Expected: (1000 - 100 - 20) / 1.0 = 880
    assert np.allclose(calibrated, 880.0)
```

#### Integration Test: Pipeline

`tests/test_pipeline.py`:

```python
import pytest
from pytransitphotometry import TransitPipeline, PipelineConfig

def test_pipeline_with_synthetic_data(tmp_path):
    """Test full pipeline with synthetic transit data."""
    # Generate synthetic data
    from tests.utils import create_synthetic_transit_dataset
    
    create_synthetic_transit_dataset(
        output_dir=tmp_path,
        n_frames=50,
        transit_params={'rp': 0.1, 'a': 7.0, 'inc': 85.0}
    )
    
    # Create config
    config = PipelineConfig.from_yaml('tests/fixtures/test_config.yaml')
    config.paths.data_dir = str(tmp_path / 'data')
    config.paths.bias_dir = str(tmp_path / 'bias')
    config.paths.dark_dir = str(tmp_path / 'darks')
    config.paths.flat_dir = str(tmp_path / 'flats')
    config.paths.output_dir = str(tmp_path / 'outputs')
    
    # Run pipeline
    pipeline = TransitPipeline(config)
    results = pipeline.run()
    
    # Validate results
    fit = results['fit_result']['fitted_params']
    
    # Check Rp/Rs recovered within 10%
    assert abs(fit['rp'][0] - 0.1) < 0.01
    
    # Check a/Rs recovered within 10%
    assert abs(fit['a'][0] - 7.0) < 0.7
    
    # Check χ²_red reasonable
    assert 0.5 < results['fit_result']['reduced_chi_squared'] < 2.0
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=pytransitphotometry --cov-report=html tests/

# Run specific test file
pytest tests/test_calibration.py

# Run specific test
pytest tests/test_calibration.py::test_scale_dark_frame

# Run with verbose output
pytest -v tests/
```

### Test Data Generation

`tests/utils.py`:

```python
import numpy as np
from astropy.io import fits

def create_synthetic_transit_dataset(
    output_dir,
    n_frames=50,
    image_size=(512, 512),
    transit_params=None
):
    """
    Create synthetic transit dataset for testing.
    
    Parameters
    ----------
    output_dir : Path
        Output directory
    n_frames : int
        Number of frames
    image_size : tuple
        Image dimensions
    transit_params : dict
        Transit parameters {'rp': 0.1, 'a': 7.0, 'inc': 85.0, ...}
    """
    from pytransitphotometry.models import batman_transit_model
    
    # Create directories
    (output_dir / 'bias').mkdir(parents=True, exist_ok=True)
    (output_dir / 'darks').mkdir(parents=True, exist_ok=True)
    (output_dir / 'flats').mkdir(parents=True, exist_ok=True)
    (output_dir / 'data').mkdir(parents=True, exist_ok=True)
    
    # Generate bias frames
    bias_level = 100.0
    for i in range(10):
        bias = np.random.normal(bias_level, 5, image_size).astype(np.float32)
        fits.writeto(output_dir / 'bias' / f'bias_{i:03d}.fits', bias, overwrite=True)
    
    # Generate dark frames
    dark_current = 0.1  # e-/px/s
    dark_exptime = 60.0
    for i in range(10):
        dark = bias_level + dark_current * dark_exptime
        dark += np.random.normal(0, 5, image_size)
        fits.writeto(output_dir / 'darks' / f'dark_{i:03d}.fits', dark.astype(np.float32), overwrite=True)
    
    # Generate flat frames
    for i in range(10):
        flat = np.ones(image_size) * 30000
        flat += np.random.normal(0, 100, image_size)
        fits.writeto(output_dir / 'flats' / f'flat_{i:03d}.fits', flat.astype(np.float32), overwrite=True)
    
    # Generate science frames with synthetic stars and transit
    times = np.linspace(0, 0.2, n_frames)  # 0.2 days
    
    if transit_params is None:
        transit_params = {'rp': 0.1, 'a': 7.0, 'inc': 85.0}
    
    transit_model = batman_transit_model(
        times, t0=0.1, period=2.0,
        rp=transit_params['rp'],
        a=transit_params['a'],
        inc=transit_params['inc']
    )
    
    for i, (time, flux_ratio) in enumerate(zip(times, transit_model)):
        # Create image with stars
        image = np.random.normal(100, 10, image_size)
        
        # Add target star (affected by transit)
        target_flux = 50000 * flux_ratio
        add_star(image, position=(256, 256), flux=target_flux, fwhm=5.0)
        
        # Add reference stars
        add_star(image, position=(150, 150), flux=40000, fwhm=5.0)
        add_star(image, position=(350, 350), flux=45000, fwhm=5.0)
        
        # Add header
        header = fits.Header()
        header['JD-HELIO'] = 2460000.0 + time
        header['AIRMASS'] = 1.2
        header['EXPTIME'] = 60.0
        
        fits.writeto(output_dir / 'data' / f'science_{i:03d}.fits',
                     image.astype(np.float32), header=header, overwrite=True)

def add_star(image, position, flux, fwhm):
    """Add synthetic star to image."""
    from photutils.datasets import make_gaussian_sources_image
    # Implementation...
```

---

## Known Limitations

### 1. Memory Consumption

**Issue:** Loads all images into memory as 3D NumPy array.

**Impact:**
- 100 frames @ 2048×2048 pixels = ~1.6 GB RAM
- Large datasets (>500 frames) may exceed available memory

**Workaround:**
- Process subset of frames
- Implement lazy loading (read frames on-demand)

**Future:** Add `lazy_loading` option to config

---

### 2. Single-Band Photometry

**Issue:** Assumes single-filter observations.

**Impact:**
- Cannot derive wavelength-dependent parameters
- No chromatic analysis (atmospheric differential refraction, limb darkening wavelength dependence)

**Workaround:**
- Run pipeline separately for each filter
- Manual combination of results

**Future:** Add multi-band support with simultaneous fitting

---

### 3. Linear Systematics Only

**Issue:** Detrending limited to linear trends and sigma clipping.

**Impact:**
- Cannot handle complex systematics (e.g., airmass², non-linear drifts)
- Struggles with variable conditions

**Workaround:**
- Pre-select good observing conditions
- Use custom detrending (GP, polynomial)

**Future:** Integrate Gaussian Process regression, polynomial fitting

---

### 4. Fixed Orbital Period

**Issue:** Period must be provided (not fitted).

**Impact:**
- Cannot discover periods
- Requires literature value or separate period analysis

**Workaround:**
- Use Lomb-Scargle periodogram on initial photometry
- Derive period from multi-epoch transits

**Future:** Add `fit_period` option for high-precision multi-transit data

---

### 5. No PSF Photometry

**Issue:** Aperture photometry only, no PSF fitting.

**Impact:**
- Poor performance in crowded fields
- Cannot deblend close binaries

**Workaround:**
- Use for sparse fields only
- Pre-processing with SExtractor/Source Extractor

**Future:** Add PSF photometry backend (e.g., DAOPHOT, PSFEx)

---

### 6. Circular Orbits Assumed

**Issue:** Default eccentricity = 0.

**Impact:**
- Incorrect fit for eccentric systems (e > 0.1)

**Workaround:**
- Set `eccentricity` and `omega` in config from literature

**Future:** Allow fitting eccentricity (with proper priors)

---

### 7. No Airmass Correction Applied

**Issue:** `test_airmass_correlation()` is diagnostic only.

**Impact:**
- If residual airmass correlation exists, bias in parameters

**Workaround:**
- Choose references with similar color to target
- Manually correct: `flux_corrected = flux × 10^(k×X/2.5)`

**Future:** Add `apply_airmass_correction` option

---

### 8. Limited Error Sources

**Issue:** Error model includes Poisson + background, but not:
- Scintillation
- Flat-fielding uncertainties
- Dark current subtraction errors

**Impact:**
- Underestimated uncertainties (especially for bright stars)

**Workaround:**
- Add systematic error floor: `σ_total² = σ_phot² + (0.001 × flux)²`

**Future:** Implement comprehensive error budget

---

## Technical Debt

### High Priority

1. **No Unit Tests**
   - **Debt:** Core algorithms not systematically tested
   - **Risk:** Regressions during refactoring
   - **Fix:** Implement test suite (see Testing Strategy above)
   - **Effort:** 2-3 weeks

2. **Inconsistent Error Handling**
   - **Debt:** Some functions raise exceptions, others return None or NaN
   - **Risk:** Difficult to debug failures
   - **Fix:** Standardize: critical errors raise exceptions, non-critical return sentinel values
   - **Effort:** 1 week

3. **Hardcoded Assumptions**
   - **Debt:** Some parameters (e.g., MAD→σ conversion factor) hardcoded
   - **Risk:** Not applicable to non-Gaussian noise
   - **Fix:** Move to config or auto-detect
   - **Effort:** 2 days

### Medium Priority

4. **No Logging Framework**
   - **Debt:** Uses `print()` instead of logging
   - **Risk:** Cannot control verbosity levels programmatically
   - **Fix:** Replace with Python `logging` module
   - **Effort:** 3 days

5. **Limited Validation**
   - **Debt:** Config validation is basic (file existence, positive values)
   - **Risk:** Invalid parameter combinations not caught early
   - **Fix:** Add semantic validation (e.g., aperture < annulus_inner)
   - **Effort:** 1 week

6. **No Progress Bars**
   - **Debt:** Long operations (frame loading, photometry) have minimal feedback
   - **Risk:** User uncertainty about pipeline status
   - **Fix:** Integrate `tqdm` for progress bars
   - **Effort:** 2 days

### Low Priority

7. **Type Hints Incomplete**
   - **Debt:** Not all functions have full type annotations
   - **Risk:** Reduced IDE support, potential type errors
   - **Fix:** Add type hints throughout
   - **Effort:** 1 week

8. **Matplotlib Warnings**
   - **Debt:** Some plots trigger deprecation warnings
   - **Risk:** Future matplotlib versions may break plotting
   - **Fix:** Update to latest matplotlib API
   - **Effort:** 1 day

---

## Future Roadmap

### Version 1.1 (Short-term: 3-6 months)

- [ ] Comprehensive unit test suite (>80% coverage)
- [ ] Logging framework integration
- [ ] Progress bars (tqdm)
- [ ] Config validation enhancements
- [ ] Multi-processing for frame-level operations
- [ ] Example Jupyter notebooks

### Version 1.2 (Medium-term: 6-12 months)

- [ ] Gaussian Process detrending
- [ ] MCMC fitting backend (emcee)
- [ ] Multi-band photometry support
- [ ] Airmass correction option
- [ ] PSF photometry backend (experimental)
- [ ] Database export (SQLite)

### Version 2.0 (Long-term: 1-2 years)

- [ ] Lazy loading (out-of-core processing)
- [ ] Distributed computing (Dask integration)
- [ ] Interactive GUI (Qt or web-based)
- [ ] Automatic period finding
- [ ] Advanced systematics modeling (polynomial, sinusoidal)
- [ ] Time-domain photometry (non-transiting variables)

### Community Wishlist

- [ ] Docker containerization
- [ ] Cloud deployment (AWS Lambda, Google Cloud Functions)
- [ ] Integration with TESS/Kepler pipelines
- [ ] Machine learning star/cosmic ray classification
- [ ] Real-time analysis (for live observations)

---

## Contribution Guidelines

### How to Contribute

1. **Fork repository** on GitHub
2. **Create feature branch**: `git checkout -b feature/my-new-feature`
3. **Make changes** with clear commit messages
4. **Add tests** for new functionality
5. **Run tests**: `pytest tests/`
6. **Format code**: `black pytransitphotometry/`
7. **Lint**: `flake8 pytransitphotometry/`
8. **Push branch**: `git push origin feature/my-new-feature`
9. **Open Pull Request** with description of changes

### Code Style

- Follow **PEP 8** (enforced by flake8)
- Use **black** for automatic formatting
- Maximum line length: **100 characters**
- Docstrings: **NumPy style**
- Type hints: **encouraged** for new code

### Commit Messages

Good commit messages help maintainers understand changes:

```
Add Gaussian Process detrending option

- Implement gp_detrend() in detrending.py
- Add use_gp_detrending config option
- Update pipeline.py to call GP detrending
- Add tests for GP detrending
- Document in usage.md

Closes #42
```

### Pull Request Checklist

- [ ] Code passes all tests (`pytest`)
- [ ] New tests added for new features
- [ ] Code formatted (`black pytransitphotometry/`)
- [ ] No linting errors (`flake8`)
- [ ] Documentation updated (docstrings, usage.md)
- [ ] CHANGELOG.md updated
- [ ] PR description explains motivation and changes

### Reporting Bugs

Open GitHub issue with:

1. **Title**: Concise description (e.g., "Transit fit fails for partial transits")
2. **Description**:
   - What you expected
   - What actually happened
   - Minimal reproducible example
3. **Environment**:
   - pyTransit version
   - Python version
   - OS (macOS, Linux, Windows)
4. **Attachments**:
   - Config file (sanitized paths)
   - Error traceback
   - Diagnostic plots (if relevant)

### Feature Requests

Open GitHub issue with:

1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How should it work?
3. **Alternatives**: What other approaches exist?
4. **Backwards compatibility**: Will it break existing code?

---

## Getting Help

### Documentation

- **Overview**: [DOCUMENTATION.md](DOCUMENTATION.md)
- **Architecture**: [architecture.md](architecture.md)
- **API**: [api.md](api.md)
- **Usage**: [usage.md](usage.md)
- **Development**: [DEVELOPMENT.md](DEVELOPMENT.md) (this file)

### Community

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Q&A, ideas, show-and-tell
- **Stack Overflow**: Tag with `pytransitphotometry`

### Academic Support

For scientific questions (not code bugs):
- Astronomy StackExchange
- Exoplanet mailing lists
- AAS Division on Extreme Solar Systems (DExSS)

---

## License

MIT License - see [LICENSE](../LICENSE) file.

**Summary:** Free to use, modify, distribute. No warranty. Cite in publications.

---

## Acknowledgments

**Built with:**
- **numpy, scipy**: Numerical computing
- **astropy**: Astronomical data structures
- **photutils**: Source detection and aperture photometry
- **batman**: Transit model computation (Kreidberg 2015)
- **matplotlib**: Visualization

**Inspired by:**
- **AstroImageJ**: GUI-based transit photometry
- **EXOTIC**: EXOplanet Transit Interpretation Code
- **lightkurve**: Kepler/TESS light curve analysis

---

**Happy coding!** 🚀 Contributions welcome. Let's advance exoplanet science together.
