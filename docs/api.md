# API Reference

**Complete Function and Class Documentation for pyTransitPhotometry**

---

## Table of Contents

1. [Core Pipeline](#core-pipeline)
2. [Configuration](#configuration)
3. [Calibration Module](#calibration-module)
4. [Detection Module](#detection-module)
5. [Photometry Module](#photometry-module)
6. [Light Curve Module](#light-curve-module)
7. [Detrending Module](#detrending-module)
8. [Models Module](#models-module)
9. [I/O Module](#io-module)
10. [Visualization Module](#visualization-module)
11. [CLI Module](#cli-module)

---

## Core Pipeline

### `TransitPipeline`

**Class:** `pytransitphotometry.pipeline.TransitPipeline`

Main orchestrator for the complete transit photometry workflow.

#### Constructor

```python
TransitPipeline(config: PipelineConfig)
```

**Parameters:**
- `config` (PipelineConfig): Pipeline configuration object loaded from YAML

**Attributes:**
- `config` (PipelineConfig): Configuration object
- `calibration_frames` (CalibrationFrames): Master calibration frames
- `science_data` (np.ndarray): Raw science images (3D array)
- `headers` (list[dict]): FITS headers for each frame
- `calibrated_images` (np.ndarray): Calibrated science images
- `sources_list` (list[Table]): Detected sources per frame
- `lightcurve` (dict): Differential photometry light curve
- `detrended_lc` (dict): Detrended light curve
- `fit_result` (dict): Transit fit parameters and uncertainties

#### Methods

##### `run() -> dict`

Execute complete pipeline from calibration to fitting.

**Returns:**
- `results` (dict): Contains keys:
  - `'lightcurve'`: Raw differential photometry
  - `'detrended_lc'`: Cleaned light curve
  - `'fit_result'`: Transit model parameters
  - `'config'`: Pipeline configuration

**Raises:**
- `RuntimeError`: If any stage fails critically

**Example:**
```python
from pytransitphotometry import TransitPipeline, PipelineConfig

config = PipelineConfig.from_yaml('config.yaml')
pipeline = TransitPipeline(config)
results = pipeline.run()

print(f"Transit depth: {results['fit_result']['derived_params']['transit_depth_pct']:.3f}%")
```

##### `run_calibration()`

Stage 1: Load calibration frames and apply CCD corrections.

**Side Effects:**
- Sets `self.calibration_frames`
- Sets `self.science_data` and `self.headers`
- Sets `self.calibrated_images`

**Raises:**
- `FileNotFoundError`: If calibration or science directories missing
- `ValueError`: If frames have inconsistent dimensions

##### `run_detection()`

Stage 2: Detect stars in all calibrated frames.

**Prerequisites:** Must call `run_calibration()` first.

**Side Effects:**
- Sets `self.sources_list`: list of astropy Tables, one per frame

**Raises:**
- `RuntimeError`: If no sources detected in any frame

##### `run_photometry()`

Stage 3: Extract aperture photometry for target and references.

**Prerequisites:** Must call `run_detection()` first.

**Side Effects:**
- Sets `self.lightcurve`: differential photometry light curve dict

**Raises:**
- `ValueError`: If target or reference indices out of range

##### `run_detrending()`

Stage 4: Remove outliers and systematic trends.

**Prerequisites:** Must call `run_photometry()` first.

**Side Effects:**
- Sets `self.detrended_lc`: cleaned light curve dict

##### `run_transit_fit()`

Stage 5: Fit batman transit model to detrended light curve.

**Prerequisites:** Must call `run_detrending()` first.

**Side Effects:**
- Sets `self.fit_result`: fitted parameters and statistics

**Raises:**
- `RuntimeError`: If optimization fails to converge
- `ImportError`: If batman-package not installed

##### `export_results()`

Stage 6: Save light curves, fit results, and diagnostic plots.

**Prerequisites:** Must have run previous stages.

**Side Effects:**
- Writes CSV files to `config.paths.output_dir`
- Saves PNG plots if `config.plot_diagnostics=True`

---

## Configuration

### `PipelineConfig`

**Class:** `pytransitphotometry.config.PipelineConfig`

Top-level configuration container.

#### Class Method

##### `from_yaml(filepath: str) -> PipelineConfig`

Load configuration from YAML file.

**Parameters:**
- `filepath` (str): Path to YAML configuration file

**Returns:**
- `config` (PipelineConfig): Parsed configuration object

**Raises:**
- `FileNotFoundError`: If YAML file doesn't exist
- `yaml.YAMLError`: If YAML syntax invalid
- `TypeError`: If required keys missing

**Example:**
```python
config = PipelineConfig.from_yaml('my_config.yaml')
print(f"Data directory: {config.paths.data_dir}")
print(f"Aperture radius: {config.photometry.aperture_radius} px")
```

##### `to_yaml(filepath: str)`

Save configuration to YAML file.

**Parameters:**
- `filepath` (str): Output path for YAML file

**Example:**
```python
config.photometry.aperture_radius = 8.0  # Modify parameter
config.to_yaml('modified_config.yaml')    # Save changes
```

##### `validate()`

Check configuration validity.

**Raises:**
- `FileNotFoundError`: If data directories don't exist
- `ValueError`: If parameters out of valid range (e.g., negative aperture radius)

**Called automatically** by `TransitPipeline.__init__()`.

#### Nested Configuration Classes

##### `PathConfig`

```python
@dataclass
class PathConfig:
    data_dir: str           # Science frames directory
    bias_dir: str           # Bias frames directory
    dark_dir: str           # Dark frames directory
    flat_dir: str           # Flat field directory
    output_dir: str         # Results output directory
    data_pattern: str       # Glob pattern (default: "*.fit")
    bias_pattern: str
    dark_pattern: str
    flat_pattern: str
```

##### `CalibrationConfig`

```python
@dataclass
class CalibrationConfig:
    dark_exptime: float            # Dark exposure time (seconds)
    flat_exptime: float            # Flat exposure time (seconds)
    science_exptime: float         # Science exposure time (seconds)
    combination_method: str        # "median" or "mean"
    sigma_clip: Optional[float]    # Sigma threshold for clipping (e.g., 3.0)
```

##### `DetectionConfig`

```python
@dataclass
class DetectionConfig:
    fwhm: float                    # PSF FWHM (pixels)
    threshold: float               # Detection threshold
    threshold_type: str            # "absolute" (counts) or "sigma" (SNR)
    exclude_border: bool           # Exclude edge sources
    min_sharpness: float           # Min sharpness (0.3 typical)
    max_sharpness: float           # Max sharpness (1.0 rejects cosmic rays)
    max_roundness: float           # Max |roundness| (0.5 typical)
```

##### `PhotometryConfig`

```python
@dataclass
class PhotometryConfig:
    aperture_radius: float                # Aperture radius (pixels)
    annulus_inner: float                  # Background annulus inner radius
    annulus_outer: float                  # Background annulus outer radius
    optimize_aperture: bool               # Auto-optimize radius?
    aperture_radii_test: list             # Radii to test for optimization
    target_star_index: int                # Target index in sorted sources
    reference_star_indices: list          # Reference star indices
    reference_weighting: str              # "inverse_variance" or "equal"
```

##### `DetrendingConfig`

```python
@dataclass
class DetrendingConfig:
    sigma_threshold: float         # Sigma clipping threshold (3.0 typical)
    max_iterations: int            # Max clipping iterations (5 typical)
    remove_linear_trend: bool      # Remove linear time trend?
    test_airmass: bool             # Test airmass correlation?
```

##### `TransitModelConfig`

```python
@dataclass
class TransitModelConfig:
    period: float                          # Orbital period (days)
    t0_guess: Optional[float]              # Transit center time (MJD)
    fix_t0: bool                           # Fix t0 during fit?
    limb_dark_u1: float                    # Limb darkening u1
    limb_dark_u2: float                    # Limb darkening u2
    eccentricity: float                    # Orbital eccentricity
    omega: float                           # Argument of periastron (deg)
    rp_guess: float                        # Initial Rp/Rs guess
    a_guess: float                         # Initial a/Rs guess
    inc_guess: float                       # Initial inclination (deg)
    rp_bounds: tuple                       # (min, max) for Rp/Rs
    a_bounds_factor: float                 # Fractional bounds on a/Rs
    inc_bounds_offset: float               # Degree bounds on inclination
    r_star_solar: Optional[float]          # Stellar radius (R☉)
    m_star_solar: Optional[float]          # Stellar mass (M☉)
```

---

## Calibration Module

### `create_master_frame()`

```python
create_master_frame(
    frames: np.ndarray,
    method: str = "median",
    sigma_clip: Optional[float] = None
) -> np.ndarray
```

Combine multiple calibration frames into master frame.

**Parameters:**
- `frames` (np.ndarray): 3D array (n_frames, height, width)
- `method` (str): "median" (robust, default) or "mean" (higher SNR)
- `sigma_clip` (float, optional): Sigma threshold for outlier rejection (e.g., 3.0)

**Returns:**
- `master` (np.ndarray): 2D master calibration frame

**Raises:**
- `ValueError`: If frames not 3D or method invalid

**Notes:**
- Median recommended for cosmic ray rejection
- Warns if fewer than 5 frames (insufficient statistics)
- Sigma clipping uses MAD-based robust scatter estimate

**Example:**
```python
master_bias = create_master_frame(bias_frames, method='median')
master_flat = create_master_frame(flat_frames, method='median', sigma_clip=3.0)
```

### `scale_dark_frame()`

```python
scale_dark_frame(
    master_dark: np.ndarray,
    master_bias: np.ndarray,
    dark_exptime: float,
    target_exptime: float
) -> np.ndarray
```

Scale dark current to match target exposure time.

**Physics:** `dark(t) = bias + dark_rate × t`

**Parameters:**
- `master_dark` (np.ndarray): Master dark frame
- `master_bias` (np.ndarray): Master bias frame
- `dark_exptime` (float): Exposure time of dark frames (seconds)
- `target_exptime` (float): Desired exposure time (seconds)

**Returns:**
- `scaled_dark` (np.ndarray): Bias-subtracted dark scaled to target time

**Raises:**
- `ValueError`: If exposure times not positive

**Example:**
```python
# Scale dark to match science frames (85s)
dark_for_science = scale_dark_frame(master_dark, master_bias, 
                                     dark_exptime=85.0, target_exptime=85.0)

# Scale dark to match flat frames (1s)
dark_for_flat = scale_dark_frame(master_dark, master_bias,
                                  dark_exptime=85.0, target_exptime=1.0)
```

### `calibrate_image()`

```python
calibrate_image(
    raw_image: np.ndarray,
    master_bias: np.ndarray,
    scaled_dark: np.ndarray,
    normalized_flat: np.ndarray
) -> np.ndarray
```

Apply full CCD calibration: `(raw - bias - dark) / flat`

**Parameters:**
- `raw_image` (np.ndarray): Raw science frame
- `master_bias` (np.ndarray): Master bias
- `scaled_dark` (np.ndarray): Dark scaled to science exposure time
- `normalized_flat` (np.ndarray): Normalized flat field (mean=1)

**Returns:**
- `calibrated` (np.ndarray): Calibrated science frame

**Raises:**
- `ValueError`: If array shapes don't match

### `CalibrationFrames`

**Class:** Container for master calibration frames with validation.

#### Constructor

```python
CalibrationFrames(
    master_bias: np.ndarray,
    master_dark: np.ndarray,
    master_flat: np.ndarray,
    dark_exptime: float,
    flat_exptime: float
)
```

**Methods:**
- `calibrate_science(raw, science_exptime)`: Apply calibration to science frame
- `get_normalized_flat()`: Return normalized flat field

---

## Detection Module

### `detect_sources()`

```python
detect_sources(
    image: np.ndarray,
    fwhm: float = 5.0,
    threshold: float = 10.0,
    threshold_type: str = "absolute",
    background_std: Optional[float] = None,
    exclude_border: bool = True,
    sort_by: str = "flux"
) -> astropy.table.Table
```

Detect point sources using DAOStarFinder.

**Parameters:**
- `image` (np.ndarray): 2D image
- `fwhm` (float): PSF FWHM in pixels (default: 5.0)
- `threshold` (float): Detection threshold (default: 10.0)
- `threshold_type` (str): "absolute" (counts) or "sigma" (SNR)
- `background_std` (float, optional): Background RMS (required if threshold_type="sigma")
- `exclude_border` (bool): Exclude edge sources (default: True)
- `sort_by` (str): Sort by "flux" (default) or "sharpness"

**Returns:**
- `sources` (astropy.table.Table): Detected sources with columns:
  - `id`: Source ID
  - `xcentroid`, `ycentroid`: Pixel coordinates
  - `flux`: Integrated flux
  - `peak`: Peak pixel value
  - `sharpness`: Point source quality metric (0.3-0.8 = good)
  - `roundness`: Elongation metric (0 = circular)
  - `npix`: Number of pixels above threshold

**Raises:**
- `ValueError`: If threshold_type="sigma" but background_std not provided
- `RuntimeError`: If no sources detected

**Algorithm:**
1. Convolve image with Gaussian kernel (FWHM)
2. Find local maxima above threshold
3. Fit 1D Gaussian profiles to refine centroids
4. Compute sharpness and roundness

**Example:**
```python
sources = detect_sources(calibrated_image, fwhm=5.0, threshold=10000.0)
print(f"Detected {len(sources)} stars")
print(f"Brightest at ({sources['xcentroid'][0]:.1f}, {sources['ycentroid'][0]:.1f})")
```

### `filter_sources()`

```python
filter_sources(
    sources: Table,
    min_sharpness: float = 0.3,
    max_sharpness: float = 1.0,
    max_roundness: float = 0.5,
    min_flux: Optional[float] = None
) -> Table
```

Apply quality filters to source catalog.

**Parameters:**
- `sources` (Table): Source table from `detect_sources()`
- `min_sharpness` (float): Minimum sharpness (0.3 rejects extended/blended sources)
- `max_sharpness` (float): Maximum sharpness (1.0 rejects cosmic rays)
- `max_roundness` (float): Maximum |roundness| (0.5 rejects elongated sources)
- `min_flux` (float, optional): Minimum flux threshold

**Returns:**
- `filtered` (Table): Filtered source catalog

**Quality Metrics:**
- **Sharpness**: Ratio of central pixel to surrounding pixels
  - Good stars: 0.4-0.8
  - Extended: <0.3
  - Cosmic rays: >1.0
- **Roundness**: $(2σ_x - 2σ_y) / (2σ_x + 2σ_y)$
  - Circular: ~0
  - Elongated: >0.3

**Example:**
```python
good_sources = filter_sources(sources, 
                                min_sharpness=0.3,
                                max_sharpness=1.0,
                                max_roundness=0.5)
```

### `select_reference_stars()`

```python
select_reference_stars(
    sources: Table,
    target_index: int,
    n_references: int = 3,
    max_separation: Optional[float] = None
) -> tuple
```

Select reference stars for differential photometry.

**Parameters:**
- `sources` (Table): Source catalog sorted by brightness
- `target_index` (int): Index of target star
- `n_references` (int): Number of reference stars (default: 3)
- `max_separation` (float, optional): Maximum pixel distance from target

**Returns:**
- `target_position` (tuple): (x, y) coordinates of target
- `reference_positions` (list): List of (x, y) for references
- `reference_indices` (list): Indices of selected references

**Selection Criteria:**
- Brightest stars (high SNR)
- Excludes target itself
- Optionally nearby (minimize differential atmospheric effects)

**Example:**
```python
target_pos, ref_positions, ref_indices = select_reference_stars(
    sources, target_index=2, n_references=2, max_separation=500.0
)
```

---

## Photometry Module

### `measure_flux()`

```python
measure_flux(
    image: np.ndarray,
    position: Tuple[float, float],
    aperture_radius: float,
    annulus_inner: float,
    annulus_outer: float,
    ccd_gain: float = 1.0,
    error_map: Optional[np.ndarray] = None
) -> dict
```

Measure background-subtracted flux with uncertainties.

**Parameters:**
- `image` (np.ndarray): 2D image
- `position` (tuple): Initial (x, y) centroid
- `aperture_radius` (float): Photometry aperture radius (pixels)
- `annulus_inner` (float): Background annulus inner radius
- `annulus_outer` (float): Background annulus outer radius
- `ccd_gain` (float): CCD gain in e-/ADU (default: 1.0)
- `error_map` (np.ndarray, optional): Pre-computed error map

**Returns:**
- `result` (dict):
  - `'flux'` (float): Background-subtracted flux
  - `'flux_err'` (float): Flux uncertainty
  - `'background_mean'` (float): Background per pixel
  - `'background_std'` (float): Background RMS
  - `'snr'` (float): Signal-to-noise ratio
  - `'aperture_sum'` (float): Raw aperture sum
  - `'centroid'` (tuple): Refined (x, y) position

**Algorithm:**
1. Refine centroid via 2D Gaussian fit
2. Measure raw aperture flux
3. Estimate local background from annulus
4. Subtract background: `flux = aperture_sum - background × aperture_area`
5. Compute uncertainty: $σ^2 = |S| × g + N_{pix} × σ_{sky}^2$

**Example:**
```python
result = measure_flux(image, (512.3, 768.9), 
                       aperture_radius=8.0,
                       annulus_inner=40.0,
                       annulus_outer=60.0,
                       ccd_gain=1.5)
print(f"Flux: {result['flux']:.1f} ± {result['flux_err']:.1f} ADU")
print(f"SNR: {result['snr']:.1f}")
```

### `optimize_aperture_radius()`

```python
optimize_aperture_radius(
    image: np.ndarray,
    position: Tuple[float, float],
    radii: np.ndarray,
    annulus_inner: float,
    annulus_outer: float,
    ccd_gain: float = 1.0,
    return_snr_curve: bool = False
) -> float
```

Find optimal aperture radius that maximizes SNR.

**Parameters:**
- `image` (np.ndarray): 2D image
- `position` (tuple): Star centroid (x, y)
- `radii` (np.ndarray): Array of radii to test (e.g., `np.arange(3, 20, 1)`)
- `annulus_inner`, `annulus_outer` (float): Background annulus radii
- `ccd_gain` (float): CCD gain (default: 1.0)
- `return_snr_curve` (bool): Also return SNR vs radius curve

**Returns:**
- `optimal_radius` (float): Radius with maximum SNR
- `(radii, snr_values)` (tuple, optional): SNR curve if `return_snr_curve=True`

**Trade-off:**
- **Larger aperture**: Collects more photons, but more sky noise
- **Smaller aperture**: Less sky noise, but misses PSF wings
- **Optimal**: Typically 1-2× FWHM

**Example:**
```python
radii_test = np.arange(3, 20, 1)
optimal_r = optimize_aperture_radius(image, (x, y), radii_test,
                                      annulus_inner=40, annulus_outer=60,
                                      ccd_gain=1.5)
print(f"Optimal aperture: {optimal_r:.1f} pixels")
```

### `refine_centroid()`

```python
refine_centroid(
    image: np.ndarray,
    initial_position: Tuple[float, float],
    box_size: int = 51,
    centroid_func=centroid_2dg
) -> Tuple[float, float]
```

Refine star centroid using 2D Gaussian fitting.

**Parameters:**
- `image` (np.ndarray): 2D image
- `initial_position` (tuple): Initial (x, y) guess
- `box_size` (int): Cutout size for fitting (default: 51, must be odd)
- `centroid_func` (callable): Centroid algorithm (default: `centroid_2dg`)

**Returns:**
- `x_refined`, `y_refined` (float): Sub-pixel centroid

**Precision:** Typically 0.01-0.1 pixels for bright, isolated stars

**Example:**
```python
x_init, y_init = 512.0, 768.0
x_refined, y_refined = refine_centroid(image, (x_init, y_init))
print(f"Shift: Δx={x_refined-x_init:.3f}, Δy={y_refined-y_init:.3f} px")
```

---

## Light Curve Module

### `differential_photometry()`

```python
differential_photometry(
    target_flux: float,
    target_err: float,
    reference_fluxes: np.ndarray,
    reference_errs: np.ndarray,
    weighting: str = "inverse_variance"
) -> Tuple[float, float]
```

Compute differential photometry ratio with error propagation.

**Parameters:**
- `target_flux` (float): Target star flux
- `target_err` (float): Target uncertainty
- `reference_fluxes` (np.ndarray): Reference star fluxes
- `reference_errs` (np.ndarray): Reference uncertainties
- `weighting` (str): "inverse_variance" (default) or "equal"

**Returns:**
- `ratio` (float): Target flux / ensemble reference flux
- `ratio_err` (float): Propagated uncertainty

**Weighting:**
- **Inverse variance**: $w_i = 1/σ_i^2$ (optimal for uncorrelated noise)
- **Equal**: $w_i = 1$ (simple average)

**Error Propagation:**
$$σ_{ratio}^2 = ratio^2 × \left[\left(\frac{σ_{target}}{F_{target}}\right)^2 + \left(\frac{σ_{ref}}{F_{ref}}\right)^2\right]$$

**Example:**
```python
ratio, ratio_err = differential_photometry(
    target_flux=50000, target_err=100,
    reference_fluxes=np.array([40000, 45000]),
    reference_errs=np.array([80, 90]),
    weighting='inverse_variance'
)
```

### `LightCurveBuilder`

**Class:** Construct differential photometry light curve from multi-frame data.

#### Constructor

```python
LightCurveBuilder(
    target_index: int,
    reference_indices: List[int],
    weighting: str = "inverse_variance"
)
```

**Parameters:**
- `target_index` (int): Index of target in source list
- `reference_indices` (list): Indices of reference stars
- `weighting` (str): Reference combination method

**Raises:**
- `ValueError`: If target in reference list or no references provided

#### Method: `build()`

```python
build(
    images: np.ndarray,
    sources_per_frame: List,
    photometry_func: callable,
    time_extractor: callable,
    verbose: bool = True
) -> dict
```

Build light curve from image sequence.

**Parameters:**
- `images` (np.ndarray): 3D array (n_frames, height, width)
- `sources_per_frame` (list): Source tables, one per frame
- `photometry_func` (callable): `func(image, source_idx) -> flux_dict`
- `time_extractor` (callable): `func(frame_idx) -> time`
- `verbose` (bool): Print progress

**Returns:**
- `lightcurve` (dict):
  - `'times'`: Observation times
  - `'fluxes'`: Differential flux ratios
  - `'errors'`: Flux uncertainties
  - `'target_fluxes'`: Raw target fluxes
  - `'reference_fluxes'`: Combined reference fluxes
  - `'centroids'`: Target centroids per frame
  - `'valid_frames'`: Boolean mask

**Frame Skipping:** Frames excluded if:
- Required stars not detected
- Photometry fails
- Flux values invalid (negative, NaN)

---

## Detrending Module

### `sigma_clip()`

```python
sigma_clip(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    sigma_threshold: float = 3.0,
    max_iterations: int = 5,
    method: str = "median"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Iterative sigma clipping for outlier removal.

**Parameters:**
- `times`, `fluxes`, `errors` (np.ndarray): Light curve data
- `sigma_threshold` (float): Clipping threshold (default: 3.0σ)
- `max_iterations` (int): Maximum iterations (default: 5)
- `method` (str): "median" (robust, default) or "mean"

**Returns:**
- `times_clipped`, `fluxes_clipped`, `errors_clipped` (np.ndarray): Cleaned data
- `mask` (np.ndarray): Boolean mask (True=kept, False=rejected)

**Algorithm:**
1. Compute robust center (median) and scatter (MAD → σ)
2. Flag points >3σ from center
3. Also flag points with errors >3× median error
4. Repeat until convergence or max iterations

**Example:**
```python
times_clean, fluxes_clean, errs_clean, mask = sigma_clip(
    times, fluxes, errors, sigma_threshold=3.0
)
print(f"Rejected {np.sum(~mask)} outliers")
```

### `test_airmass_correlation()`

```python
test_airmass_correlation(
    airmass: np.ndarray,
    fluxes: np.ndarray
) -> dict
```

Test for correlation between airmass and flux.

**Parameters:**
- `airmass` (np.ndarray): Airmass values (1 = zenith)
- `fluxes` (np.ndarray): Flux ratios

**Returns:**
- `result` (dict):
  - `'correlation'`: Pearson r
  - `'slope'`: Linear fit slope
  - `'intercept'`: Linear fit intercept
  - `'trend_percent'`: Flux change over airmass range (%)
  - `'needs_correction'`: True if |r| > 0.3

**Interpretation:**
- **|r| < 0.3**: Weak correlation, no action needed
- **0.3 < |r| < 0.5**: Moderate, correction optional
- **|r| > 0.5**: Strong, investigate reference star selection

**Physics:** Differential photometry should cancel airmass effects. Residual correlation suggests:
- Color mismatch (target/references different spectral types)
- Non-photometric conditions
- Poor reference star choice

**Example:**
```python
result = test_airmass_correlation(airmass, fluxes)
if result['needs_correction']:
    print(f"Warning: Correlation r={result['correlation']:.3f}")
```

### `remove_linear_trend()`

```python
remove_linear_trend(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: Optional[np.ndarray] = None,
    return_model: bool = False
) -> Tuple[np.ndarray, float, float]
```

Remove linear trend from light curve.

**Parameters:**
- `times`, `fluxes` (np.ndarray): Light curve data
- `errors` (np.ndarray, optional): For weighted fit
- `return_model` (bool): Return model instead of detrended data

**Returns:**
- `detrended_fluxes` (np.ndarray): Fluxes with trend removed
- `slope` (float): Fitted slope
- `intercept` (float): Fitted intercept

**Model:** $F = slope × (t - t_{mean}) + intercept$

**Example:**
```python
fluxes_detrended, slope, intercept = remove_linear_trend(times, fluxes)
```

### `detrend_lightcurve()`

```python
detrend_lightcurve(
    lightcurve: dict,
    config: DetrendingConfig,
    airmass: Optional[np.ndarray] = None
) -> dict
```

Complete detrending workflow.

**Parameters:**
- `lightcurve` (dict): Raw light curve from `LightCurveBuilder`
- `config` (DetrendingConfig): Detrending configuration
- `airmass` (np.ndarray, optional): For correlation test

**Returns:**
- `detrended_lc` (dict): Cleaned light curve

**Steps:**
1. Sigma clipping (removes outliers)
2. Linear trend removal (optional)
3. Airmass correlation test (optional, diagnostic only)

---

## Models Module

### `batman_transit_model()`

```python
batman_transit_model(
    times: np.ndarray,
    t0: float,
    period: float,
    rp: float,
    a: float,
    inc: float,
    ecc: float = 0.0,
    w: float = 90.0,
    u1: float = 0.4,
    u2: float = 0.26,
    limb_dark: str = "quadratic"
) -> np.ndarray
```

Generate batman transit model light curve.

**Parameters:**
- `times` (np.ndarray): Observation times (MJD, BJD, or any consistent unit)
- `t0` (float): Transit center time (same units as times)
- `period` (float): Orbital period (days)
- `rp` (float): Planet-to-star radius ratio (Rp/Rs)
- `a` (float): Scaled semi-major axis (a/Rs)
- `inc` (float): Orbital inclination (degrees, 90°=edge-on)
- `ecc` (float): Eccentricity (0=circular)
- `w` (float): Argument of periastron (degrees)
- `u1`, `u2` (float): Quadratic limb darkening coefficients
- `limb_dark` (str): Limb darkening law ("quadratic", "linear", etc.)

**Returns:**
- `model_flux` (np.ndarray): Normalized transit light curve (1.0 out of transit)

**Physical Relations:**
- Transit depth: $δ = (R_p/R_s)^2$
- Impact parameter: $b = (a/R_s) × \cos(i)$
- Full transit when $b < 1 - R_p/R_s$

**Example:**
```python
model = batman_transit_model(
    times, t0=2460235.752, period=2.4842,
    rp=0.103, a=7.17, inc=82.0
)
```

### `TransitFitter`

**Class:** Fit transit model to light curve using optimization.

#### Constructor

```python
TransitFitter(
    period: float,
    t0_guess: float,
    limb_dark_u1: float = 0.40,
    limb_dark_u2: float = 0.26,
    ecc: float = 0.0,
    w: float = 90.0
)
```

**Parameters:**
- `period` (float): Orbital period (FIXED during fit)
- `t0_guess` (float): Initial guess for transit center
- `limb_dark_u1`, `limb_dark_u2` (float): Limb darkening (FIXED)
- `ecc` (float): Eccentricity (FIXED, 0 for circular)
- `w` (float): Argument of periastron (FIXED)

**Why Fix Parameters?**
- Period: Known precisely from multi-epoch observations
- Limb darkening: From stellar atmosphere models (fitting introduces degeneracies)
- Eccentricity: Most hot Jupiters circularized

#### Method: `fit()`

```python
fit(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    initial_params: Optional[Dict[str, float]] = None,
    bounds: Optional[Dict[str, Tuple]] = None,
    fix_t0: bool = True,
    maxfev: int = 10000
) -> Dict
```

Fit transit model to data.

**Parameters:**
- `times`, `fluxes`, `errors` (np.ndarray): Light curve data
- `initial_params` (dict, optional): Initial guesses for `rp`, `a`, `inc`, `baseline`, `slope`
- `bounds` (dict, optional): Parameter bounds (e.g., `{'rp': (0.08, 0.12)}`)
- `fix_t0` (bool): Fix transit center time (default: True, recommended)
- `maxfev` (int): Maximum function evaluations (default: 10000)

**Returns:**
- `result` (dict):
  - `'fitted_params'`: Dict of {param: (value, uncertainty)}
  - `'t0'`: Transit center time
  - `'model_flux'`: Best-fit model
  - `'residuals'`: Data - model
  - `'chi_squared'`: χ²
  - `'reduced_chi_squared'`: χ²/(N - n_params)
  - `'derived_params'`: Physical parameters (transit depth, duration, etc.)

**Fitted Parameters (if fix_t0=True):**
1. `rp`: Radius ratio (Rp/Rs)
2. `a`: Scaled semi-major axis (a/Rs)
3. `inc`: Inclination (degrees)
4. `baseline`: Flux normalization
5. `slope`: Linear time trend

**Model:**
$$F(t) = baseline × \text{transit}(t) × [1 + slope × (t - t_{mean})]$$

**Example:**
```python
fitter = TransitFitter(period=2.4842, t0_guess=2460235.752)
result = fitter.fit(times, fluxes, errors, fix_t0=True)

print(f"Rp/Rs = {result['fitted_params']['rp'][0]:.4f} ± {result['fitted_params']['rp'][1]:.4f}")
print(f"χ²_red = {result['reduced_chi_squared']:.2f}")
```

---

## I/O Module

### `load_fits_files()`

```python
load_fits_files(
    directory: str,
    pattern: str = "*.fits",
    dtype: np.dtype = np.float32,
    verbose: bool = True
) -> Tuple[np.ndarray, List[Dict]]
```

Load FITS files from directory.

**Parameters:**
- `directory` (str): Path to FITS directory
- `pattern` (str): Glob pattern (default: "*.fits")
- `dtype` (np.dtype): Array data type (default: float32 for memory efficiency)
- `verbose` (bool): Print progress

**Returns:**
- `data` (np.ndarray): 3D array (n_frames, height, width)
- `headers` (list[dict]): FITS headers as dictionaries

**Raises:**
- `FileNotFoundError`: If directory missing or no files match pattern
- `ValueError`: If files have inconsistent dimensions

**Notes:**
- Files sorted alphabetically for consistent ordering
- Skips files with no image data (warns)

### `extract_header_value()`

```python
extract_header_value(
    headers: List[Dict],
    key: str,
    default: Any = None,
    fallback_keys: List[str] = None
) -> np.ndarray
```

Extract keyword from multiple FITS headers.

**Parameters:**
- `headers` (list[dict]): FITS headers
- `key` (str): Primary keyword to extract
- `default` (any): Value if key not found
- `fallback_keys` (list[str], optional): Alternative keys to try

**Returns:**
- `values` (np.ndarray): Extracted values

**Example:**
```python
times = extract_header_value(headers, 'JD-HELIO', default=0.0)
airmass = extract_header_value(headers, 'AIRMASS', 
                                 fallback_keys=['SECZ', 'AIRMASS-START'])
```

### `get_ccd_gain()`

```python
get_ccd_gain(header: Dict) -> float
```

Extract CCD gain from FITS header.

**Parameters:**
- `header` (dict): FITS header

**Returns:**
- `gain` (float): CCD gain in e-/ADU

**Searches Keywords:** `GAIN`, `EGAIN`, `CCGAIN`, `CAMGAIN`, `CCDGAIN`

**Defaults to 1.0** if not found (with warning).

### `export_lightcurve()`

```python
export_lightcurve(
    output_path: str,
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    **metadata
)
```

Export light curve to CSV file.

**Parameters:**
- `output_path` (str): Output file path
- `times`, `fluxes`, `errors` (np.ndarray): Light curve data
- `**metadata`: Additional columns (e.g., `airmass=airmass_array`)

**Output Format:** CSV with columns: `time`, `flux`, `error`, [metadata columns]

### `export_fit_results()`

```python
export_fit_results(
    output_path: str,
    fit_result: Dict
)
```

Export transit fit results to JSON.

**Parameters:**
- `output_path` (str): Output file path
- `fit_result` (dict): Fit result from `TransitFitter.fit()`

---

## Visualization Module

### `plot_calibration_comparison()`

```python
plot_calibration_comparison(
    raw_image: np.ndarray,
    calibrated_image: np.ndarray,
    vmin: float = 700,
    vmax: float = 1400,
    figsize: Tuple = (14, 6),
    save_path: Optional[str] = None
)
```

Side-by-side comparison of raw and calibrated images.

**Parameters:**
- `raw_image`, `calibrated_image` (np.ndarray): 2D images
- `vmin`, `vmax` (float): Display range for LogNorm
- `figsize` (tuple): Figure size in inches
- `save_path` (str, optional): Save to file instead of displaying

### `plot_detected_sources()`

```python
plot_detected_sources(
    image: np.ndarray,
    sources: Table,
    target_index: Optional[int] = None,
    reference_indices: Optional[list] = None,
    figsize: Tuple = (12, 6),
    save_path: Optional[str] = None
)
```

Overlay detected sources on image.

**Parameters:**
- `image` (np.ndarray): Calibrated image
- `sources` (Table): Detected sources
- `target_index` (int, optional): Highlight target star
- `reference_indices` (list, optional): Highlight references
- `figsize`, `save_path`: Display options

### `plot_lightcurve()`

```python
plot_lightcurve(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    title: str = "Transit Light Curve",
    xlabel: str = "Time (MJD)",
    ylabel: str = "Relative Flux",
    figsize: Tuple = (12, 6),
    save_path: Optional[str] = None
)
```

Light curve with error bars and statistics.

**Parameters:**
- `times`, `fluxes`, `errors` (np.ndarray): Light curve data
- `title`, `xlabel`, `ylabel` (str): Labels
- `figsize`, `save_path`: Display options

**Features:**
- Error bars with caps
- Median baseline (dashed line)
- Statistics text box (N points, time span, RMS scatter)

### `plot_transit_fit()`

```python
plot_transit_fit(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    model_flux: np.ndarray,
    fit_result: Dict,
    figsize: Tuple = (14, 10),
    save_path: Optional[str] = None
)
```

Transit fit with data, model, and residuals.

**Parameters:**
- `times`, `fluxes`, `errors` (np.ndarray): Observed light curve
- `model_flux` (np.ndarray): Best-fit model
- `fit_result` (dict): Fit results from `TransitFitter`
- `figsize`, `save_path`: Display options

**Layout:**
- Top panel: Data + model
- Bottom panel: Residuals
- Text box: Fitted parameters with uncertainties

---

## CLI Module

### Command-Line Interface

**Entry point:** `pytransit` (installed via `pip install`)

#### Usage

```bash
# Run full pipeline
pytransit config.yaml

# Create example config
pytransit --create-config my_config.yaml

# Custom output directory
pytransit config.yaml --output ./my_results

# Run specific stages
pytransit config.yaml --stages calibration detection photometry

# Disable plots
pytransit config.yaml --no-plots

# Verbose mode
pytransit config.yaml --verbose
```

#### Arguments

- `config` (positional): Path to YAML configuration file
- `--create-config PATH`: Generate example config at PATH
- `--output DIR`, `-o DIR`: Override output directory from config
- `--stages STAGE [STAGE ...]`: Run only specified stages
  - Choices: `calibration`, `detection`, `photometry`, `detrending`, `fit`, `export`
- `--no-plots`: Disable diagnostic plotting
- `--verbose`, `-v`: Enable verbose output
- `--version`: Show version and exit

#### Example Workflow

```bash
# 1. Create template config
pytransit --create-config my_config.yaml

# 2. Edit config file (set data paths, target parameters)
nano my_config.yaml

# 3. Run pipeline
pytransit my_config.yaml

# 4. Results saved to outputs/ directory
ls outputs/
# lightcurve_raw.csv
# lightcurve_detrended.csv
# fit_results.json
# calibration_comparison.png
# detected_sources.png
# lightcurve_plot.png
# transit_fit.png
```

---

## Module Import Guide

### High-Level API (Recommended)

```python
from pytransitphotometry import TransitPipeline, PipelineConfig

config = PipelineConfig.from_yaml('config.yaml')
pipeline = TransitPipeline(config)
results = pipeline.run()
```

### Low-Level API (Advanced)

```python
# Direct access to algorithms
from pytransitphotometry.calibration import calibrate_image
from pytransitphotometry.detection import detect_sources
from pytransitphotometry.photometry import measure_flux
from pytransitphotometry.models import batman_transit_model

# Custom workflows
calibrated = calibrate_image(raw, bias, dark, flat)
sources = detect_sources(calibrated, fwhm=5.0, threshold=10000)
flux_dict = measure_flux(calibrated, (x, y), 8.0, 40, 60)
model = batman_transit_model(times, t0, period, rp, a, inc)
```

---

## Function Summary by Category

### Data Loading
- `load_fits_files()`: Directory → NumPy arrays
- `extract_header_value()`: Parse FITS keywords
- `get_ccd_gain()`: Extract gain from header

### Calibration
- `create_master_frame()`: Combine calibration frames
- `scale_dark_frame()`: Exposure-time scaling
- `calibrate_image()`: Apply CCD corrections
- `CalibrationFrames`: Container class

### Detection
- `detect_sources()`: DAOStarFinder wrapper
- `filter_sources()`: Quality filtering
- `select_reference_stars()`: Choose comparisons

### Photometry
- `measure_flux()`: Background-subtracted flux
- `optimize_aperture_radius()`: SNR maximization
- `refine_centroid()`: Sub-pixel positioning

### Light Curves
- `differential_photometry()`: Compute flux ratios
- `LightCurveBuilder.build()`: Multi-frame processing

### Detrending
- `sigma_clip()`: Outlier removal
- `test_airmass_correlation()`: Diagnostics
- `remove_linear_trend()`: Linear detrending
- `detrend_lightcurve()`: Complete workflow

### Modeling
- `batman_transit_model()`: Generate model
- `TransitFitter.fit()`: Parameter optimization

### Visualization
- `plot_calibration_comparison()`: Before/after images
- `plot_detected_sources()`: Source overlay
- `plot_lightcurve()`: Time series
- `plot_transit_fit()`: Fit diagnostics

### Export
- `export_lightcurve()`: Save to CSV
- `export_fit_results()`: Save to JSON

---

**Next:** See [usage.md](usage.md) for practical examples and [DEVELOPMENT.md](DEVELOPMENT.md) for extending the library.
