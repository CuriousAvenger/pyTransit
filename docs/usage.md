# Usage Guide

**Complete Guide to Installing, Configuring, and Running pyTransit**

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration File Anatomy](#configuration-file-anatomy)
4. [Step-by-Step Workflow](#step-by-step-workflow)
5. [Example Workflows](#example-workflows)
6. [Output Files](#output-files)
7. [Interpreting Results](#interpreting-results)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tips](#performance-tips)

---

## Installation

### Requirements

- **Python**: 3.8 or later
- **Operating System**: macOS, Linux, or Windows
- **RAM**: 4+ GB recommended (for typical 2048×2048 CCD images)
- **Disk Space**: ~500 MB for package + dependencies

### Option 1: Install from PyPI (Recommended)

```bash
pip install pytransitphotometry
```

This installs the library and all required dependencies.

### Option 2: Install from Source

```bash
# Clone repository
git clone https://github.com/CuriousAvenger/pyTransit.git
cd pytransitphotometry

# Install in editable mode (for development)
pip install -e .

# Or install normally
pip install .
```

### Option 3: Install in Virtual Environment (Best Practice)

```bash
# Create virtual environment
python -m venv transit-env

# Activate (macOS/Linux)
source transit-env/bin/activate

# Activate (Windows)
transit-env\Scripts\activate

# Install package
pip install pytransitphotometry
```

### Verify Installation

```bash
# Check command-line tool
pytransit --version
# Output: pytransitphotometry 1.1.0

# Check Python import
python -c "import pytransitphotometry; print(pytransitphotometry.__version__)"
# Output: 1.1.0
```

### Dependencies Installed Automatically

- **numpy** (≥1.20): Numerical arrays
- **scipy** (≥1.7): Optimization
- **astropy** (≥5.0): FITS I/O, astronomical utilities
- **photutils** (≥1.5): DAOStarFinder, aperture photometry
- **batman-package** (≥2.4.8): Transit model computation
- **matplotlib** (≥3.5): Plotting
- **pyyaml** (≥6.0): Configuration parsing
- **pandas** (≥1.3): CSV export

---

## Quick Start

### 1. Create Example Configuration

```bash
# Command-line
pytransit --create-config my_config.yaml

# Or in Python
from pytransitphotometry.config import create_example_config
create_example_config('my_config.yaml')
```

This generates a template configuration file with documentation.

### 2. Edit Configuration

Open `my_config.yaml` and update:

```yaml
paths:
  data_dir: "/path/to/your/science/frames"
  bias_dir: "/path/to/your/bias/frames"
  dark_dir: "/path/to/your/dark/frames"
  flat_dir: "/path/to/your/flat/frames"
  output_dir: "./outputs"

transit_model:
  t0_guess: 2460235.752  # Update to your predicted transit time!
```

### 3. Run Pipeline

**Command-line:**
```bash
pytransit my_config.yaml
```

**Python:**
```python
from pytransitphotometry import TransitPipeline, PipelineConfig

config = PipelineConfig.from_yaml('my_config.yaml')
pipeline = TransitPipeline(config)
results = pipeline.run()
```

### 4. Check Results

Results saved to `outputs/` directory:
- `lightcurve_raw.csv`: Raw differential photometry
- `lightcurve_detrended.csv`: Cleaned light curve
- `fit_results.json`: Transit parameters
- `*.png`: Diagnostic plots

---

## Configuration File Anatomy

### Complete Example with Annotations

```yaml
# ========================================================================
# DATA PATHS
# ========================================================================
paths:
  # Science frames: Your target observations
  data_dir: "./Group8.nosync/data"
  
  # Calibration frames (same instrument/CCD)
  bias_dir: "./Group8.nosync/bias"
  dark_dir: "./Group8.nosync/darks"
  flat_dir: "./Group8.nosync/flats"
  
  # Where to save results
  output_dir: "./outputs"
  
  # File patterns (use *.fits, *.fit, *.fts, etc.)
  data_pattern: "*.fit"
  bias_pattern: "*.fit"
  dark_pattern: "*.fit"
  flat_pattern: "*.fit"

# ========================================================================
# CCD CALIBRATION
# ========================================================================
calibration:
  # Exposure times (MUST match your observations)
  dark_exptime: 85.0      # Dark frame exposure (seconds)
  flat_exptime: 1.0       # Flat field exposure (seconds)
  science_exptime: 85.0   # Science frame exposure (seconds)
  
  # Frame combination method
  combination_method: "median"  # "median" (robust) or "mean" (higher SNR)
  
  # Outlier rejection during frame combination
  sigma_clip: null        # null = no clipping, 3.0 = 3-sigma clipping

# ========================================================================
# STAR DETECTION
# ========================================================================
detection:
  # PSF characteristics
  fwhm: 5.0               # Full-width half-maximum (pixels)
                          # Typical: 3-8 px for ground-based seeing
  
  # Detection threshold
  threshold: 10000.0      # Counts (if threshold_type="absolute")
                          # or SNR (if threshold_type="sigma")
  threshold_type: "absolute"  # "absolute" or "sigma"
  
  # Edge exclusion
  exclude_border: true    # Reject sources near image edges
  
  # Quality filters (reject blended sources, cosmic rays)
  min_sharpness: 0.3      # Reject extended sources (< 0.3)
  max_sharpness: 1.0      # Reject cosmic rays (> 1.0)
  max_roundness: 0.5      # Reject elongated sources (|r| > 0.5)

# ========================================================================
# APERTURE PHOTOMETRY
# ========================================================================
photometry:
  # Aperture sizes (pixels)
  aperture_radius: 6.0    # Photometry aperture (typically 1-2× FWHM)
  annulus_inner: 40.0     # Background annulus inner radius
  annulus_outer: 60.0     # Background annulus outer radius
  
  # Auto-optimization (finds radius that maximizes SNR)
  optimize_aperture: false          # Enable optimization?
  aperture_radii_test: [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18]
  
  # Star selection (0-indexed, sorted by brightness after detection)
  target_star_index: 2              # Which star is your target?
                                     # 0 = brightest, 1 = 2nd brightest, etc.
  
  reference_star_indices: [0, 1]    # Reference stars for differential photometry
                                     # Use 2-5 bright, non-variable stars
  
  # Reference star weighting
  reference_weighting: "inverse_variance"  # "inverse_variance" (optimal)
                                           # or "equal" (simple average)

# ========================================================================
# DETRENDING
# ========================================================================
detrending:
  # Outlier removal (sigma clipping)
  sigma_threshold: 3.0    # Remove points >3σ from median
  max_iterations: 5       # Maximum clipping iterations
  
  # Trend removal
  remove_linear_trend: true   # Remove linear time trend?
  test_airmass: true          # Test for airmass correlation?
                              # (diagnostic only, doesn't apply correction)

# ========================================================================
# TRANSIT MODEL
# ========================================================================
transit_model:
  # Orbital parameters (FROM LITERATURE)
  period: 2.4842                # Orbital period (days) - FIXED during fit
  t0_guess: 2460235.752         # Predicted transit center (MJD)
                                # ⚠️ UPDATE THIS FOR YOUR OBSERVATION!
  
  fix_t0: true                  # Fix t0 during fit?
                                # Recommended: true (unless full transit)
  
  # Limb darkening (from stellar atmosphere models)
  # For Sun-like stars in V-band: u1≈0.4, u2≈0.26
  # For other stars/filters: use lookup tables
  limb_dark_u1: 0.40
  limb_dark_u2: 0.26
  
  # Orbital configuration (usually fixed)
  eccentricity: 0.0       # 0 = circular orbit (most hot Jupiters)
  omega: 90.0             # Argument of periastron (degrees)
  
  # Initial guesses for fitting
  rp_guess: 0.103         # Planet-to-star radius ratio (Rp/Rs)
  a_guess: 7.17           # Scaled semi-major axis (a/Rs)
  inc_guess: 82.0         # Orbital inclination (degrees)
  
  # Parameter bounds (prevent unphysical solutions)
  rp_bounds: [0.085, 0.120]       # Hard min/max for Rp/Rs
  a_bounds_factor: 0.2            # ±20% of a_guess
  inc_bounds_offset: 6.0          # ±6° of inc_guess
  
  # Stellar parameters (OPTIONAL, for physical unit conversion)
  r_star_solar: 1.51      # Stellar radius (R☉)
  m_star_solar: 1.24      # Stellar mass (M☉)

# ========================================================================
# PROCESSING OPTIONS
# ========================================================================
options:
  verbose: true             # Print detailed progress
  save_intermediate: true   # Save intermediate results (calibrated images, etc.)
  plot_diagnostics: true    # Generate diagnostic plots
```

### Key Parameters to Adjust for Your Data

#### Must Update
1. **Paths**: Point to your data directories
2. **t0_guess**: Predicted transit center time for your observation
3. **Exposure times**: Match your FITS headers

#### Should Verify
4. **FWHM**: Check a few calibrated images, measure typical star widths
5. **Threshold**: Adjust until ~10-50 stars detected
6. **Aperture radius**: Should be 1-2× FWHM
7. **Target/reference indices**: Inspect detection plot, verify correct stars selected

#### May Need Tuning
8. **Limb darkening**: Look up coefficients for your star's Teff, log(g), [Fe/H], and filter
9. **Initial guesses** (rp, a, inc): Get from literature (exoplanetarchive.ipac.caltech.edu)

---

## Step-by-Step Workflow

### Step 1: Prepare Your Data

**Directory structure:**
```
my_observation/
├── bias/
│   ├── bias_001.fit
│   ├── bias_002.fit
│   └── ... (10+ frames recommended)
├── darks/
│   ├── dark_001.fit
│   ├── dark_002.fit
│   └── ... (10+ frames)
├── flats/
│   ├── flat_001.fit
│   ├── flat_002.fit
│   └── ... (10+ frames)
└── data/
    ├── target_001.fit
    ├── target_002.fit
    └── ... (20-100 frames)
```

**Requirements:**
- All frames same dimensions (e.g., 2048×2048)
- Bias: 0s exposure
- Darks: Same exposure as science frames (or scale will handle it)
- Flats: Uniform illumination (twilight sky or dome)
- Science: Continuous time series

### Step 2: Create Configuration

```bash
pytransit --create-config wasp75b_config.yaml
```

Edit `wasp75b_config.yaml` with your paths and parameters.

### Step 3: Run Full Pipeline

```bash
pytransit wasp75b_config.yaml
```

Or run interactively:

```python
from pytransitphotometry import TransitPipeline, PipelineConfig

# Load config
config = PipelineConfig.from_yaml('wasp75b_config.yaml')

# Create pipeline
pipeline = TransitPipeline(config)

# Run all stages
results = pipeline.run()

# Access results
print(f"Transit depth: {results['fit_result']['derived_params']['transit_depth_pct']:.3f}%")
```

### Step 4: Inspect Diagnostic Plots

Plots saved to `outputs/`:

1. **calibration_comparison.png**: Check if calibration improved image quality
2. **detected_sources.png**: Verify target and references correctly identified
3. **lightcurve_plot.png**: See raw light curve, check for outliers
4. **transit_fit.png**: Evaluate model fit quality

### Step 5: Validate Results

**Check fit quality:**
- **χ²_reduced** should be ~1.0 (±0.2)
  - If >> 1: Underestimated errors or poor model
  - If << 1: Overestimated errors
- **Residuals** should be randomly scattered (no patterns)

**Compare to literature:**
- Look up your target on [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- Check if your Rp/Rs, a/Rs, inc agree within ~2σ

**Physical sanity checks:**
- Transit depth: 0.1-5% typical (larger = bigger planet or smaller star)
- Impact parameter b < 1 for full transit
- Inclination: 80-90° for transiting systems

---

## Example Workflows

### Example 1: Full Pipeline (WASP-75b)

**Data:** Included in repository at `Group8.nosync/`

```python
from pytransitphotometry import TransitPipeline, PipelineConfig

# Load example config
config = PipelineConfig.from_yaml('examples/config_example.yaml')

# Run pipeline
pipeline = TransitPipeline(config)
results = pipeline.run()

# Print results
fit = results['fit_result']['fitted_params']
print(f"Rp/Rs = {fit['rp'][0]:.4f} ± {fit['rp'][1]:.4f}")
print(f"a/Rs = {fit['a'][0]:.2f} ± {fit['a'][1]:.2f}")
print(f"inc = {fit['inc'][0]:.2f} ± {fit['inc'][1]:.2f}°")

# Expected results (literature):
# Rp/Rs ≈ 0.1034 ± 0.0015
# a/Rs ≈ 7.17 ± 0.37
# inc ≈ 82.0° ± 0.3°
```

### Example 2: Stage-by-Stage Execution

```python
from pytransitphotometry import TransitPipeline, PipelineConfig

config = PipelineConfig.from_yaml('my_config.yaml')
pipeline = TransitPipeline(config)

# Stage 1: Calibration
print("Running calibration...")
pipeline.run_calibration()

# Inspect calibrated image
import matplotlib.pyplot as plt
plt.imshow(pipeline.calibrated_images[0], vmin=700, vmax=1400)
plt.colorbar()
plt.title("First Calibrated Frame")
plt.show()

# Stage 2: Detection
print("Detecting sources...")
pipeline.run_detection()
sources = pipeline.sources_list[0]
print(f"Detected {len(sources)} stars in first frame")

# Stage 3: Photometry
print("Measuring photometry...")
pipeline.run_photometry()
lc = pipeline.lightcurve
print(f"Built light curve: {len(lc['times'])} points")

# Stage 4: Detrending
print("Detrending...")
pipeline.run_detrending()
print(f"RMS scatter: {np.std(pipeline.detrended_lc['fluxes']):.6f}")

# Stage 5: Fitting
print("Fitting transit model...")
pipeline.run_transit_fit()
print(f"χ²_red = {pipeline.fit_result['reduced_chi_squared']:.2f}")

# Stage 6: Export
print("Exporting results...")
pipeline.export_results()
print("Done! Check outputs/ directory.")
```

### Example 3: Custom Photometry (No Transit Fitting)

```python
from pytransitphotometry import TransitPipeline, PipelineConfig

config = PipelineConfig.from_yaml('my_config.yaml')
pipeline = TransitPipeline(config)

# Run calibration, detection, photometry only
pipeline.run_calibration()
pipeline.run_detection()
pipeline.run_photometry()

# Extract light curve
lc = pipeline.lightcurve
times = lc['times']
fluxes = lc['fluxes']
errors = lc['errors']

# Custom analysis (e.g., periodogram search)
from scipy.signal import lombscargle
frequencies = np.linspace(0.1, 10, 10000)
power = lombscargle(times, fluxes - np.mean(fluxes), frequencies)

# Plot periodogram
plt.plot(frequencies, power)
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram')
plt.show()
```

### Example 4: Aperture Optimization

```python
from pytransitphotometry import TransitPipeline, PipelineConfig
import numpy as np

config = PipelineConfig.from_yaml('my_config.yaml')

# Enable aperture optimization
config.photometry.optimize_aperture = True
config.photometry.aperture_radii_test = list(range(3, 20))

pipeline = TransitPipeline(config)
pipeline.run_calibration()
pipeline.run_detection()

# Photometry will automatically find optimal radius
pipeline.run_photometry()

print(f"Optimal aperture radius: {pipeline.config.photometry.aperture_radius:.1f} px")
```

### Example 5: Batch Processing (Multiple Nights)

```python
from pytransitphotometry import TransitPipeline, PipelineConfig
import glob

config_files = glob.glob('configs/night*.yaml')

results_summary = []

for config_file in config_files:
    print(f"\nProcessing {config_file}...")
    
    config = PipelineConfig.from_yaml(config_file)
    pipeline = TransitPipeline(config)
    
    try:
        results = pipeline.run()
        
        fit = results['fit_result']['fitted_params']
        results_summary.append({
            'night': config_file,
            'rp': fit['rp'][0],
            'rp_err': fit['rp'][1],
            'chi2_red': results['fit_result']['reduced_chi_squared']
        })
    
    except Exception as e:
        print(f"Failed: {e}")
        continue

# Print summary
import pandas as pd
df = pd.DataFrame(results_summary)
print("\n=== Results Summary ===")
print(df)
print(f"\nWeighted mean Rp/Rs: {np.average(df['rp'], weights=1/df['rp_err']**2):.4f}")
```

---

## Output Files

### CSV Files

#### `lightcurve_raw.csv`
Raw differential photometry (target/references).

**Columns:**
- `time`: Observation time (MJD or HJD from FITS headers)
- `flux`: Differential flux ratio
- `error`: Flux uncertainty
- `airmass` (optional): If present in headers
- `centroid_x`, `centroid_y` (optional): Target centroid per frame

#### `lightcurve_detrended.csv`
After sigma clipping and optional linear detrending.

**Same columns as raw**, but with outliers removed.

### JSON Files

#### `fit_results.json`
Complete transit fit results.

**Structure:**
```json
{
  "fitted_params": {
    "rp": [0.1034, 0.0015],      // [value, uncertainty]
    "a": [7.17, 0.37],
    "inc": [82.0, 0.3],
    "baseline": [0.9998, 0.0002],
    "slope": [0.0001, 0.0001]
  },
  "t0": 2460235.752,
  "period": 2.4842,
  "chi_squared": 123.4,
  "reduced_chi_squared": 1.05,
  "derived_params": {
    "transit_depth_pct": 1.07,     // (Rp/Rs)² × 100
    "impact_parameter": 0.23,       // b = (a/Rs) × cos(inc)
    "transit_duration_hours": 2.8,
    "planet_radius_jupiter": 1.23,  // If r_star_solar provided
    "planet_radius_earth": 13.8
  }
}
```

#### `config_used.yaml`
Copy of configuration used for run (for reproducibility).

### PNG Files

#### `calibration_comparison.png`
Side-by-side: raw vs. calibrated first frame.

**What to check:**
- Calibrated image should have flatter background
- Stars should have similar brightness across field

#### `detected_sources.png`
Calibrated image with detected sources marked.

**What to check:**
- Target star (green X) correctly identified?
- Reference stars (red +) are bright, non-saturated?
- No false detections (cosmic rays, hot pixels)?

#### `lightcurve_plot.png`
Raw or detrended light curve with error bars.

**What to check:**
- Transit visible as dip?
- Outliers removed by sigma clipping?
- Scatter consistent with error bars?

#### `transit_fit.png`
Three panels:
1. Data + best-fit model
2. Residuals (data - model)
3. Parameter table

**What to check:**
- Model matches data?
- Residuals randomly scattered (no patterns)?
- χ²_red ≈ 1?
- Parameter uncertainties reasonable?

---

## Interpreting Results

### Transit Parameters

#### Radius Ratio (Rp/Rs)
**Physical meaning:** Ratio of planet radius to star radius

**Typical values:**
- Hot Jupiters: 0.08-0.15 (8-15% of star's radius)
- Hot Neptunes: 0.03-0.06
- Super-Earths: 0.01-0.03

**Transit depth:** $δ = (R_p/R_s)^2$

**Example:**
```python
rp_rs = 0.1034 ± 0.0015
transit_depth = rp_rs**2 = 0.0107 = 1.07%
```

#### Scaled Semi-Major Axis (a/Rs)
**Physical meaning:** Orbital distance in units of stellar radius

**Typical values:**
- Hot Jupiters: 5-10
- Temperate planets: >50

**Related to period by Kepler's 3rd law:**
$$a/R_s = \left(\frac{GM_\star P^2}{4\pi^2 R_\star^3}\right)^{1/3}$$

**Constrains stellar density:**
$$\rho_\star = \frac{3\pi}{GP^2} (a/R_s)^3$$

#### Inclination (inc)
**Physical meaning:** Orbital tilt (90° = edge-on, 0° = face-on)

**Typical values:**
- Transiting planets: 80-90° (must be nearly edge-on to see transit)

**Impact parameter:**
$$b = (a/R_s) × \cos(i)$$

- $b = 0$: Central transit (planet crosses star's center)
- $0 < b < 1$: Partial transit
- $b > 1$: No transit (grazing or non-transiting)

### Goodness of Fit

#### Reduced χ² (χ²_red)
**Formula:**
$$\chi^2_{red} = \frac{1}{N - n_{params}} \sum \frac{(data - model)^2}{\sigma^2}$$

**Interpretation:**
- **χ²_red ≈ 1**: Good fit (model explains data given uncertainties)
- **χ²_red >> 1**: Poor fit or underestimated errors
- **χ²_red << 1**: Overestimated errors (too conservative)

**Acceptable range:** 0.8-1.5

#### Residual Analysis
**Random scatter:** Good (noise-dominated)  
**Patterns/structure:** Bad (systematic errors)

**Red flags:**
- U-shaped residuals: Wrong limb darkening
- Slope in residuals: Incorrect detrending
- Outliers: Non-photometric conditions or bad frames

---

## Troubleshooting

### Issue: "No sources detected"

**Symptoms:**
```
RuntimeError: No sources detected with threshold=10000.0, fwhm=5.0
```

**Solutions:**
1. Lower `detection.threshold`:
   ```yaml
   detection:
     threshold: 5000.0  # Try 50% of original
   ```

2. Check calibration:
   ```python
   plt.imshow(pipeline.calibrated_images[0])
   plt.colorbar()
   plt.show()
   # Are stars visible? Background flat?
   ```

3. Adjust FWHM to match seeing:
   ```yaml
   detection:
     fwhm: 7.0  # Increase if seeing poor
   ```

4. Try sigma-based threshold:
   ```yaml
   detection:
     threshold: 5.0  # 5-sigma
     threshold_type: "sigma"
   ```

---

### Issue: "Target or reference stars not detected in frame X"

**Symptoms:**
```
⚠ Frame 23/100: Missing stars (only 2 detected), skipping
```

**Solutions:**
1. Check that stars stay in frame (tracking OK?)
2. Verify reference indices valid:
   ```python
   # Inspect first frame detections
   sources = pipeline.sources_list[0]
   print(sources['xcentroid', 'ycentroid', 'flux'][:10])
   ```

3. Reduce detection threshold for problematic frames
4. Consider excluding bad frames manually

---

### Issue: "Transit fit failed to converge"

**Symptoms:**
```
RuntimeError: Transit fit failed: Optimal parameters not found
```

**Solutions:**
1. Check t0_guess is correct:
   ```python
   # Plot light curve, identify transit center visually
   plt.plot(times, fluxes, 'o')
   plt.axvline(config.transit_model.t0_guess, color='r', label='t0_guess')
   plt.legend()
   plt.show()
   ```

2. Widen parameter bounds:
   ```yaml
   transit_model:
     rp_bounds: [0.05, 0.20]  # Wider
     a_bounds_factor: 0.5      # ±50% instead of ±20%
   ```

3. Improve initial guesses (get from literature)

4. Check for transit signal:
   ```python
   plt.plot(times, fluxes, 'o')
   plt.xlabel('Time (MJD)')
   plt.ylabel('Relative Flux')
   plt.title('Is there a dip?')
   plt.show()
   ```

---

### Issue: "Large χ²_red (> 2)"

**Symptoms:** Fit doesn't match data well

**Causes & solutions:**

1. **Underestimated errors:**
   - Check if error bars too small
   - Photon noise only? (should include systematics)

2. **Wrong transit parameters:**
   - Verify period, limb darkening from literature
   - Try floating t0 if you have full transit

3. **Systematics not removed:**
   ```yaml
   detrending:
     remove_linear_trend: true
     test_airmass: true
   ```
   
   Check airmass correlation output. If |r| > 0.3, consider:
   - Different reference stars (similar color to target)
   - Exclude non-photometric portions

4. **Incomplete transit coverage:**
   - Can only fit visible portion accurately
   - Parameter uncertainties larger if missing ingress/egress

---

### Issue: "Flux ratios > 1.1 or < 0.9"

**Symptoms:** Light curve baseline far from 1.0

**Causes:**
1. **Wrong star selected as target:**
   ```python
   # Check detection plot
   from pytransitphotometry.visualization import plot_detected_sources
   plot_detected_sources(pipeline.calibrated_images[0], 
                          pipeline.sources_list[0],
                          target_index=2,
                          reference_indices=[0, 1])
   ```

2. **Variable reference stars:**
   - Choose different references (avoid variables)
   - Check reference fluxes:
   ```python
   plt.plot(times, pipeline.lightcurve['reference_fluxes'])
   plt.xlabel('Time')
   plt.ylabel('Reference Flux')
   plt.title('Are references constant?')
   plt.show()
   ```

3. **Saturation:**
   - Target or references saturated?
   - Check peak pixel values: should be < 80% of CCD well depth

---

### Issue: "Negative fluxes after calibration"

**Symptoms:** `measure_flux()` returns negative values

**Causes:**
1. **Over-subtraction of bias/dark:**
   - Check exposure times in config match FITS headers
   - Verify darks scaled correctly

2. **Bad calibration frames:**
   - Inspect master bias/dark/flat:
   ```python
   plt.figure(figsize=(15, 5))
   plt.subplot(131); plt.imshow(pipeline.calibration_frames.master_bias); plt.title('Bias')
   plt.subplot(132); plt.imshow(pipeline.calibration_frames.master_dark); plt.title('Dark')
   plt.subplot(133); plt.imshow(pipeline.calibration_frames.get_normalized_flat()); plt.title('Flat')
   plt.show()
   ```

3. **Sky background too high:**
   - Adjust background annulus (avoid stars):
   ```yaml
   photometry:
     annulus_inner: 50.0  # Increase
     annulus_outer: 70.0
   ```

---

## Performance Tips

### Memory Usage

**Typical RAM requirements:**
- 2048×2048 images, 100 frames, float32: ~1.6 GB
- 1024×1024 images, 100 frames, float32: ~400 MB

**Reduce memory:**
```python
# Load subset of frames
config = PipelineConfig.from_yaml('config.yaml')

# Modify after loading
from pytransitphotometry.io import load_fits_files
data, headers = load_fits_files(config.paths.data_dir, 
                                  config.paths.data_pattern)
data = data[::2]  # Every other frame
headers = headers[::2]
```

### Processing Speed

**Bottlenecks:**
1. **FITS loading:** ~0.1-1s per file
2. **Source detection:** ~1-3s per frame
3. **Transit fitting:** ~5-30s (depends on convergence)

**Speed up:**
- Use SSD for data storage
- Reduce number of test radii for aperture optimization
- Disable plots: `plot_diagnostics: false`

---

## Next Steps

- **Understand the codebase:** Read [architecture.md](architecture.md)
- **Explore API:** See [api.md](api.md)
- **Extend the library:** Check [development.md](development.md)
- **Report issues:** Open GitHub issue with config file and error message

**Happy transiting! 🌟🪐**
