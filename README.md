# pyTransitPhotometry

A professional, modular Python library for exoplanet transit photometry analysis. Designed for graduate students, astronomers, and reproducibility-focused researchers.

## Features

- **Complete CCD calibration**: Bias, dark, and flat field correction with exposure time scaling
- **Automated star detection**: Using photutils DAOStarFinder with quality filtering
- **Aperture photometry**: With SNR optimization and error propagation
- **Differential photometry**: Weighted reference star ensemble for systematic error removal
- **Detrending**: Sigma clipping, airmass correlation testing, linear trend removal
- **Transit modeling**: Batman-based transit fitting with parameter uncertainties
- **Publication-quality plots**: Diagnostic visualizations for every pipeline stage

## Scientific Workflow

```
Raw FITS frames
    ↓
[1] CCD Calibration (bias, dark, flat)
    ↓
[2] Star Detection (DAOStarFinder)
    ↓
[3] Aperture Photometry (background-subtracted flux extraction)
    ↓
[4] Light Curve Construction (differential photometry)
    ↓
[5] Detrending (outlier removal, airmass test)
    ↓
[6] Transit Model Fitting (batman + scipy.optimize)
    ↓
Results: Rp/Rs, a/Rs, inclination, transit depth, impact parameter
```

## Installation

### From PyPI (recommended):
```bash
pip install pytransitphotometry
```

### From source:
```bash
git clone <repository-url>
cd pytransitphotometry
pip install -e .
```

### Dependencies:
- numpy
- scipy
- astropy
- photutils
- batman-package
- matplotlib
- pyyaml
- pandas

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Create a configuration file

```python
from pytransitphotometry.config import create_example_config

# Creates config_example.yaml
create_example_config('my_config.yaml')
```

Edit `my_config.yaml` to point to your data directories and set target parameters.

### 2. Run the complete pipeline

```python
from pytransitphotometry import TransitPipeline, PipelineConfig

# Load configuration
config = PipelineConfig.from_yaml('my_config.yaml')

# Create and run pipeline
pipeline = TransitPipeline(config)
results = pipeline.run()

# Access results
print(f"Transit depth: {results['fit_result']['derived_params']['transit_depth_pct']}")
```

### 3. Or run individual stages

```python
pipeline = TransitPipeline(config)

# Run one stage at a time
pipeline.run_calibration()
pipeline.run_detection()
pipeline.run_photometry()
pipeline.run_detrending()
pipeline.run_transit_fit()
pipeline.export_results()
```

## Configuration

Configuration is managed via YAML files for reproducibility. Key sections:

### Paths
```yaml
paths:
  data_dir: "./Group8.nosync/data"
  bias_dir: "./Group8.nosync/bias"
  dark_dir: "./Group8.nosync/darks"
  flat_dir: "./Group8.nosync/flats"
  output_dir: "./outputs"
```

### Photometry
```yaml
photometry:
  aperture_radius: 6.0          # pixels
  annulus_inner: 40.0           # background annulus
  annulus_outer: 60.0
  target_star_index: 2          # 0-indexed, sorted by brightness
  reference_star_indices: [0, 1]  # reference stars for differential photometry
```

### Transit Model
```yaml
transit_model:
  period: 2.4842                # orbital period (days, fixed)
  t0_guess: 2460235.752         # predicted transit center (MJD)
  fix_t0: true                  # fix t0 during fit?
  limb_dark_u1: 0.40            # quadratic limb darkening
  limb_dark_u2: 0.26
  rp_guess: 0.103               # initial Rp/Rs
  a_guess: 7.17                 # initial a/Rs
  inc_guess: 82.0               # initial inclination (deg)
  r_star_solar: 1.51            # for physical units (optional)
  m_star_solar: 1.24
```

See `config_example.yaml` for complete options.

## Output Files

The pipeline generates:

- `lightcurve_raw.csv`: Raw differential photometry
- `lightcurve_detrended.csv`: Cleaned light curve after sigma clipping
- `fit_results.json`: Transit parameters with uncertainties
- `config_used.yaml`: Configuration used for reproducibility

## Scientific Notes

### CCD Calibration
Implements standard CCD reduction:
```
calibrated = (raw - bias - dark×scale) / flat_normalized
```

Dark current is scaled linearly with exposure time. Flat fields are normalized to unity mean.

### Differential Photometry
Divides target flux by weighted reference ensemble:
```
ratio = F_target / Σ(w_i × F_ref_i) / Σ(w_i)
```
where weights `w_i = 1/σ_i²` (inverse variance weighting).

This removes common systematic effects (airmass, clouds, transparency variations).

### Aperture Selection
Aperture radius should be 1-2× FWHM. The pipeline can auto-optimize by maximizing SNR:
```
SNR = Signal / √(Signal×gain + N_pixels×σ_background²)
```

### Transit Depth
Measured directly from photometry:
```
δ = (Rp/Rs)² × 100%
```

Impact parameter:
```
b = (a/Rs) × cos(i)
```

Stellar density (from Kepler's 3rd law, photometry-only):
```
ρ★ = (3π)/(GP²) × (a/Rs)³
```

## Assumptions & Limitations

### Assumptions
1. **Circular orbits**: Default eccentricity = 0 (changeable in config)
2. **Fixed orbital period**: Must be provided (from literature or previous analysis)
3. **Known limb darkening**: Coefficients from stellar atmosphere models
4. **Non-variable references**: Reference stars assumed constant
5. **Photometric conditions**: Best results require clear skies

### Limitations
1. **Single-band photometry**: No color information
2. **No deblending**: Assumes well-separated stars
3. **Limited airmass correction**: Only linear trend removal (differential photometry should handle most)
4. **Requires full transit coverage**: For accurate Rp/Rs measurement
5. **No systematics modeling**: Beyond linear trends (no GP, no pixel-level corrections)

### When to use this pipeline
✅ Ground-based CCD transit observations  
✅ Well-separated target and bright references  
✅ Standard filters (BVRI, griz)  
✅ Full or near-full transit coverage  
✅ Learning transit photometry methods

### When NOT to use
❌ Space-based data (use specialized pipelines: lightkurve, TESS/Kepler tools)  
❌ Crowded fields (needs PSF photometry)  
❌ Grazing transits (specialized modeling required)  
❌ High-precision differential spectroscopy  
❌ Production pipeline for large surveys (this is for learning/small batches)

## Validation

Test the pipeline on WASP-75b (included example):
- **Expected Rp/Rs**: 0.1034 ± 0.0015
- **Expected a/Rs**: 7.17 ± 0.37
- **Expected inclination**: 82.0° ± 0.3°
- **Period**: 2.4842 days (fixed)

Compare your results with [ExoFOP](https://exofop.ipac.caltech.edu/) or [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).

## Troubleshooting

### No sources detected
- Lower `detection.threshold`
- Check image quality (are stars visible?)
- Verify calibration worked (`plot_calibration_comparison()`)

### Poor transit fit
- Check reference star selection (avoid variables, use bright stars)
- Ensure full transit coverage
- Verify t0_guess is close to actual transit time
- Adjust parameter bounds if hitting limits

### Large residuals
- Test for airmass correlation (`detrending.test_airmass = true`)
- Check for clouds/non-photometric conditions
- Consider excluding bad frames manually

### Negative fluxes
- Check calibration: bias/dark may be over-subtracted
- Verify exposure times are correct
- Try different background annulus (avoid stars)

## Contributing

Contributions welcome! Areas for enhancement:
- Additional detrending methods (Gaussian processes, polynomial)
- Multi-band photometry support
- Automatic period finding
- Additional transit models

## License

MIT License

## Support

For issues or questions:
- Open a GitHub issue
- Check documentation in `docs/`
- See `config_example.yaml` for configuration help

