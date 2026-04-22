# pyTransitPhotometry

[![CI](https://github.com/CuriousAvenger/pyTransitPhotometry/actions/workflows/ci.yml/badge.svg)](https://github.com/CuriousAvenger/pyTransitPhotometry/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://joss.theoj.org/papers/placeholder/status.svg)](https://joss.theoj.org)

An automated, config-driven Python library for ground-based exoplanet transit photometry. From raw FITS frames to fitted planetary parameters in a single reproducible pipeline.

---

## Overview

Ground-based transit photometry is affected by three systematic error sources that degrade the precision of recovered planetary parameters: flux dilution from blended background stars, atmospheric scintillation and tracking drift spikes, and extinction bias from airmass gradients. `pyTransitPhotometry` addresses all three within a single, documented, tested pipeline governed by a versioned YAML configuration file.

**Pipeline stages:**

```
Raw FITS frames
    ↓  [1] CCD Calibration       — bias, dark (scaled), flat
    ↓  [2] Source Detection      — DAOStarFinder with quality filters
    ↓  [3] PSF Photometry        — ePSF fitting + 2D background subtraction
    ↓  [4] Differential LC       — inverse-variance weighted reference ensemble
    ↓  [5] Detrending            — rolling MAD / Isolation Forest + Huber airmass
    ↓  [6] Transit Fitting       — batman + scipy NLS → Rp/Rs, a/Rs, inclination
```

## Features

| Feature | Details |
|---------|---------|
| **Photometry** | Aperture *or* empirical PSF fitting (`EPSFBuilder` + `PSFPhotometry`) |
| **Background** | Mesh-based `Background2D` or third-order `Polynomial2D` |
| **Outlier rejection** | Rolling-window MAD, Isolation Forest, or global sigma-clip |
| **Detrending** | Huber-robust airmass regression (replaces OLS) |
| **Transit model** | `batman` analytic model with quadratic limb darkening |
| **Config** | Single YAML file → reproducible, version-controlled reductions |
| **Tests** | 36 `pytest` unit tests; CI on Python 3.9–3.12, Linux + macOS |

## Installation

### From source (recommended during active development)

```bash
git clone https://github.com/CuriousAvenger/pyTransitPhotometry.git
cd pyTransitPhotometry
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `scipy`, `astropy`, `photutils>=1.8`, `batman-package`, `matplotlib`, `pyyaml`, `pandas`, `scikit-learn`.

### Conda environment

```bash
conda create -n pytransit python=3.11
conda activate pytransit
pip install -e .
```

## Quick Start

### 1. Copy and edit the example configuration

```bash
cp examples/config_example.yaml my_transit.yaml
# Edit paths, star indices, transit priors …
```

### 2. Run the complete pipeline

```python
from pyTransitPhotometry import TransitPipeline, PipelineConfig

config = PipelineConfig.from_yaml("my_transit.yaml")
pipeline = TransitPipeline(config)
results = pipeline.run()

rp, rp_err = results["fit_result"]["fitted_params"]["rp"]
print(f"Rp/Rs = {rp:.4f} ± {rp_err:.4f}")
```

### 3. Run individual stages

```python
pipeline = TransitPipeline(config)
pipeline.run_calibration()
pipeline.run_detection()
pipeline.run_photometry()
pipeline.run_detrending()
pipeline.run_transit_fit()
pipeline.export_results()
```

### 4. Command-line interface

```bash
pytransit --config my_transit.yaml --output ./results/
```

## Configuration Reference

A full annotated template is provided in [`examples/config_example.yaml`](examples/config_example.yaml). Key sections:

```yaml
photometry:
  method: "psf"                 # "aperture" or "psf"
  background_method: "background2d"  # "background2d" or "polynomial"
  aperture_radius: 6.0
  target_star_index: 2
  reference_star_indices: [0, 1]

detrending:
  outlier_method: "rolling_mad" # "sigma_clip", "rolling_mad", "isolation_forest"
  window_size: 20
  mad_sigma: 3.5
  airmass_regression: "huber"

transit_model:
  period: 2.4842                # days — fixed
  t0_guess: 2460235.752         # predicted mid-transit (MJD)
  limb_dark_u1: 0.40
  limb_dark_u2: 0.26
```

## Validation

The pipeline has been validated on a ground-based *R*-band transit of WASP-75b
[@hellier2014]:

| Metric | Aperture + OLS | PSF + Huber (this work) |
|--------|---------------|------------------------|
| χ²_red | 2.52 | **1.08** |
| Rp/Rs | 0.096 ± 0.005 | **0.1025 ± 0.002** |
| Literature Rp/Rs | 0.1034 | 0.1034 |

The reduction configuration and output summary are in [`examples/`](examples/).

## Documentation

| Document | Description |
|----------|-------------|
| [docs/installation.md](docs/installation.md) | Detailed installation and environment setup |
| [docs/usage.md](docs/usage.md) | Step-by-step workflow guide |
| [docs/api.md](docs/api.md) | Full API reference |
| [docs/architecture.md](docs/architecture.md) | Design decisions and module structure |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [examples/tutorial.ipynb](examples/tutorial.ipynb) | Jupyter tutorial |

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run the full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=pyTransitPhotometry --cov-report=term-missing
```

All 36 tests cover: CCD calibration, 2D background estimation, aperture photometry, sigma-clipping, rolling MAD, Isolation Forest, Huber airmass detrending, and transit parameter injection–recovery.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. In brief:

1. Fork the repository and create a feature branch from `main`.
2. Add tests for any new functionality.
3. Ensure `pytest tests/` passes and `black`/`flake8` are clean.
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Open a pull request describing the motivation and changes.

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/CuriousAvenger/pyTransitPhotometry/issues). Use the provided templates.

## Citation

If you use `pyTransitPhotometry` in your research, please cite the JOSS paper (pending):

```bibtex
@article{praneth2026,
  author  = {Praneth, Sai},
  title   = {pyTransitPhotometry: An Automated, Config-Driven Pipeline
             for Ground-Based Exoplanet Transit Photometry},
  journal = {Journal of Open Source Software},
  year    = {2026},
  doi     = {TBD}
}
```

## License

`pyTransitPhotometry` is released under the [MIT License](LICENSE).

