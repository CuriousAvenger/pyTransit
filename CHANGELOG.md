# Changelog

All notable changes to `pyTransitPhotometry` are documented here.

This project follows [Semantic Versioning](https://semver.org/) and the format from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [1.1.0] – 2026-04-21

### Added

- **PSF fitting photometry** (`pyTransitPhotometry.photometry`): empirical PSF construction via `photutils.psf.EPSFBuilder` and simultaneous source fitting via `PSFPhotometry`, replacing classical circular aperture photometry as the default for crowded fields.
- **2D background estimation** (`estimate_2d_background`): mesh-based `Background2D` and third-order `Polynomial2D` methods, selectable via `config.photometry.background_method`.
- **Rolling-window MAD outlier filter** (`rolling_mad_filter`): local median absolute deviation rejection that protects transit ingress/egress from aggressive global clipping.
- **Isolation Forest anomaly detector** (`isolation_forest_filter`): scikit-learn `IsolationForest` trained on time/flux/gradient feature space for non-Gaussian noise patterns.
- **Huber airmass detrending** (`huber_airmass_detrend`): robust $L_1$/$L_2$ airmass regression replacing ordinary least squares, resistant to outlier-driven extinction bias.
- **`detrend_lightcurve_advanced`** orchestration function combining all new detrending methods.
- Extended `PhotometryConfig` with PSF and background fields; extended `DetrendingConfig` with outlier method and Huber epsilon fields.
- Full `pytest` test suite in `tests/test_pipeline.py` (36 tests: calibration, 2D background, photometry, detrending, transit injection–recovery).
- `CONTRIBUTING.md`, `LICENSE`, `pyproject.toml`, and GitHub Actions CI workflow.

### Changed

- `pipeline.py`: `run_photometry()` now branches on `config.photometry.method` (`"aperture"` or `"psf"`); `run_detrending()` delegates to `detrend_lightcurve_advanced`.
- `config_example.yaml`: updated schema for new photometry and detrending fields.
- `requirements.txt`: pinned `photutils>=1.8.0`; added `scikit-learn>=1.1.0`.

### Fixed

- `measure_flux`: `calc_total_error` now receives a 2D `bkg_error` array (required by photutils ≥ 3.0).
- `refine_centroid`: now guards against NaN or out-of-bounds divergence on featureless sky backgrounds.

---

## [1.0.0] – 2025-10-15

### Added

- Initial public release.
- Complete CCD calibration pipeline: bias, dark (exposure-time scaled), flat field correction.
- `DAOStarFinder`-based source detection with quality filters.
- Aperture photometry with background annulus and CCD noise error propagation.
- Differential photometry via inverse-variance weighted reference ensemble.
- Sigma-clip-based outlier rejection and airmass correlation testing.
- Transit model fitting using `batman` + `scipy.optimize.curve_fit`.
- YAML-driven `PipelineConfig` / `PhotometryConfig` / `DetrendingConfig` dataclasses.
- Publication-quality visualisation suite.
- CLI entry-point `pytransit`.
- `EXAMPLE.ipynb` Jupyter tutorial.

---

[Unreleased]: https://github.com/CuriousAvenger/pyTransitPhotometry/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/CuriousAvenger/pyTransitPhotometry/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/CuriousAvenger/pyTransitPhotometry/releases/tag/v1.0.0
