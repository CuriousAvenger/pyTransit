---
title: 'pyTransitPhotometry: An Automated, Config-Driven Pipeline for Ground-Based Exoplanet Transit Photometry'
tags:
  - Python
  - astronomy
  - exoplanets
  - transit photometry
  - CCD reduction
  - PSF fitting
  - time-series analysis
authors:
  - name: Sai Praneth
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 21 April 2026
bibliography: paper.bib
---

# Summary

`pyTransitPhotometry` is a Python library that automates the complete workflow for ground-based exoplanet transit photometry: from raw CCD frames to fitted planetary parameters. A single YAML configuration file governs every tunable parameter, making reductions fully reproducible and auditable. The library targets graduate students and researchers who perform ground-based transit observations and need a documented, tested, version-controlled pipeline to replace ad hoc notebook workflows.

The pipeline sequentially applies: (1) CCD calibration (bias, dark, flat), (2) source detection via `DAOStarFinder` [@stetson1987], (3) empirical PSF fitting photometry with 2D background modelling, (4) differential photometry against a weighted reference-star ensemble, (5) robust anomaly rejection and atmospheric detrending, and (6) non-linear transit model fitting using the analytic model of @mandel2002 as implemented in `batman` [@kreidberg2015]. Each stage produces diagnostic plots and structured output files.

# Statement of Need

Ground-based transit photometry is routinely affected by three classes of systematic error that degrade the precision of recovered planetary parameters.

**Flux dilution from blended background stars.** Classical circular aperture photometry integrates light from all sources within the aperture, including faint companions spatially unresolved in typical ground-based seeing (FWHM 3–6 arcsec). The extra flux dilutes the transit depth by a factor $(1 + f_\mathrm{contam})^{-1}$, systematically underestimating $R_p/R_\star$ and inflating the reduced chi-squared of the fit. In our WASP-75b validation case, a 6-pixel aperture produced $\chi^2_\mathrm{red} = 2.52$ and a transit depth 8% shallower than the published value [@hellier2014].

**Tracking drifts and atmospheric scintillation spikes.** Ground-based photometric series contain sharp flux excursions from guide-star lock failures, scintillation events, and passing thin cloud. Standard global sigma clipping either retains instrumental artefacts (threshold too high) or clips genuine transit signal at ingress/egress (threshold too low).

**Extinction bias from airmass gradients.** The nightly airmass variation introduces a colour-dependent flux trend. Ordinary least-squares fitting of this trend is biased when outliers coexist with the signal, producing a systematic offset in the fitted transit depth.

These three issues together motivate an integrated, automated pipeline that addresses them simultaneously within a single reproducible framework. The target audience is any researcher performing ground-based follow-up of transit candidates from TESS or other all-sky surveys, who needs results that are reproducible between independent reduction runs.

# State of the Field

Several existing tools address overlapping subsets of this problem. `AstroImageJ` [@collins2017] provides an interactive graphical interface for aperture photometry widely used in citizen-science transit campaigns, but requires manual interaction for each dataset and does not expose a Python API or implement PSF fitting. `EXOTIC` [@zellem2020] offers an automated end-to-end pipeline targeting citizen-science observers, but is limited to aperture photometry and global sigma clipping. `lightkurve` [@lightkurve2018] is a mature toolkit for space-based time-series data from Kepler and TESS but does not handle raw CCD calibration from ground-based observatories. The `ccdproc` package [@matt_craig2017] handles CCD calibration frames competently but implements neither photometry nor light curve detrending.

`AstroImageJ` and `EXOTIC` are the closest functional analogues; neither exposes a scriptable Python API suitable for programmatic, reproducible research workflows, and neither implements PSF fitting photometry or rolling-window anomaly rejection. `pyTransitPhotometry` fills this gap. The decision to build a standalone library rather than extend one of these tools reflects the absence of a PSF-fitting and robust-detrending capability in any existing ground-based transit pipeline that also provides a Python API and a versioned configuration system.

# Software Design

The central design decision is **separation of configuration from code**. All tunable parameters — file paths, aperture sizes, PSF construction settings, detrending thresholds, transit priors — live in a versioned YAML file parsed into a hierarchy of frozen dataclasses (`PipelineConfig`, `PhotometryConfig`, `DetrendingConfig`). Re-running a reduction on different data, or replicating a published result, requires only swapping the YAML file.

The pipeline follows a **stage-gate architecture**: each of the six stages is implemented as a stateless function with well-typed inputs and structured outputs. The `TransitPipeline` orchestrator sequences stages without owning algorithmic logic; every algorithm lives in its own module (`calibration.py`, `photometry.py`, `detrending.py`, `models.py`). This makes it straightforward to invoke any single stage from a notebook and to swap implementations — for example enabling PSF mode by setting `method: "psf"` in the YAML.

**Photometry method selection** reflects a deliberate trade-off. PSF fitting decomposes blended sources quantitatively at the cost of increased computation; aperture photometry remains available for non-crowded fields. The ePSF is built once per night from isolated bright stars and reused across all science frames to amortise the construction cost.

**Anomaly rejection** is similarly configurable. The rolling-window MAD filter operates locally, preserving ingress/egress points that a global threshold would clip. The Isolation Forest mode handles time series with complex, non-Gaussian noise that a single scale parameter cannot describe. Both are exposed as a single YAML field (`outlier_method`) to keep the interface simple.

**Error propagation** follows the standard CCD noise equation throughout:
$$\sigma_F = \sqrt{F/g + N_\mathrm{pix}(\sigma_\mathrm{sky}^2 + (R/g)^2)}$$
where $F$ is the background-subtracted flux, $g$ is the CCD gain, $N_\mathrm{pix}$ is the aperture area in pixels, $\sigma_\mathrm{sky}$ is the per-pixel sky noise, and $R$ is the read noise. These weights propagate into the differential photometry, the Huber regression, and the transit fit.

## Pipeline Stages

**Stage 1 — CCD Calibration.** Raw science frames are corrected by
$$I_\mathrm{cal} = \frac{I_\mathrm{raw} - I_\mathrm{bias} - I_\mathrm{dark}(t_\mathrm{sci}/t_\mathrm{dark})}{I_\mathrm{flat,norm}}$$
Master calibration frames are built by median-combining individual exposures with optional sigma clipping.

**Stage 2 — Source Detection.** `DAOStarFinder` [@bradley2023] detects point sources above a configurable threshold with sharpness and roundness quality filters.

**Stage 3 — PSF Fitting and 2D Background.** An oversampled ePSF is constructed from isolated bright stars using `EPSFBuilder` [@anderson2000] and fitted to all detected sources via `PSFPhotometry`. Sky background is independently estimated using either a tiled `Background2D` or a global `Polynomial2D` fit, then subtracted before fitting.

**Stage 4 — Differential Photometry.** Inverse-variance weighted reference-star ensemble photometry cancels common-mode atmospheric and instrumental variations.

**Stage 5 — Detrending.** Rolling MAD or Isolation Forest outlier rejection is followed by Huber-robust airmass regression:
$$F_\mathrm{corr}(t) = \frac{F_\mathrm{obs}(t)}{\hat{F}_\mathrm{Huber}(X(t))\,/\,\langle\hat{F}_\mathrm{Huber}\rangle}$$
where the Huber regressor [@huber1964] minimises a combined $L_1$/$L_2$ loss robust to co-occurring outliers.

**Stage 6 — Transit Fitting.** The @mandel2002 model with quadratic limb darkening is fitted by `scipy.optimize.curve_fit`, optimising $R_p/R_\star$, $a/R_\star$, and orbital inclination with physically motivated YAML-specified bounds.

# Research Impact Statement

The library was developed to support ground-based photometric follow-up of exoplanet transit candidates identified by TESS. Applied to a *R*-band transit observation of WASP-75b [@hellier2014], the pipeline recovers a transit depth $\delta = (R_p/R_\star)^2 = 1.05 \pm 0.04\%$, consistent with the published value of $1.07 \pm 0.02\%$. The reduced chi-squared improves from $\chi^2_\mathrm{red} = 2.52$ (aperture photometry, OLS detrending) to $\chi^2_\mathrm{red} = 1.08$ (PSF photometry, Huber detrending), providing quantitative evidence that the systematic-error mitigation strategies produce scientifically meaningful improvements. The WASP-75b configuration file and reduction summary are included in `examples/` as a reproducible benchmark for reviewers and users.

# Automated Testing

The library ships with a `pytest` test suite in `tests/test_pipeline.py` (36 tests) that runs automatically via GitHub Actions on Python 3.9–3.12 on both Linux and macOS:

- **Calibration unit tests**: dark-current linear scaling, normalised flat production, and full image calibration against analytically known values.
- **2D background tests**: mesh-based and polynomial estimators verified for shape and accuracy on synthetic images.
- **Photometry tests**: flux positivity, error positivity, and SNR verified for a synthetic Gaussian star.
- **Detrending tests**: each outlier rejection method (sigma clipping, rolling MAD, Isolation Forest) verified to reject synthetic spikes; Huber regression verified to reduce airmass correlation.
- **Transit injection–recovery tests**: a `batman` transit ($R_p/R_\star = 0.103$, $a/R_\star = 7.17$, $i = 82°$) injected into a 500 ppm noise series; the NLS optimiser must recover all three parameters within $3\sigma$, including from an initial guess offset by 10–20%.

# AI Usage Disclosure

Portions of the code scaffolding, test suite, and documentation were developed with assistance from GitHub Copilot (powered by Claude Sonnet 4.6, Anthropic). Specifically, AI assistance was used for: (1) initial drafts of docstrings and type annotations across all modules; (2) test case structure and fixture code in `tests/test_pipeline.py`; (3) portions of the documentation in `docs/`; and (4) draft text for sections of this paper. All AI-generated outputs were reviewed, validated, and substantially revised by the author. All algorithmic logic, design decisions, scientific correctness checks (including the WASP-75b validation), and the final content of this paper reflect the author's own judgment. The author accepts full responsibility for the accuracy and integrity of all submitted materials.

# Acknowledgements

The author thanks the developers of `astropy` [@astropy2022], `photutils` [@bradley2023], `batman` [@kreidberg2015], and `scikit-learn` [@pedregosa2011], whose foundational software this library builds upon.

# References
