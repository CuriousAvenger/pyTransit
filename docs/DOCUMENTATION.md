# pyTransitPhotometry: Comprehensive Documentation

**Version:** 1.0.0  
**Authors:** Transit Photometry Team  
**Last Updated:** January 5, 2026

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Problem Domain](#problem-domain)
3. [Target Audience](#target-audience)
4. [Unique Design Aspects](#unique-design-aspects)
5. [Core Philosophy](#core-philosophy)
6. [Quick Navigation](#quick-navigation)

---

## High-Level Overview

### What is pyTransitPhotometry?

**pyTransitPhotometry** is a professional-grade Python library for analyzing exoplanet transit observations from ground-based telescopes. It implements a complete photometric pipeline that transforms raw CCD images into scientifically validated planetary parameters, specifically the radius ratio (Rp/Rs), orbital inclination, and scaled semi-major axis.

### What Problem Does It Solve?

Detecting and characterizing exoplanets via the transit method requires:

1. **Precision photometry** at the 0.1-1% level to detect small flux decreases
2. **Systematic error removal** from atmospheric, instrumental, and telescope-tracking effects
3. **Physical parameter extraction** from noisy, incomplete light curves
4. **Reproducibility** and validation against literature values

This library automates the entire workflow from raw FITS files to publication-quality results, eliminating the error-prone manual steps typical in bespoke analysis scripts.

### What Makes This Implementation Unique?

#### 1. **Modular Architecture with Clear Separation**
Unlike monolithic analysis scripts, pyTransitPhotometry separates concerns into:
- **Calibration**: Pure CCD physics with exposure-time-aware dark scaling
- **Detection**: Source finding with quality filtering
- **Photometry**: Background-subtracted flux with SNR optimization
- **Light Curve**: Differential photometry with weighted references
- **Detrending**: Statistical outlier removal and systematic trend analysis
- **Modeling**: Batman-based transit fitting with bounded optimization

Each module can be tested, validated, and replaced independently.

#### 2. **Configuration-Driven Execution**
All pipeline parameters live in a single YAML file, enabling:
- Version control of analysis choices
- Reproducible runs across machines
- Batch processing of multiple datasets
- Transparent documentation of methodology

#### 3. **Error Propagation Throughout**
Every stage computes and propagates uncertainties using proper CCD noise models:
```
σ² = Poisson(signal+sky) + Npix × σ_sky² + read_noise²
```
This ensures final parameter uncertainties reflect true measurement precision.

#### 4. **Built-in Validation and Diagnostics**
The pipeline includes:
- Automatic detection of insufficient calibration frames
- Airmass correlation tests for reference star quality
- Reduced χ² goodness-of-fit metrics
- Publication-quality diagnostic plots at every stage

#### 5. **Production-Ready Packaging**
Not just a collection of functions—it's a properly packaged library:
- PyPI-installable with dependency management
- Command-line interface for batch processing
- Comprehensive docstrings (NumPy style)
- Type hints for IDE support

---

## Problem Domain

### Scientific Context

**Exoplanet Transit Photometry** observes the slight dimming of a star when a planet passes in front of it. For a Jupiter-sized planet around a Sun-like star, the flux decrease is ~1%. For Earth-sized planets, it's ~0.01%.

**Key Challenges:**
1. **Low Signal-to-Noise**: Atmospheric turbulence (scintillation) introduces ~0.5-2% flux variations
2. **Systematics**: Airmass changes, telescope tracking errors, and cloud passages create correlated noise
3. **Calibration Precision**: Pixel-to-pixel sensitivity variations must be corrected to <0.1%
4. **Limited Data**: Ground-based observations often capture only partial transits due to daylight constraints

### Observational Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                     TELESCOPE SESSION                        │
├─────────────────────────────────────────────────────────────┤
│ Pre-science:                                                 │
│   - Bias frames (10x, 0s exposure)                          │
│   - Dark frames (10x, matching science exposure)            │
│   - Flat fields (10x, twilight sky)                         │
│                                                              │
│ Science:                                                     │
│   - Target observations (20-100 frames, 30-120s each)       │
│   - Continuous time series spanning transit window          │
└─────────────────────────────────────────────────────────────┘
```

The pipeline must handle:
- **Bias**: Additive electronic offset (10-50 ADU)
- **Dark current**: Thermally generated electrons (0.01-1 e-/px/s)
- **Flat field**: Pixel sensitivity variations (vignetting, dust shadows)
- **Science frames**: Target + reference stars + background sky

---

## Target Audience

### Primary Users

1. **Graduate Students in Astronomy**
   - Learning exoplanet characterization techniques
   - Conducting thesis research on transiting systems
   - Need: Robust defaults, clear diagnostics, educational documentation

2. **Observational Astronomers**
   - Analyzing data from university/research telescopes
   - Publishing peer-reviewed transit studies
   - Need: Publication-quality results, parameter uncertainties, reproducibility

3. **Educators and Workshop Instructors**
   - Teaching photometric data reduction
   - Running hands-on observing projects
   - Need: Example datasets, pedagogical clarity, step-by-step execution

### Secondary Users

4. **Professional Astronomers**
   - Validating literature results
   - Comparative analyses across multiple systems
   - Need: Extensibility, custom detrending, batch processing

5. **Amateur Astronomers**
   - Contributing to citizen science projects (e.g., AAVSO)
   - Analyzing data from backyard telescopes
   - Need: Straightforward installation, default parameters, minimal tuning

---

## Unique Design Aspects

### 1. Exposure-Time-Aware Dark Scaling

Many pipelines naively subtract dark frames without considering exposure time differences. This implementation correctly models dark current as:

```
dark(t) = bias + dark_rate × t
```

The calibration module scales darks independently for flats (short exposure) and science frames (long exposure), preventing systematic over/under-subtraction.

### 2. Weighted Ensemble Reference Stars

Instead of using a single comparison star, the library computes a weighted mean of multiple references:

```
R_ensemble = Σ(w_i × F_i) / Σ(w_i)    where w_i = 1/σ_i²
```

This:
- Maximizes SNR by using all available signal
- Reduces impact if one reference is variable
- Properly propagates uncertainties

### 3. Airmass Correlation Diagnostics

The detrending module explicitly tests for residual airmass correlation:

```
r = corr(airmass, flux_ratio)
```

If |r| > 0.3, it suggests:
- Color mismatch between target and references
- Non-photometric conditions
- Poor reference star selection

This catches systematic errors that would bias transit parameters.

### 4. Bounded Optimization with Physical Constraints

The transit fitter uses `scipy.optimize.curve_fit` with hard bounds:

```yaml
rp: [0.085, 0.120]     # Radius ratio must be physical
a: ±20% of guess        # Semi-major axis within orbital mechanics
inc: ±6° of guess       # Inclination constrained by impact parameter
```

This prevents optimizer divergence to unphysical solutions (e.g., Rp/Rs > 1).

### 5. Centroids Refinement for Precision

Even though DAOStarFinder provides initial centroids, every photometry measurement refines positions using 2D Gaussian fitting. This achieves 0.01-0.1 pixel precision, critical for:
- Aperture placement accuracy
- Minimizing flux contamination from nearby stars
- Tracking PSF drift during long sequences

---

## Core Philosophy

### Design Principles

1. **Separation of Concerns**
   - Each module has a single, well-defined responsibility
   - No tangled dependencies (e.g., calibration doesn't know about transit models)
   - Clear contracts: inputs → outputs with documented types

2. **Configuration Over Code**
   - Analysis choices explicit in YAML, not buried in source code
   - Users modify config files, not Python scripts
   - Enables reproducibility: "Here's my config file" ≈ "Here's my analysis"

3. **Fail Loudly, Validate Early**
   - Invalid parameters raise exceptions immediately
   - Config validation before pipeline execution
   - Informative error messages guide users to fixes

4. **Defaults from Literature**
   - Parameter guesses based on known exoplanet population statistics
   - Limb-darkening coefficients from stellar atmosphere models
   - Aperture sizes scaled to typical seeing conditions

5. **Observability**
   - Every stage prints human-readable progress updates
   - Diagnostic plots at each step (not just final result)
   - Numerical summaries (e.g., "Rejected 3/120 outliers")

6. **Extensibility Without Modification**
   - Users can swap out fitting algorithms (e.g., MCMC instead of curve_fit)
   - Custom detrending functions via subclassing
   - Plugin architecture for alternative photometry backends

### Trade-offs and Assumptions

#### Assumption 1: Circular Orbits
**Default:** `eccentricity = 0.0`  
**Rationale:** Most hot Jupiters are circularized by tidal forces. For eccentric systems, users must provide literature values.

#### Assumption 2: Fixed Limb-Darkening
**Default:** Quadratic law with u1=0.4, u2=0.26 (V-band, Sun-like star)  
**Rationale:** Fitting LD coefficients requires high-SNR, high-cadence data. For typical ground-based observations, fixing LD prevents degeneracies with Rp/Rs.

#### Assumption 3: Linear Systematics
**Detrending:** Removes linear trends in time and tests for linear airmass correlation  
**Limitation:** Cannot handle complex, time-varying systematics (e.g., variable clouds). Users must pre-select good nights.

#### Trade-off: Memory vs. Speed
**Implementation:** Loads all images into memory as 3D NumPy array  
**Benefit:** Fast vectorized operations, easy frame indexing  
**Cost:** Requires ~4 GB RAM for 100 frames @ 2048×2048 pixels (float32)  
**Alternative:** Frame-by-frame processing (not implemented due to complexity increase)

---

## Quick Navigation

### Documentation Structure

This documentation suite consists of:

1. **[DOCUMENTATION.md](DOCUMENTATION.md)** (this file)
   - High-level overview and philosophy
   - Problem domain and design rationale

2. **[ARCHITECTURE.md](ARCHITECTURE.md)**
   - System flow and data pipeline
   - Module interactions and dependencies
   - Design patterns and abstractions

3. **[API_REFERENCE.md](API_REFERENCE.md)**
   - Complete function/class documentation
   - Parameter specifications and examples
   - Edge cases and failure modes

4. **[USAGE_GUIDE.md](USAGE_GUIDE.md)**
   - Installation instructions
   - Configuration file anatomy
   - Step-by-step execution guide
   - Example workflows

5. **[DEVELOPMENT.md](DEVELOPMENT.md)**
   - Extension guide (adding new features)
   - Testing strategy
   - Known limitations and technical debt
   - Future roadmap

### Getting Started Fast

- **Install:** `pip install pytransitphotometry`
- **Configure:** Edit `config_example.yaml` with your data paths
- **Run:** `pytransit config.yaml` or use Python API
- **Results:** Check `outputs/` for light curves, fit parameters, and plots

### Support and Contributing

- **Issues:** Report bugs or request features on GitHub
- **Questions:** Astronomy StackExchange with `pytransitphotometry` tag
- **Contributions:** See [DEVELOPMENT.md](DEVELOPMENT.md) for guidelines

---

## Executive Summary

**pyTransitPhotometry** bridges the gap between raw telescope data and peer-reviewed science. Its modular, configuration-driven architecture ensures:

- **Reproducibility:** All decisions encoded in version-controlled config files
- **Correctness:** Proper CCD calibration, error propagation, and physical constraints
- **Usability:** Sensible defaults, diagnostic plots, and clear error messages
- **Extensibility:** Plugin architecture for custom algorithms without forking

Whether you're a student learning the ropes or a professional validating literature results, this library provides a solid foundation for exoplanet transit analysis.

**Next Steps:** Proceed to [USAGE_GUIDE.md](USAGE_GUIDE.md) for hands-on instructions, or [ARCHITECTURE.md](ARCHITECTURE.md) for technical deep-dives.
