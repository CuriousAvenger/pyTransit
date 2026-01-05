"""
pyTransitPhotometry

A professional Python library for exoplanet transit photometry analysis.

This package provides tools for:
- CCD calibration (bias, dark, flat correction)
- Star detection and photometry
- Differential photometry and light curve extraction
- Transit model fitting with batman
- Results validation against literature

Designed for graduate students and researchers performing reproducible transit analyses.
"""

__version__ = "1.0.0"
__author__ = "Transit Photometry Team"

from .pipeline import TransitPipeline
from .config import PipelineConfig

__all__ = ["TransitPipeline", "PipelineConfig"]
