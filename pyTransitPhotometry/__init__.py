"""pyTransitPhotometry — exoplanet transit photometry analysis."""

__version__ = "1.0.0"
__author__ = "Transit Photometry Team"

from .pipeline import TransitPipeline
from .config import PipelineConfig
from .lightcurve import LightCurve

__all__ = ["TransitPipeline", "PipelineConfig", "LightCurve"]
