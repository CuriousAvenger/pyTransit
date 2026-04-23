"""
Physical constants for transit photometry calculations.

REFACTOR: Extracted from models.py where G, R_sun, R_jup, and AU were
hard-coded inline inside ``derive_physical_params``.  Centralising them
here satisfies the single-source-of-truth rule (OWASP / clean-code) and
makes unit tests trivial.

All values in SI units (metres, kilograms, seconds) unless the name says
otherwise.

Notes
-----
Adopted values:
- G   : CODATA 2018  (6.67430e-11 m³ kg⁻¹ s⁻²)
- R☉  : IAU 2015 nominal solar radius (6.9600e8 m)
- Rjup: IAU 2015 nominal Jupiter equatorial radius (7.1492e7 m)
- AU  : IAU 2012 (1.49598e11 m)
- R⊕  : IAU 2015 nominal Earth equatorial radius (6.3710e6 m)
"""

# ── Gravitational constant ────────────────────────────────────────────────────
G_SI: float = 6.67430e-11
"""Gravitational constant in m³ kg⁻¹ s⁻²."""

# ── Stellar / solar ───────────────────────────────────────────────────────────
R_SUN_M: float = 6.9600e8
"""Nominal solar radius in metres (IAU 2015)."""

# ── Planetary / jovian ────────────────────────────────────────────────────────
R_JUP_M: float = 7.1492e7
"""Nominal Jupiter equatorial radius in metres (IAU 2015)."""

R_EARTH_M: float = 6.3710e6
"""Nominal Earth equatorial radius in metres (IAU 2015)."""

# ── Orbital ───────────────────────────────────────────────────────────────────
AU_M: float = 1.49598e11
"""Astronomical unit in metres (IAU 2012)."""
