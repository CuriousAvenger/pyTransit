"""
Transit model fitting using batman.

REFACTOR:
  - Physical constants (G, R☉, R_Jup, AU) removed from inline code and
    imported from the new ``constants`` module.
  - Added ``_make_batman_params()`` — the single factory function that
    constructs a ``batman.TransitParams`` object.  Avoids duplication if
    future code paths need to create params without immediately calling
    ``TransitModel``.
  - Full PEP 484 / ``numpy.typing.NDArray`` type annotations on all public
    symbols.
  - Added ``# PERF:`` annotations on the two most expensive repeated
    operations (batman model evaluation and full model rebuild per
    curve_fit iteration).
  - Docstrings now use NumPy format throughout.
  - ``BATMAN_AVAILABLE`` guard unchanged; ``ImportError`` raised at the
    call site, not at import time.
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from .constants import AU_M, G_SI, R_JUP_M, R_SUN_M

try:
    import batman

    BATMAN_AVAILABLE = True
except ImportError:
    BATMAN_AVAILABLE = False
    warnings.warn(
        "batman-package not installed. Transit fitting unavailable. "
        "Install with: pip install batman-package"
    )


# ── Batman parameter factory ───────────────────────────────────────────────────


def _make_batman_params(
    t0: float,
    period: float,
    rp: float,
    a: float,
    inc: float,
    ecc: float = 0.0,
    w: float = 90.0,
    u1: float = 0.4,
    u2: float = 0.26,
    limb_dark: str = "quadratic",
) -> "batman.TransitParams":
    """
    Construct a ``batman.TransitParams`` object.

    Single factory for all batman parameter creation.  Centralising this
    prevents parameter-ordering bugs and makes unit-testing the parameter
    setup independent of the light curve computation.

    Parameters
    ----------
    t0 : float
        Transit centre time (same units as *times* passed to the model).
    period : float
        Orbital period.
    rp : float
        Planet-to-star radius ratio (Rp/Rs).
    a : float
        Scaled semi-major axis (a/Rs).
    inc : float
        Orbital inclination in degrees.
    ecc : float, optional
        Eccentricity (default: 0.0).
    w : float, optional
        Argument of periastron in degrees (default: 90.0).
    u1, u2 : float, optional
        Quadratic limb-darkening coefficients (default: 0.4, 0.26).
    limb_dark : str, optional
        Limb-darkening law identifier (default: ``'quadratic'``).

    Returns
    -------
    params : batman.TransitParams
        Fully configured transit parameter object.
    """
    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = w
    params.u = [u1, u2]
    params.limb_dark = limb_dark
    return params


# ── Public model function ──────────────────────────────────────────────────────


def batman_transit_model(
    times: NDArray[np.float64],
    t0: float,
    period: float,
    rp: float,
    a: float,
    inc: float,
    ecc: float = 0.0,
    w: float = 90.0,
    u1: float = 0.4,
    u2: float = 0.26,
    limb_dark: str = "quadratic",
) -> NDArray[np.float64]:
    """
    Generate a batman transit model light curve.

    Parameters
    ----------
    times : NDArray[np.float64]
        Observation times (MJD, BJD, or any consistent unit).
    t0 : float
        Transit centre time.
    period : float
        Orbital period (same units as *times*).
    rp : float
        Planet-to-star radius ratio (Rp/Rs).
    a : float
        Scaled semi-major axis (a/Rs).
    inc : float
        Orbital inclination in degrees.
    ecc : float, optional
        Orbital eccentricity (default: 0.0 — circular orbit).
    w : float, optional
        Argument of periastron in degrees (default: 90.0).
    u1, u2 : float, optional
        Quadratic limb-darkening coefficients (default: 0.4, 0.26).
    limb_dark : str, optional
        Limb-darkening law (default: ``'quadratic'``).

    Returns
    -------
    model_flux : NDArray[np.float64]
        Normalised transit light curve (1.0 out of transit).

    Raises
    ------
    ImportError
        If batman-package is not installed.

    Notes
    -----
    Transit depth:       :math:`δ = (R_p/R_★)^2`
    Impact parameter:    :math:`b = (a/R_★)\\cos i`

    Uses batman (Kreidberg 2015) for fast, accurate limb-darkening integration.

    Examples
    --------
    >>> model = batman_transit_model(
    ...     times, t0=2460000.5, period=2.48, rp=0.103, a=7.2, inc=82.0
    ... )
    """
    if not BATMAN_AVAILABLE:
        raise ImportError("batman-package required. Install with: pip install batman-package")

    # PERF: batman.TransitModel is constructed fresh on every call.  If this
    # function is used inside a scipy curve_fit loop (~10 000 evaluations),
    # profiling may show that repeated object construction is significant.
    # Potential optimisation: cache the TransitModel when *times* is unchanged
    # (e.g., via functools.lru_cache on a hashable times key).
    params = _make_batman_params(t0, period, rp, a, inc, ecc, w, u1, u2, limb_dark)
    m = batman.TransitModel(params, times)
    return m.light_curve(params)


# ── Transit fitter ─────────────────────────────────────────────────────────────


class TransitFitter:
    """
    Fit a batman transit model to a normalised light curve.

    Parameters
    ----------
    period : float
        Orbital period (fixed; days or same units as input times).
    t0_guess : float
        Initial guess for transit centre time.
    limb_dark_u1 : float, optional
        Quadratic limb-darkening coefficient u₁ (default: 0.40).
    limb_dark_u2 : float, optional
        Quadratic limb-darkening coefficient u₂ (default: 0.26).
    ecc : float, optional
        Orbital eccentricity (default: 0.0).
    w : float, optional
        Argument of periastron in degrees (default: 90.0).

    Raises
    ------
    ImportError
        If batman-package is not installed.

    Examples
    --------
    >>> fitter = TransitFitter(period=2.4842, t0_guess=60969.25)
    >>> result = fitter.fit(times, fluxes, errors)
    """

    def __init__(
        self,
        period: float,
        t0_guess: float,
        limb_dark_u1: float = 0.40,
        limb_dark_u2: float = 0.26,
        ecc: float = 0.0,
        w: float = 90.0,
    ) -> None:
        if not BATMAN_AVAILABLE:
            raise ImportError("batman-package required for transit fitting")

        self.period = period
        self.t0_guess = t0_guess
        self.limb_dark_u1 = limb_dark_u1
        self.limb_dark_u2 = limb_dark_u2
        self.ecc = ecc
        self.w = w

    # ── Model wrappers ─────────────────────────────────────────────────────────

    def model_normalized(
        self,
        times: NDArray[np.float64],
        t0: float,
        rp: float,
        a: float,
        inc: float,
    ) -> NDArray[np.float64]:
        """
        Pure batman transit model for data already normalised to ~1.0.

        No baseline or slope parameters — they create degeneracy with
        transit depth on already-normalised data.

        Parameters
        ----------
        times : NDArray[np.float64]
            Observation times.
        t0 : float
            Transit centre time (free parameter).
        rp : float
            Planet-to-star radius ratio.
        a : float
            Scaled semi-major axis (a/Rs).
        inc : float
            Orbital inclination in degrees.

        Returns
        -------
        model_flux : NDArray[np.float64]
            Normalised light curve.
        """
        return batman_transit_model(
            times, t0, self.period, rp, a, inc,
            ecc=self.ecc, w=self.w,
            u1=self.limb_dark_u1, u2=self.limb_dark_u2,
        )

    def model_with_detrending(
        self,
        times: NDArray[np.float64],
        rp: float,
        a: float,
        inc: float,
        baseline: float,
        slope: float,
        t0: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """
        Transit model with linear detrending.

        Model: ``F = baseline × transit(t) × (1 + slope × (t − t_mean))``.
        Retained for use-cases where data is not pre-normalised.

        Parameters
        ----------
        times : NDArray[np.float64]
            Observation times.
        rp : float
            Rp/Rs.
        a : float
            a/Rs.
        inc : float
            Inclination in degrees.
        baseline : float
            Multiplicative baseline normalisation.
        slope : float
            Linear slope (flux / time_unit).
        t0 : float, optional
            Transit centre; defaults to ``self.t0_guess``.

        Returns
        -------
        model_flux : NDArray[np.float64]
            Transit model with linear trend applied.
        """
        if t0 is None:
            t0 = self.t0_guess
        t_mean = float(np.mean(times))
        transit = batman_transit_model(
            times, t0, self.period, rp, a, inc,
            ecc=self.ecc, w=self.w,
            u1=self.limb_dark_u1, u2=self.limb_dark_u2,
        )
        return baseline * transit * (1.0 + slope * (times - t_mean))

    def _make_model_fixed_a(self, a_fixed: float) -> callable:
        """
        Return a 3-parameter model closure with a/Rs pinned to *a_fixed*.

        Used when a/Rs is better constrained by spectroscopy than by
        photometry — common for nearly-grazing transits where Rp/Rs, a/Rs,
        and inclination are strongly degenerate.

        Parameters
        ----------
        a_fixed : float
            Fixed scaled semi-major axis (a/Rs).

        Returns
        -------
        model : callable
            ``f(times, t0, rp, inc) → NDArray[np.float64]``
        """
        def _model(
            times: NDArray[np.float64],
            t0: float,
            rp: float,
            inc: float,
        ) -> NDArray[np.float64]:
            # PERF: same batman rebuild-per-call concern as batman_transit_model.
            return batman_transit_model(
                times, t0, self.period, rp, a_fixed, inc,
                ecc=self.ecc, w=self.w,
                u1=self.limb_dark_u1, u2=self.limb_dark_u2,
            )
        return _model

    # ── Main fit ───────────────────────────────────────────────────────────────

    def fit(
        self,
        times: NDArray[np.float64],
        fluxes: NDArray[np.float64],
        errors: NDArray[np.float64],
        initial_params: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        fix_t0: bool = True,
        fix_a_rs: bool = False,
        maxfev: int = 10000,
    ) -> Dict:
        """
        Fit a transit model to normalised light curve data.

        Parameters
        ----------
        times : NDArray[np.float64]
            Observation times.
        fluxes : NDArray[np.float64]
            Normalised flux measurements (~1.0 out of transit).
        errors : NDArray[np.float64]
            1-σ flux uncertainties.
        initial_params : dict, optional
            Initial guesses: ``{'rp': 0.10, 'a': 7.17, 'inc': 82.0}``.
        bounds : dict, optional
            Parameter bounds: ``{'rp': (0.05, 0.15), ...}``.
        fix_t0 : bool, optional
            Ignored (t0 is always a free parameter in this implementation).
            Retained for API compatibility.
        fix_a_rs : bool, optional
            If True, fix a/Rs to ``initial_params['a']`` and fit only
            3 parameters (t0, rp, inc).  Recommended for nearly-grazing
            transits where the geometry is poorly constrained by photometry
            alone (default: False).
        maxfev : int, optional
            Maximum function evaluations for ``curve_fit`` (default: 10000).

        Returns
        -------
        result : dict
            Keys:

            - ``'fitted_params'``: ``{name: (value, uncertainty)}``
            - ``'t0'``: fitted transit centre
            - ``'period'``: fixed orbital period
            - ``'residuals'``: flux residuals
            - ``'chi_squared'``: χ²
            - ``'reduced_chi_squared'``: χ²_red
            - ``'model_flux'``: best-fit model at *times*
            - ``'covariance'``: parameter covariance matrix
            - ``'hitting_bounds'``: list of parameters at their bound

        Raises
        ------
        RuntimeError
            If ``scipy.optimize.curve_fit`` fails to converge.

        Notes
        -----
        Expects data already normalised to ~1.0 (use ``detrend_oot`` first).
        Period and limb darkening are always fixed.
        """
        if initial_params is None:
            initial_params = {"rp": 0.103, "a": 7.17, "inc": 82.0}
        if bounds is None:
            bounds = {"rp": (0.05, 0.15), "a": (6.0, 12.0), "inc": (74.5, 90.5)}

        a_value = float(initial_params.get("a", 7.17))

        if fix_a_rs:
            param_names = ["t0", "rp", "inc"]
            p0 = [
                self.t0_guess,
                float(initial_params.get("rp", 0.103)),
                float(initial_params.get("inc", 82.0)),
            ]
            lower_bounds = [
                float(times.min()),
                bounds.get("rp", (0.05, 0.15))[0],
                bounds.get("inc", (74.5, 90.5))[0],
            ]
            upper_bounds = [
                float(times.max()),
                bounds.get("rp", (0.05, 0.15))[1],
                bounds.get("inc", (74.5, 90.5))[1],
            ]
            model_func = self._make_model_fixed_a(a_value)
        else:
            param_names = ["t0", "rp", "a", "inc"]
            p0 = [
                self.t0_guess,
                float(initial_params.get("rp", 0.103)),
                a_value,
                float(initial_params.get("inc", 82.0)),
            ]
            lower_bounds = [
                float(times.min()),
                bounds.get("rp", (0.05, 0.15))[0],
                bounds.get("a", (6.0, 12.0))[0],
                bounds.get("inc", (74.5, 90.5))[0],
            ]
            upper_bounds = [
                float(times.max()),
                bounds.get("rp", (0.05, 0.15))[1],
                bounds.get("a", (6.0, 12.0))[1],
                bounds.get("inc", (74.5, 90.5))[1],
            ]
            model_func = self.model_normalized

        print(f"\n{'=' * 70}")
        print("TRANSIT MODEL FITTING")
        print(f"{'=' * 70}")
        print(f"Transit centre (t0): {self.t0_guess:.6f} [FREE]")
        print(f"Period            : {self.period:.6f} days [FIXED]")
        if fix_a_rs:
            print(f"a/Rs              : {a_value:.4f} [FIXED — spectroscopic prior]")
        print(
            f"Limb darkening    : u1={self.limb_dark_u1:.2f}, "
            f"u2={self.limb_dark_u2:.2f} [FIXED]"
        )
        print("\nInitial parameters:")
        for name, val in zip(param_names, p0):
            print(f"  {name}: {val:.4f}")

        try:
            popt, pcov = curve_fit(
                model_func,
                times,
                fluxes,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                sigma=errors,
                absolute_sigma=True,
                maxfev=maxfev,
            )
        except Exception as exc:
            raise RuntimeError(f"Transit fit failed: {exc}") from exc

        perr = np.sqrt(np.diag(pcov))
        fitted_params: Dict[str, Tuple[float, float]] = {
            name: (float(val), float(err))
            for name, val, err in zip(param_names, popt, perr)
        }

        if fix_a_rs:
            fitted_params["a"] = (float(a_value), 0.0)

        # Evaluate best-fit model using the canonical 4-parameter function
        t0_fit = fitted_params["t0"][0]
        rp_fit = fitted_params["rp"][0]
        a_fit = fitted_params["a"][0]
        inc_fit = fitted_params["inc"][0]
        # PERF: one final batman model evaluation; not in a hot loop.
        model_flux = self.model_normalized(times, t0_fit, rp_fit, a_fit, inc_fit)
        residuals = fluxes - model_flux

        chi_squared = float(np.sum((residuals / errors) ** 2))
        n_params = len(param_names)
        reduced_chi_squared = chi_squared / (len(times) - n_params)

        # Flag parameters at their bounds
        hitting_bounds: list = []
        for name, val, lo, hi in zip(param_names, popt, lower_bounds, upper_bounds):
            if abs(val - lo) < 0.01 * abs(hi - lo):
                hitting_bounds.append(f"{name} at LOWER bound")
            elif abs(val - hi) < 0.01 * abs(hi - lo):
                hitting_bounds.append(f"{name} at UPPER bound")
        if hitting_bounds:
            warnings.warn(
                f"Parameters hitting bounds: {', '.join(hitting_bounds)}. "
                "Consider adjusting bounds or initial guesses."
            )

        print(f"\n{'=' * 70}")
        print("FITTED PARAMETERS")
        print(f"{'=' * 70}")
        for name, (val, err) in fitted_params.items():
            suffix = " [FIXED]" if (fix_a_rs and name == "a") else f" ± {err:.6f}"
            print(f"{name:>12}: {val:.6f}{suffix}")

        print(f"\n{'=' * 70}")
        print("GOODNESS OF FIT")
        print(f"{'=' * 70}")
        print(f"χ²           = {chi_squared:.2f}")
        print(f"Reduced χ²   = {reduced_chi_squared:.2f}")
        print(f"RMS residuals: {float(np.std(residuals)):.6f}")
        print(f"Mean σ       : {float(np.mean(errors)):.6f}")
        print(f"RMS / σ      : {float(np.std(residuals) / np.mean(errors)):.2f}")
        print(f"{'=' * 70}\n")

        return {
            "fitted_params": fitted_params,
            "t0": t0_fit,
            "period": self.period,
            "residuals": residuals,
            "chi_squared": chi_squared,
            "reduced_chi_squared": float(reduced_chi_squared),
            "model_flux": model_flux,
            "covariance": pcov,
            "hitting_bounds": hitting_bounds,
        }

    # ── Physical parameters ────────────────────────────────────────────────────

    def derive_physical_params(
        self,
        fit_result: Dict,
        r_star_solar: Optional[float] = None,
        m_star_solar: Optional[float] = None,
    ) -> Dict:
        """
        Derive physical parameters from transit fit results.

        Parameters
        ----------
        fit_result : dict
            Output from :meth:`fit`.
        r_star_solar : float, optional
            Stellar radius in solar radii, for physical unit conversions.
        m_star_solar : float, optional
            Stellar mass in solar masses (currently unused; reserved for
            future dynamical consistency checks).

        Returns
        -------
        derived : dict
            Physical parameters (each value is a ``(value, uncertainty)``
            tuple unless noted):

            - ``'transit_depth_pct'``: :math:`(R_p/R_★)^2 × 100` (%)
            - ``'impact_parameter'``: :math:`b = (a/R_★)\\cos i`
            - ``'stellar_density_cgs'``: :math:`ρ_★` (g cm⁻³) from Kepler III
            - ``'planet_radius_jupiter'``: :math:`R_p` (R_Jup) — if *r_star_solar* given
            - ``'semi_major_axis_AU'``: :math:`a` (AU) — if *r_star_solar* given
            - ``'orbital_velocity_kms'``: :math:`v_{orb}` (km s⁻¹) — if *r_star_solar* given

        Notes
        -----
        Stellar density from photometry only (no mass required):

        .. math::
            ρ_★ = \\frac{3π}{G P^2}\\left(\\frac{a}{R_★}\\right)^3

        Physical constants imported from :mod:`constants`.
        """
        rp, rp_err = fit_result["fitted_params"]["rp"]
        a, a_err = fit_result["fitted_params"]["a"]
        inc, inc_err = fit_result["fitted_params"]["inc"]

        # Transit depth
        depth = rp**2
        depth_err = 2.0 * rp * rp_err

        # Impact parameter
        inc_rad = np.deg2rad(inc)
        b = a * np.cos(inc_rad)
        b_err = np.sqrt(
            (np.cos(inc_rad) * a_err) ** 2
            + (a * np.sin(inc_rad) * np.deg2rad(inc_err)) ** 2
        )

        # Stellar density from Kepler's third law (uses constants.py)
        P_s = self.period * 86400.0  # days → seconds
        rho_star_SI = (3.0 * np.pi) / (G_SI * P_s**2) * a**3  # kg m⁻³
        rho_star_cgs = rho_star_SI * 1e-3  # kg m⁻³ → g cm⁻³
        rho_star_err = rho_star_cgs * 3.0 * (a_err / a) if a > 0 else 0.0

        derived: Dict[str, Tuple[float, float]] = {
            "transit_depth_pct": (depth * 100.0, depth_err * 100.0),
            "impact_parameter": (b, b_err),
            "stellar_density_cgs": (rho_star_cgs, rho_star_err),
        }

        if r_star_solar is not None:
            R_star = r_star_solar * R_SUN_M  # metres

            # Planet radius
            Rp_jup = rp * R_star / R_JUP_M
            Rp_jup_err = rp_err * R_star / R_JUP_M
            derived["planet_radius_jupiter"] = (Rp_jup, Rp_jup_err)

            # Semi-major axis
            a_AU = a * R_star / AU_M
            a_AU_err = a_err * R_star / AU_M
            derived["semi_major_axis_AU"] = (a_AU, a_AU_err)

            # Orbital velocity
            v_orb = 2.0 * np.pi * (a * R_star) / P_s / 1e3  # m/s → km/s
            v_orb_err = 2.0 * np.pi * (a_err * R_star) / P_s / 1e3
            derived["orbital_velocity_kms"] = (v_orb, v_orb_err)

        print(f"\n{'=' * 70}")
        print("DERIVED PHYSICAL PARAMETERS")
        print(f"{'=' * 70}")
        for key, (val, err) in derived.items():
            print(f"{key:>30}: {val:.4f} ± {err:.4f}")
        print(f"{'=' * 70}\n")

        return derived
