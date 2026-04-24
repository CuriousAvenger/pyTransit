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
    warnings.warn('batman-package not installed. Transit fitting unavailable. Install with: pip install batman-package')

def _make_batman_params(t0: float, period: float, rp: float, a: float, inc: float, ecc: float=0.0, w: float=90.0, u1: float=0.4, u2: float=0.26, limb_dark: str='quadratic') -> 'batman.TransitParams':
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

def batman_transit_model(times: NDArray[np.float64], t0: float, period: float, rp: float, a: float, inc: float, ecc: float=0.0, w: float=90.0, u1: float=0.4, u2: float=0.26, limb_dark: str='quadratic') -> NDArray[np.float64]:
    if not BATMAN_AVAILABLE:
        raise ImportError('batman-package required. Install with: pip install batman-package')
    params = _make_batman_params(t0, period, rp, a, inc, ecc, w, u1, u2, limb_dark)
    m = batman.TransitModel(params, times)
    return m.light_curve(params)

class TransitFitter:

    def __init__(self, period: float, t0_guess: float, limb_dark_u1: float=0.4, limb_dark_u2: float=0.26, ecc: float=0.0, w: float=90.0) -> None:
        if not BATMAN_AVAILABLE:
            raise ImportError('batman-package required for transit fitting')
        self.period = period
        self.t0_guess = t0_guess
        self.limb_dark_u1 = limb_dark_u1
        self.limb_dark_u2 = limb_dark_u2
        self.ecc = ecc
        self.w = w

    def model_normalized(self, times: NDArray[np.float64], t0: float, rp: float, a: float, inc: float) -> NDArray[np.float64]:
        return batman_transit_model(times, t0, self.period, rp, a, inc, ecc=self.ecc, w=self.w, u1=self.limb_dark_u1, u2=self.limb_dark_u2)

    def model_with_detrending(self, times: NDArray[np.float64], rp: float, a: float, inc: float, baseline: float, slope: float, t0: Optional[float]=None) -> NDArray[np.float64]:
        if t0 is None:
            t0 = self.t0_guess
        t_mean = float(np.mean(times))
        transit = batman_transit_model(times, t0, self.period, rp, a, inc, ecc=self.ecc, w=self.w, u1=self.limb_dark_u1, u2=self.limb_dark_u2)
        return baseline * transit * (1.0 + slope * (times - t_mean))

    def _make_model_fixed_a(self, a_fixed: float) -> callable:

        def _model(times: NDArray[np.float64], t0: float, rp: float, inc: float) -> NDArray[np.float64]:
            return batman_transit_model(times, t0, self.period, rp, a_fixed, inc, ecc=self.ecc, w=self.w, u1=self.limb_dark_u1, u2=self.limb_dark_u2)
        return _model

    def fit(self, times: NDArray[np.float64], fluxes: NDArray[np.float64], errors: NDArray[np.float64], initial_params: Optional[Dict[str, float]]=None, bounds: Optional[Dict[str, Tuple[float, float]]]=None, fix_t0: bool=True, fix_a_rs: bool=False, maxfev: int=10000) -> Dict:
        if initial_params is None:
            initial_params = {'rp': 0.103, 'a': 7.17, 'inc': 82.0}
        if bounds is None:
            bounds = {'rp': (0.05, 0.15), 'a': (6.0, 12.0), 'inc': (74.5, 90.5)}
        a_value = float(initial_params.get('a', 7.17))
        if fix_a_rs:
            param_names = ['t0', 'rp', 'inc']
            p0 = [self.t0_guess, float(initial_params.get('rp', 0.103)), float(initial_params.get('inc', 82.0))]
            lower_bounds = [float(times.min()), bounds.get('rp', (0.05, 0.15))[0], bounds.get('inc', (74.5, 90.5))[0]]
            upper_bounds = [float(times.max()), bounds.get('rp', (0.05, 0.15))[1], bounds.get('inc', (74.5, 90.5))[1]]
            model_func = self._make_model_fixed_a(a_value)
        else:
            param_names = ['t0', 'rp', 'a', 'inc']
            p0 = [self.t0_guess, float(initial_params.get('rp', 0.103)), a_value, float(initial_params.get('inc', 82.0))]
            lower_bounds = [float(times.min()), bounds.get('rp', (0.05, 0.15))[0], bounds.get('a', (6.0, 12.0))[0], bounds.get('inc', (74.5, 90.5))[0]]
            upper_bounds = [float(times.max()), bounds.get('rp', (0.05, 0.15))[1], bounds.get('a', (6.0, 12.0))[1], bounds.get('inc', (74.5, 90.5))[1]]
            model_func = self.model_normalized
        print(f"\n{'=' * 70}")
        print('TRANSIT MODEL FITTING')
        print(f"{'=' * 70}")
        print(f'Transit centre (t0): {self.t0_guess:.6f} [FREE]')
        print(f'Period            : {self.period:.6f} days [FIXED]')
        if fix_a_rs:
            print(f'a/Rs              : {a_value:.4f} [FIXED — spectroscopic prior]')
        print(f'Limb darkening    : u1={self.limb_dark_u1:.2f}, u2={self.limb_dark_u2:.2f} [FIXED]')
        print('\nInitial parameters:')
        for name, val in zip(param_names, p0):
            print(f'  {name}: {val:.4f}')
        try:
            popt, pcov = curve_fit(model_func, times, fluxes, p0=p0, bounds=(lower_bounds, upper_bounds), sigma=errors, absolute_sigma=True, maxfev=maxfev)
        except Exception as exc:
            raise RuntimeError(f'Transit fit failed: {exc}') from exc
        perr = np.sqrt(np.diag(pcov))
        fitted_params: Dict[str, Tuple[float, float]] = {name: (float(val), float(err)) for name, val, err in zip(param_names, popt, perr)}
        if fix_a_rs:
            fitted_params['a'] = (float(a_value), 0.0)
        t0_fit = fitted_params['t0'][0]
        rp_fit = fitted_params['rp'][0]
        a_fit = fitted_params['a'][0]
        inc_fit = fitted_params['inc'][0]
        model_flux = self.model_normalized(times, t0_fit, rp_fit, a_fit, inc_fit)
        residuals = fluxes - model_flux
        chi_squared = float(np.sum((residuals / errors) ** 2))
        n_params = len(param_names)
        reduced_chi_squared = chi_squared / (len(times) - n_params)
        hitting_bounds: list = []
        for name, val, lo, hi in zip(param_names, popt, lower_bounds, upper_bounds):
            if abs(val - lo) < 0.01 * abs(hi - lo):
                hitting_bounds.append(f'{name} at LOWER bound')
            elif abs(val - hi) < 0.01 * abs(hi - lo):
                hitting_bounds.append(f'{name} at UPPER bound')
        if hitting_bounds:
            warnings.warn(f"Parameters hitting bounds: {', '.join(hitting_bounds)}. Consider adjusting bounds or initial guesses.")
        print(f"\n{'=' * 70}")
        print('FITTED PARAMETERS')
        print(f"{'=' * 70}")
        for name, (val, err) in fitted_params.items():
            suffix = ' [FIXED]' if fix_a_rs and name == 'a' else f' ± {err:.6f}'
            print(f'{name:>12}: {val:.6f}{suffix}')
        print(f"\n{'=' * 70}")
        print('GOODNESS OF FIT')
        print(f"{'=' * 70}")
        print(f'χ²           = {chi_squared:.2f}')
        print(f'Reduced χ²   = {reduced_chi_squared:.2f}')
        print(f'RMS residuals: {float(np.std(residuals)):.6f}')
        print(f'Mean σ       : {float(np.mean(errors)):.6f}')
        print(f'RMS / σ      : {float(np.std(residuals) / np.mean(errors)):.2f}')
        print(f"{'=' * 70}\n")
        return {'fitted_params': fitted_params, 't0': t0_fit, 'period': self.period, 'residuals': residuals, 'chi_squared': chi_squared, 'reduced_chi_squared': float(reduced_chi_squared), 'model_flux': model_flux, 'covariance': pcov, 'hitting_bounds': hitting_bounds}

    def derive_physical_params(self, fit_result: Dict, r_star_solar: Optional[float]=None, m_star_solar: Optional[float]=None) -> Dict:
        rp, rp_err = fit_result['fitted_params']['rp']
        a, a_err = fit_result['fitted_params']['a']
        inc, inc_err = fit_result['fitted_params']['inc']
        depth = rp ** 2
        depth_err = 2.0 * rp * rp_err
        inc_rad = np.deg2rad(inc)
        b = a * np.cos(inc_rad)
        b_err = np.sqrt((np.cos(inc_rad) * a_err) ** 2 + (a * np.sin(inc_rad) * np.deg2rad(inc_err)) ** 2)
        P_s = self.period * 86400.0
        rho_star_SI = 3.0 * np.pi / (G_SI * P_s ** 2) * a ** 3
        rho_star_cgs = rho_star_SI * 0.001
        rho_star_err = rho_star_cgs * 3.0 * (a_err / a) if a > 0 else 0.0
        derived: Dict[str, Tuple[float, float]] = {'transit_depth_pct': (depth * 100.0, depth_err * 100.0), 'impact_parameter': (b, b_err), 'stellar_density_cgs': (rho_star_cgs, rho_star_err)}
        if r_star_solar is not None:
            R_star = r_star_solar * R_SUN_M
            Rp_jup = rp * R_star / R_JUP_M
            Rp_jup_err = rp_err * R_star / R_JUP_M
            derived['planet_radius_jupiter'] = (Rp_jup, Rp_jup_err)
            a_AU = a * R_star / AU_M
            a_AU_err = a_err * R_star / AU_M
            derived['semi_major_axis_AU'] = (a_AU, a_AU_err)
            v_orb = 2.0 * np.pi * (a * R_star) / P_s / 1000.0
            v_orb_err = 2.0 * np.pi * (a_err * R_star) / P_s / 1000.0
            derived['orbital_velocity_kms'] = (v_orb, v_orb_err)
        print(f"\n{'=' * 70}")
        print('DERIVED PHYSICAL PARAMETERS')
        print(f"{'=' * 70}")
        for key, (val, err) in derived.items():
            print(f'{key:>30}: {val:.4f} ± {err:.4f}')
        print(f"{'=' * 70}\n")
        return derived
