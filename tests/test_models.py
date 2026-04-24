import numpy as np
import pytest
_INIT_PARAMS = {'rp': 0.1, 'a': 7.0, 'inc': 83.0, 'baseline': 1.0, 'slope': 0.0}
_BOUNDS = {'rp': (0.07, 0.14), 'a': (5.0, 10.0), 'inc': (75.0, 90.0), 'baseline': (0.95, 1.05), 'slope': (-0.05, 0.05)}

class TestTransitInjectionRecovery:
    TRUTH = {'rp': 0.103, 'a': 7.17, 'inc': 82.0, 't0': 0.0, 'period': 2.4842, 'u1': 0.4, 'u2': 0.26}
    NOISE = 0.0005

    @pytest.fixture
    def synthetic_transit(self):
        batman = pytest.importorskip('batman', reason='batman-package not installed')
        truth = self.TRUTH
        times = np.linspace(-0.12, 0.12, 300)
        params = batman.TransitParams()
        params.t0 = truth['t0']
        params.per = truth['period']
        params.rp = truth['rp']
        params.a = truth['a']
        params.inc = truth['inc']
        params.ecc = 0.0
        params.w = 90.0
        params.u = [truth['u1'], truth['u2']]
        params.limb_dark = 'quadratic'
        m = batman.TransitModel(params, times)
        flux_model = m.light_curve(params)
        rng = np.random.default_rng(7)
        flux_noisy = flux_model + rng.normal(0, self.NOISE, len(times))
        errors = np.full(len(times), self.NOISE)
        return (times, flux_noisy, errors)

    def _make_fitter(self):
        from pyTransitPhotometry.models import TransitFitter
        truth = self.TRUTH
        return TransitFitter(period=truth['period'], t0_guess=truth['t0'], limb_dark_u1=truth['u1'], limb_dark_u2=truth['u2'])

    def test_fit_converges(self, synthetic_transit):
        times, fluxes, errors = synthetic_transit
        result = self._make_fitter().fit(times, fluxes, errors, initial_params=_INIT_PARAMS, bounds=_BOUNDS, fix_t0=True)
        assert 'fitted_params' in result, "fit() must return a 'fitted_params' key"
        assert 'reduced_chi_squared' in result

    def test_rp_recovery(self, synthetic_transit):
        times, fluxes, errors = synthetic_transit
        result = self._make_fitter().fit(times, fluxes, errors, initial_params=_INIT_PARAMS, bounds=_BOUNDS, fix_t0=True)
        rp_fit, rp_err = result['fitted_params']['rp']
        tol = max(3 * rp_err, 0.005)
        assert abs(rp_fit - self.TRUTH['rp']) < tol, f"Rp/Rs recovery failed: fit={rp_fit:.5f}, truth={self.TRUTH['rp']:.5f}, tol={tol:.5f}"

    def test_inclination_recovery(self, synthetic_transit):
        times, fluxes, errors = synthetic_transit
        result = self._make_fitter().fit(times, fluxes, errors, initial_params=_INIT_PARAMS, bounds=_BOUNDS, fix_t0=True)
        inc_fit, inc_err = result['fitted_params']['inc']
        tol = max(3 * inc_err, 1.0)
        assert abs(inc_fit - self.TRUTH['inc']) < tol, f"Inclination recovery failed: fit={inc_fit:.3f}°, truth={self.TRUTH['inc']:.3f}°, tol={tol:.3f}°"

    def test_semi_major_axis_recovery(self, synthetic_transit):
        times, fluxes, errors = synthetic_transit
        result = self._make_fitter().fit(times, fluxes, errors, initial_params=_INIT_PARAMS, bounds=_BOUNDS, fix_t0=True)
        a_fit, a_err = result['fitted_params']['a']
        tol = max(3 * a_err, 0.5)
        assert abs(a_fit - self.TRUTH['a']) < tol, f"a/Rs recovery failed: fit={a_fit:.4f}, truth={self.TRUTH['a']:.4f}, tol={tol:.4f}"

    def test_fit_from_off_initial_guess(self, synthetic_transit):
        times, fluxes, errors = synthetic_transit
        truth = self.TRUTH
        off_params = {'rp': truth['rp'] * 1.1, 'a': truth['a'] * 0.9, 'inc': truth['inc'] - 3.0, 'baseline': 1.0, 'slope': 0.0}
        off_bounds = {'rp': (0.06, 0.15), 'a': (4.0, 11.0), 'inc': (72.0, 90.0), 'baseline': (0.93, 1.07), 'slope': (-0.1, 0.1)}
        result = self._make_fitter().fit(times, fluxes, errors, initial_params=off_params, bounds=off_bounds, fix_t0=True)
        assert 'fitted_params' in result, 'Fit should return fitted_params'
        rp_fit = result['fitted_params']['rp'][0]
        assert abs(rp_fit - truth['rp']) < 0.015, f'Rp/Rs recovery from off-guess failed: fit={rp_fit:.4f}'
