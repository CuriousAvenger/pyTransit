"""Tests for detrending routines: sigma clipping, rolling MAD, Isolation Forest, Huber."""

import numpy as np
import pytest

from .helpers import RNG


# ===========================================================================
# TestSigmaClip
# ===========================================================================


class TestSigmaClip:
    @pytest.fixture
    def clean_lc(self):
        times = np.linspace(0, 1, 150)
        fluxes = 1.0 + RNG.normal(0, 1e-3, 150)
        errors = np.full(150, 1e-3)
        return times, fluxes, errors

    def test_removes_spikes(self, clean_lc):
        from pyTransitPhotometry.detrending import sigma_clip

        times, fluxes, errors = clean_lc
        fluxes_spiked = fluxes.copy()
        spike_idx = [10, 75, 140]
        fluxes_spiked[spike_idx] = 1.5
        _, _, _, mask = sigma_clip(times, fluxes_spiked, errors, sigma_threshold=3.0)
        for idx in spike_idx:
            assert not mask[idx], f"Spike at index {idx} should be rejected"

    def test_preserves_clean_points(self, clean_lc):
        from pyTransitPhotometry.detrending import sigma_clip

        times, fluxes, errors = clean_lc
        _, _, _, mask = sigma_clip(times, fluxes, errors, sigma_threshold=3.0)
        fraction_kept = mask.sum() / len(mask)
        assert fraction_kept > 0.90, f"Too many clean points rejected: {fraction_kept:.2%} kept"

    def test_output_shapes(self, clean_lc):
        from pyTransitPhotometry.detrending import sigma_clip

        times, fluxes, errors = clean_lc
        t, f, e, mask = sigma_clip(times, fluxes, errors)
        assert len(t) == len(f) == len(e) == mask.sum()


# ===========================================================================
# TestRollingMADFilter
# ===========================================================================


class TestRollingMADFilter:
    @pytest.fixture
    def clean_lc(self):
        times = np.linspace(0, 2, 200)
        fluxes = 1.0 + RNG.normal(0, 5e-4, 200)
        errors = np.full(200, 5e-4)
        return times, fluxes, errors

    def test_rejects_tracking_spikes(self, clean_lc):
        from pyTransitPhotometry.detrending import rolling_mad_filter

        times, fluxes, errors = clean_lc
        fluxes_spiked = fluxes.copy()
        spike_idx = [25, 100, 175]
        fluxes_spiked[spike_idx] = 2.0  # far outside local MAD
        _, _, _, mask = rolling_mad_filter(
            times, fluxes_spiked, errors, window_size=15, sigma_mad=3.0
        )
        for idx in spike_idx:
            assert not mask[idx], f"Tracking spike at index {idx} should be rejected by rolling MAD"

    def test_protects_transit_ingress(self, clean_lc):
        """Smooth transit dip of ~1% should not be removed by rolling MAD."""
        from pyTransitPhotometry.detrending import rolling_mad_filter

        times, fluxes, errors = clean_lc
        # Inject a 1% Gaussian-shaped transit centred at t=1.0
        transit = 0.01 * np.exp(-((times - 1.0) ** 2) / (2 * 0.05**2))
        fluxes_transit = fluxes - transit
        _, _, _, mask = rolling_mad_filter(
            times, fluxes_transit, errors, window_size=20, sigma_mad=3.5
        )
        # Transit region: indices where transit depth > 0.5%
        transit_region = np.where(transit > 0.005)[0]
        if len(transit_region) > 0:
            fraction_kept = mask[transit_region].mean()
            assert (
                fraction_kept > 0.70
            ), f"Too many transit points rejected: {fraction_kept:.0%} kept"

    def test_output_consistency(self, clean_lc):
        from pyTransitPhotometry.detrending import rolling_mad_filter

        times, fluxes, errors = clean_lc
        t, f, e, mask = rolling_mad_filter(times, fluxes, errors)
        assert len(t) == len(f) == len(e) == mask.sum()


# ===========================================================================
# TestIsolationForestFilter
# ===========================================================================


class TestIsolationForestFilter:
    @pytest.fixture
    def clean_lc(self):
        times = np.linspace(0, 3, 300)
        fluxes = 1.0 + RNG.normal(0, 5e-4, 300)
        errors = np.full(300, 5e-4)
        return times, fluxes, errors

    def test_rejects_anomalies(self, clean_lc):
        from pyTransitPhotometry.detrending import isolation_forest_filter

        times, fluxes, errors = clean_lc
        fluxes_anomaly = fluxes.copy()
        fluxes_anomaly[[50, 150, 250]] = 1.8  # clear anomalies
        _, _, _, mask = isolation_forest_filter(times, fluxes_anomaly, errors, contamination=0.05)
        n_rejected = (~mask).sum()
        assert n_rejected >= 1, "Isolation Forest should reject at least one anomaly"

    def test_output_consistency(self, clean_lc):
        from pyTransitPhotometry.detrending import isolation_forest_filter

        times, fluxes, errors = clean_lc
        t, f, e, mask = isolation_forest_filter(times, fluxes, errors)
        assert len(t) == len(f) == len(e) == mask.sum()

    def test_no_sklearn_raises(self, clean_lc, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _block_sklearn(name, *args, **kwargs):
            if name.startswith("sklearn"):
                raise ImportError("sklearn blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_sklearn)
        from pyTransitPhotometry.detrending import isolation_forest_filter

        times, fluxes, errors = clean_lc
        with pytest.raises(ImportError, match="scikit-learn"):
            isolation_forest_filter(times, fluxes, errors)


# ===========================================================================
# TestHuberAirmassDetrend
# ===========================================================================


class TestHuberAirmassDetrend:
    @pytest.fixture
    def airmass_lc(self):
        times = np.linspace(0, 2, 200)
        airmass = 1.0 + 0.5 * times  # increasing airmass
        # True extinction: 3% flux change per unit airmass
        fluxes = 1.0 - 0.03 * (airmass - 1.0) + RNG.normal(0, 5e-4, 200)
        errors = np.full(200, 5e-4)
        return times, fluxes, errors, airmass

    def test_reduces_airmass_correlation(self, airmass_lc):
        from pyTransitPhotometry.detrending import huber_airmass_detrend

        times, fluxes, errors, airmass = airmass_lc
        f_detrended, slope, intercept = huber_airmass_detrend(times, fluxes, errors, airmass)
        corr_before = abs(np.corrcoef(airmass, fluxes)[0, 1])
        corr_after = abs(np.corrcoef(airmass, f_detrended)[0, 1])
        assert corr_after < corr_before, (
            f"Huber detrending did not reduce airmass correlation: "
            f"before={corr_before:.3f}, after={corr_after:.3f}"
        )

    def test_slope_sign(self, airmass_lc):
        from pyTransitPhotometry.detrending import huber_airmass_detrend

        times, fluxes, errors, airmass = airmass_lc
        _, slope, _ = huber_airmass_detrend(times, fluxes, errors, airmass)
        # Increasing airmass → decreasing flux → slope < 0
        assert slope < 0, f"Expected negative extinction slope, got {slope:.6f}"

    def test_output_shape_preserved(self, airmass_lc):
        from pyTransitPhotometry.detrending import huber_airmass_detrend

        times, fluxes, errors, airmass = airmass_lc
        f_detrended, _, _ = huber_airmass_detrend(times, fluxes, errors, airmass)
        assert f_detrended.shape == fluxes.shape
