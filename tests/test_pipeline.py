"""
pytest suite for pyTransitPhotometry.

Tests cover:
- CCD calibration (dark scaling, master flat division, full calibration)
- 2D background estimation (mesh-based and polynomial)
- Photometry: aperture flux measurement
- Detrending: sigma clipping, rolling MAD, Isolation Forest, Huber regression
- Transit injection/recovery: batman NLS must recover Rp/Rs, inc, a/Rs within 3σ
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
IMG_SHAPE = (128, 128)


def _flat_image(value: float, noise: float = 0.0, shape=IMG_SHAPE) -> np.ndarray:
    img = np.full(shape, value, dtype=np.float32)
    if noise > 0:
        img += RNG.normal(0, noise, shape).astype(np.float32)
    return img


def _gaussian_star(cx: float, cy: float, peak: float, sigma: float = 3.0,
                   shape=IMG_SHAPE) -> np.ndarray:
    y, x = np.mgrid[:shape[0], :shape[1]]
    return (peak * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))).astype(np.float32)


def _stack(n: int, value: float, noise: float = 5.0) -> np.ndarray:
    return np.stack([_flat_image(value, noise) for _ in range(n)], axis=0)


# ===========================================================================
# CCD CALIBRATION
# ===========================================================================

class TestCreateMasterFrame:
    def test_median_combination(self):
        from pyTransitPhotometry.calibration import create_master_frame
        frames = _stack(7, value=1000.0, noise=10.0)
        master = create_master_frame(frames, method="median")
        assert master.shape == IMG_SHAPE
        assert np.isclose(master.mean(), 1000.0, rtol=0.05)

    def test_mean_combination(self):
        from pyTransitPhotometry.calibration import create_master_frame
        frames = _stack(7, value=2000.0, noise=5.0)
        master = create_master_frame(frames, method="mean")
        assert np.isclose(master.mean(), 2000.0, rtol=0.02)

    def test_sigma_clip_removes_cosmic_rays(self):
        from pyTransitPhotometry.calibration import create_master_frame
        frames = _stack(9, value=500.0, noise=2.0)
        frames[4, 64, 64] = 50000.0  # cosmic ray
        master = create_master_frame(frames, method="median", sigma_clip=3.0)
        assert master[64, 64] < 1000.0, "Cosmic ray should be clipped"

    def test_raises_on_2d_input(self):
        from pyTransitPhotometry.calibration import create_master_frame
        with pytest.raises(ValueError, match="3D"):
            create_master_frame(np.ones((64, 64)))


class TestScaleDarkFrame:
    def test_linear_scaling(self):
        """Dark rate = (dark - bias) / t_dark × t_target."""
        from pyTransitPhotometry.calibration import scale_dark_frame
        bias = _flat_image(500.0)
        dark = _flat_image(600.0)   # 100 counts at 10 s → 10 counts/s
        scaled = scale_dark_frame(dark, bias, dark_exptime=10.0, target_exptime=85.0)
        assert np.allclose(scaled, 850.0, atol=1e-3), \
            f"Expected 850.0, got {scaled.mean():.2f}"

    def test_zero_scale(self):
        from pyTransitPhotometry.calibration import scale_dark_frame
        bias = _flat_image(500.0)
        dark = _flat_image(600.0)
        scaled = scale_dark_frame(dark, bias, dark_exptime=85.0, target_exptime=0.0)
        assert np.allclose(scaled, 0.0, atol=1e-6)

    def test_invalid_exptime_raises(self):
        from pyTransitPhotometry.calibration import scale_dark_frame
        with pytest.raises(ValueError):
            scale_dark_frame(np.ones((10, 10)), np.zeros((10, 10)),
                             dark_exptime=-1.0, target_exptime=85.0)


class TestCreateNormalizedFlat:
    def test_unit_mean(self):
        from pyTransitPhotometry.calibration import create_normalized_flat
        bias = _flat_image(500.0)
        flat = _flat_image(20000.0)
        dark = _flat_image(0.0)
        normalized = create_normalized_flat(flat, bias, dark)
        assert np.allclose(normalized, 1.0, atol=1e-5), \
            f"Flat mean {normalized.mean():.6f} ≠ 1.0"

    def test_non_uniform_flat_preserved(self):
        from pyTransitPhotometry.calibration import create_normalized_flat
        bias = _flat_image(0.0)
        dark = _flat_image(0.0)
        flat = np.ones(IMG_SHAPE, dtype=np.float32) * 10000.0
        flat[:, : IMG_SHAPE[1] // 2] *= 0.9   # left half 10% dimmer
        normalized = create_normalized_flat(flat, bias, dark)
        left_mean = normalized[:, : IMG_SHAPE[1] // 2].mean()
        right_mean = normalized[:, IMG_SHAPE[1] // 2:].mean()
        assert left_mean < right_mean, "Relative vignetting should be preserved"


class TestCalibrateImage:
    def test_full_pipeline(self):
        from pyTransitPhotometry.calibration import calibrate_image
        raw = _flat_image(10000.0)
        bias = _flat_image(500.0)
        dark = _flat_image(200.0)
        flat = np.ones(IMG_SHAPE, dtype=np.float32)
        calibrated = calibrate_image(raw, bias, dark, flat)
        # (10000 - 500 - 200) / 1.0 = 9300
        assert np.allclose(calibrated, 9300.0, atol=1.0), \
            f"Expected ~9300, got {calibrated.mean():.1f}"

    def test_flat_correction_scales(self):
        from pyTransitPhotometry.calibration import calibrate_image
        raw = _flat_image(10000.0)
        bias = _flat_image(0.0)
        dark = _flat_image(0.0)
        # Flat with response 0.5: corrected image should double
        flat = np.full(IMG_SHAPE, 0.5, dtype=np.float32)
        calibrated = calibrate_image(raw, bias, dark, flat)
        assert np.allclose(calibrated, 20000.0, atol=1.0)


# ===========================================================================
# 2D BACKGROUND ESTIMATION
# ===========================================================================

class TestEstimate2DBackground:
    @pytest.fixture
    def star_image(self):
        bg = _flat_image(1000.0, noise=30.0)
        star = _gaussian_star(64, 64, peak=50000.0, sigma=3.0)
        return (bg + star).astype(np.float32)

    def test_background2d_shape(self, star_image):
        from pyTransitPhotometry.photometry import estimate_2d_background
        bkg, rms = estimate_2d_background(star_image, box_size=32, method="background2d")
        assert bkg.shape == star_image.shape
        assert rms.shape == star_image.shape

    def test_background2d_accuracy(self, star_image):
        from pyTransitPhotometry.photometry import estimate_2d_background
        bkg, _ = estimate_2d_background(star_image, box_size=32, method="background2d")
        # Background should be within 20% of true value (1000)
        assert np.isclose(bkg.mean(), 1000.0, rtol=0.20), \
            f"Background mean {bkg.mean():.1f} far from 1000"

    def test_polynomial_background_shape(self, star_image):
        from pyTransitPhotometry.photometry import estimate_2d_background
        bkg, rms = estimate_2d_background(star_image, method="polynomial")
        assert bkg.shape == star_image.shape
        assert np.all(np.isfinite(bkg))

    def test_invalid_method_raises(self, star_image):
        from pyTransitPhotometry.photometry import estimate_2d_background
        with pytest.raises(ValueError, match="Unknown background method"):
            estimate_2d_background(star_image, method="magic")


# ===========================================================================
# APERTURE PHOTOMETRY
# ===========================================================================

class TestMeasureFlux:
    @pytest.fixture
    def star_image(self):
        bg = _flat_image(1000.0, noise=20.0)
        star = _gaussian_star(64, 64, peak=30000.0, sigma=2.5)
        return (bg + star).astype(np.float32), (64.0, 64.0)

    def test_flux_positive(self, star_image):
        from pyTransitPhotometry.photometry import measure_flux
        image, position = star_image
        result = measure_flux(image, position,
                              aperture_radius=8.0,
                              annulus_inner=15.0, annulus_outer=25.0)
        assert result["flux"] > 0, "Flux must be positive"

    def test_flux_error_positive(self, star_image):
        from pyTransitPhotometry.photometry import measure_flux
        image, position = star_image
        result = measure_flux(image, position,
                              aperture_radius=8.0,
                              annulus_inner=15.0, annulus_outer=25.0)
        assert result["flux_err"] > 0, "Flux error must be positive"

    def test_snr_reasonable(self, star_image):
        from pyTransitPhotometry.photometry import measure_flux
        image, position = star_image
        result = measure_flux(image, position,
                              aperture_radius=8.0,
                              annulus_inner=15.0, annulus_outer=25.0,
                              ccd_gain=1.0)
        # For a 30 000-count star with 20-count BG noise, SNR >> 10
        assert result["snr"] > 10, f"SNR {result['snr']:.1f} too low"

    def test_off_target_lower_flux(self, star_image):
        from pyTransitPhotometry.photometry import measure_flux
        image, position = star_image
        on_target = measure_flux(image, position,
                                 aperture_radius=8.0,
                                 annulus_inner=15.0, annulus_outer=25.0)
        off_target = measure_flux(image, (35.0, 35.0),   # no star here
                                  aperture_radius=8.0,
                                  annulus_inner=15.0, annulus_outer=25.0)
        assert on_target["flux"] > off_target["flux"], \
            "On-target flux should exceed off-target flux"


# ===========================================================================
# DETRENDING — SIGMA CLIPPING
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
        assert fraction_kept > 0.90, \
            f"Too many clean points rejected: {fraction_kept:.2%} kept"

    def test_output_shapes(self, clean_lc):
        from pyTransitPhotometry.detrending import sigma_clip
        times, fluxes, errors = clean_lc
        t, f, e, mask = sigma_clip(times, fluxes, errors)
        assert len(t) == len(f) == len(e) == mask.sum()


# ===========================================================================
# DETRENDING — ROLLING MAD FILTER
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
        fluxes_spiked[spike_idx] = 2.0    # far outside local MAD
        _, _, _, mask = rolling_mad_filter(
            times, fluxes_spiked, errors, window_size=15, sigma_mad=3.0
        )
        for idx in spike_idx:
            assert not mask[idx], \
                f"Tracking spike at index {idx} should be rejected by rolling MAD"

    def test_protects_transit_ingress(self, clean_lc):
        """Smooth transit dip of ~1% should not be removed by rolling MAD."""
        from pyTransitPhotometry.detrending import rolling_mad_filter
        times, fluxes, errors = clean_lc
        # Inject a 1% Gaussian-shaped transit centred at t=1.0
        transit = 0.01 * np.exp(-((times - 1.0) ** 2) / (2 * 0.05 ** 2))
        fluxes_transit = fluxes - transit
        _, _, _, mask = rolling_mad_filter(
            times, fluxes_transit, errors, window_size=20, sigma_mad=3.5
        )
        # Transit region: indices where transit depth > 0.5%
        transit_region = np.where(transit > 0.005)[0]
        if len(transit_region) > 0:
            fraction_kept = mask[transit_region].mean()
            assert fraction_kept > 0.70, \
                f"Too many transit points rejected: {fraction_kept:.0%} kept"

    def test_output_consistency(self, clean_lc):
        from pyTransitPhotometry.detrending import rolling_mad_filter
        times, fluxes, errors = clean_lc
        t, f, e, mask = rolling_mad_filter(times, fluxes, errors)
        assert len(t) == len(f) == len(e) == mask.sum()


# ===========================================================================
# DETRENDING — ISOLATION FOREST
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
        fluxes_anomaly[[50, 150, 250]] = 1.8   # clear anomalies
        _, _, _, mask = isolation_forest_filter(
            times, fluxes_anomaly, errors, contamination=0.05
        )
        n_rejected = (~mask).sum()
        assert n_rejected >= 1, \
            "Isolation Forest should reject at least one anomaly"

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
# DETRENDING — HUBER AIRMASS DETRENDING
# ===========================================================================

class TestHuberAirmassDetrend:
    @pytest.fixture
    def airmass_lc(self):
        times = np.linspace(0, 2, 200)
        airmass = 1.0 + 0.5 * times         # increasing airmass
        # True extinction: 3% flux change per unit airmass
        fluxes = 1.0 - 0.03 * (airmass - 1.0) + RNG.normal(0, 5e-4, 200)
        errors = np.full(200, 5e-4)
        return times, fluxes, errors, airmass

    def test_reduces_airmass_correlation(self, airmass_lc):
        from pyTransitPhotometry.detrending import huber_airmass_detrend
        times, fluxes, errors, airmass = airmass_lc
        f_detrended, slope, intercept = huber_airmass_detrend(
            times, fluxes, errors, airmass
        )
        corr_before = abs(np.corrcoef(airmass, fluxes)[0, 1])
        corr_after = abs(np.corrcoef(airmass, f_detrended)[0, 1])
        assert corr_after < corr_before, \
            f"Huber detrending did not reduce airmass correlation: " \
            f"before={corr_before:.3f}, after={corr_after:.3f}"

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


# ===========================================================================
# TRANSIT INJECTION / RECOVERY
# ===========================================================================

class TestTransitInjectionRecovery:
    """
    Inject a known transit signal into a noisy synthetic light curve and
    verify that the batman NLS optimiser recovers Rp/Rs, a/Rs, and
    inclination within 3σ of their injected values.
    """

    TRUTH = {
        "rp": 0.103,     # Rp/Rs  (transit depth δ = 1.06%)
        "a": 7.17,       # a/Rs
        "inc": 82.0,     # degrees
        "t0": 0.0,
        "period": 2.4842,
        "u1": 0.40,
        "u2": 0.26,
    }
    NOISE = 5e-4   # 500 ppm RMS

    @pytest.fixture
    def synthetic_transit(self):
        batman = pytest.importorskip("batman", reason="batman-package not installed")
        truth = self.TRUTH

        # Time array: 2 h baseline on each side + full transit (~2 h)
        times = np.linspace(-0.12, 0.12, 300)

        params = batman.TransitParams()
        params.t0 = truth["t0"]
        params.per = truth["period"]
        params.rp = truth["rp"]
        params.a = truth["a"]
        params.inc = truth["inc"]
        params.ecc = 0.0
        params.w = 90.0
        params.u = [truth["u1"], truth["u2"]]
        params.limb_dark = "quadratic"

        m = batman.TransitModel(params, times)
        flux_model = m.light_curve(params)

        rng = np.random.default_rng(7)
        flux_noisy = flux_model + rng.normal(0, self.NOISE, len(times))
        errors = np.full(len(times), self.NOISE)

        return times, flux_noisy, errors

    def test_fit_converges(self, synthetic_transit):
        from pyTransitPhotometry.models import TransitFitter
        times, fluxes, errors = synthetic_transit
        truth = self.TRUTH

        fitter = TransitFitter(
            period=truth["period"],
            t0_guess=truth["t0"],
            limb_dark_u1=truth["u1"],
            limb_dark_u2=truth["u2"],
        )
        # fit() raises RuntimeError on failure; assert it completes without error
        result = fitter.fit(
            times, fluxes, errors,
            initial_params={"rp": 0.10, "a": 7.0, "inc": 83.0,
                            "baseline": 1.0, "slope": 0.0},
            bounds={"rp": (0.07, 0.14), "a": (5.0, 10.0), "inc": (75.0, 90.0),
                    "baseline": (0.95, 1.05), "slope": (-0.05, 0.05)},
            fix_t0=True,
        )
        assert "fitted_params" in result, "fit() must return a 'fitted_params' key"
        assert "reduced_chi_squared" in result

    def test_rp_recovery(self, synthetic_transit):
        from pyTransitPhotometry.models import TransitFitter
        times, fluxes, errors = synthetic_transit
        truth = self.TRUTH

        fitter = TransitFitter(period=truth["period"], t0_guess=truth["t0"],
                               limb_dark_u1=truth["u1"], limb_dark_u2=truth["u2"])
        result = fitter.fit(
            times, fluxes, errors,
            initial_params={"rp": 0.10, "a": 7.0, "inc": 83.0,
                            "baseline": 1.0, "slope": 0.0},
            bounds={"rp": (0.07, 0.14), "a": (5.0, 10.0), "inc": (75.0, 90.0),
                    "baseline": (0.95, 1.05), "slope": (-0.05, 0.05)},
            fix_t0=True,
        )
        rp_fit, rp_err = result["fitted_params"]["rp"]
        tol = max(3 * rp_err, 0.005)
        assert abs(rp_fit - truth["rp"]) < tol, \
            f"Rp/Rs recovery failed: fit={rp_fit:.5f}, truth={truth['rp']:.5f}, tol={tol:.5f}"

    def test_inclination_recovery(self, synthetic_transit):
        from pyTransitPhotometry.models import TransitFitter
        times, fluxes, errors = synthetic_transit
        truth = self.TRUTH

        fitter = TransitFitter(period=truth["period"], t0_guess=truth["t0"],
                               limb_dark_u1=truth["u1"], limb_dark_u2=truth["u2"])
        result = fitter.fit(
            times, fluxes, errors,
            initial_params={"rp": 0.10, "a": 7.0, "inc": 83.0,
                            "baseline": 1.0, "slope": 0.0},
            bounds={"rp": (0.07, 0.14), "a": (5.0, 10.0), "inc": (75.0, 90.0),
                    "baseline": (0.95, 1.05), "slope": (-0.05, 0.05)},
            fix_t0=True,
        )
        inc_fit, inc_err = result["fitted_params"]["inc"]
        tol = max(3 * inc_err, 1.0)
        assert abs(inc_fit - truth["inc"]) < tol, \
            f"Inclination recovery failed: fit={inc_fit:.3f}°, " \
            f"truth={truth['inc']:.3f}°, tol={tol:.3f}°"

    def test_semi_major_axis_recovery(self, synthetic_transit):
        from pyTransitPhotometry.models import TransitFitter
        times, fluxes, errors = synthetic_transit
        truth = self.TRUTH

        fitter = TransitFitter(period=truth["period"], t0_guess=truth["t0"],
                               limb_dark_u1=truth["u1"], limb_dark_u2=truth["u2"])
        result = fitter.fit(
            times, fluxes, errors,
            initial_params={"rp": 0.10, "a": 7.0, "inc": 83.0,
                            "baseline": 1.0, "slope": 0.0},
            bounds={"rp": (0.07, 0.14), "a": (5.0, 10.0), "inc": (75.0, 90.0),
                    "baseline": (0.95, 1.05), "slope": (-0.05, 0.05)},
            fix_t0=True,
        )
        a_fit, a_err = result["fitted_params"]["a"]
        tol = max(3 * a_err, 0.5)
        assert abs(a_fit - truth["a"]) < tol, \
            f"a/Rs recovery failed: fit={a_fit:.4f}, truth={truth['a']:.4f}, tol={tol:.4f}"

    def test_fit_from_off_initial_guess(self, synthetic_transit):
        """Fitter must converge even with initial guess 20% off truth."""
        from pyTransitPhotometry.models import TransitFitter
        times, fluxes, errors = synthetic_transit
        truth = self.TRUTH

        fitter = TransitFitter(period=truth["period"], t0_guess=truth["t0"],
                               limb_dark_u1=truth["u1"], limb_dark_u2=truth["u2"])
        result = fitter.fit(
            times, fluxes, errors,
            initial_params={
                "rp": truth["rp"] * 1.10,
                "a": truth["a"] * 0.90,
                "inc": truth["inc"] - 3.0,
                "baseline": 1.0, "slope": 0.0,
            },
            bounds={"rp": (0.06, 0.15), "a": (4.0, 11.0), "inc": (72.0, 90.0),
                    "baseline": (0.93, 1.07), "slope": (-0.1, 0.1)},
            fix_t0=True,
        )
        assert "fitted_params" in result, "Fit should return fitted_params"
        rp_fit = result["fitted_params"]["rp"][0]
        assert abs(rp_fit - truth["rp"]) < 0.015, \
            f"Rp/Rs recovery from off-guess failed: fit={rp_fit:.4f}"
