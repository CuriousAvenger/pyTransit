import numpy as np
import pytest
from .helpers import IMG_SHAPE, flat_image, stack

class TestCreateMasterFrame:

    def test_median_combination(self):
        from pyTransitPhotometry.calibration import create_master_frame
        frames = stack(7, value=1000.0, noise=10.0)
        master = create_master_frame(frames, method='median')
        assert master.shape == IMG_SHAPE
        assert np.isclose(master.mean(), 1000.0, rtol=0.05)

    def test_mean_combination(self):
        from pyTransitPhotometry.calibration import create_master_frame
        frames = stack(7, value=2000.0, noise=5.0)
        master = create_master_frame(frames, method='mean')
        assert np.isclose(master.mean(), 2000.0, rtol=0.02)

    def test_sigma_clip_removes_cosmic_rays(self):
        from pyTransitPhotometry.calibration import create_master_frame
        frames = stack(9, value=500.0, noise=2.0)
        frames[4, 64, 64] = 50000.0
        master = create_master_frame(frames, method='median', sigma_clip=3.0)
        assert master[64, 64] < 1000.0, 'Cosmic ray should be clipped'

    def test_raises_on_2d_input(self):
        from pyTransitPhotometry.calibration import create_master_frame
        with pytest.raises(ValueError, match='3D'):
            create_master_frame(np.ones((64, 64)))

class TestScaleDarkFrame:

    def test_linear_scaling(self):
        from pyTransitPhotometry.calibration import scale_dark_frame
        bias = flat_image(500.0)
        dark = flat_image(600.0)
        scaled = scale_dark_frame(dark, bias, dark_exptime=10.0, target_exptime=85.0)
        assert np.allclose(scaled, 850.0, atol=0.001), f'Expected 850.0, got {scaled.mean():.2f}'

    def test_zero_scale(self):
        from pyTransitPhotometry.calibration import scale_dark_frame
        bias = flat_image(500.0)
        dark = flat_image(600.0)
        scaled = scale_dark_frame(dark, bias, dark_exptime=85.0, target_exptime=0.0)
        assert np.allclose(scaled, 0.0, atol=1e-06)

    def test_invalid_exptime_raises(self):
        from pyTransitPhotometry.calibration import scale_dark_frame
        with pytest.raises(ValueError):
            scale_dark_frame(np.ones((10, 10)), np.zeros((10, 10)), dark_exptime=-1.0, target_exptime=85.0)

class TestCreateNormalizedFlat:

    def test_unit_mean(self):
        from pyTransitPhotometry.calibration import create_normalized_flat
        bias = flat_image(500.0)
        flat = flat_image(20000.0)
        dark = flat_image(0.0)
        normalized = create_normalized_flat(flat, bias, dark)
        assert np.allclose(normalized, 1.0, atol=1e-05), f'Flat mean {normalized.mean():.6f} ≠ 1.0'

    def test_non_uniform_flat_preserved(self):
        from pyTransitPhotometry.calibration import create_normalized_flat
        bias = flat_image(0.0)
        dark = flat_image(0.0)
        flat = np.ones(IMG_SHAPE, dtype=np.float32) * 10000.0
        flat[:, :IMG_SHAPE[1] // 2] *= 0.9
        normalized = create_normalized_flat(flat, bias, dark)
        left_mean = normalized[:, :IMG_SHAPE[1] // 2].mean()
        right_mean = normalized[:, IMG_SHAPE[1] // 2:].mean()
        assert left_mean < right_mean, 'Relative vignetting should be preserved'

class TestCalibrateImage:

    def test_full_pipeline(self):
        from pyTransitPhotometry.calibration import calibrate_image
        raw = flat_image(10000.0)
        bias = flat_image(500.0)
        dark = flat_image(200.0)
        flat = np.ones(IMG_SHAPE, dtype=np.float32)
        calibrated = calibrate_image(raw, bias, dark, flat)
        assert np.allclose(calibrated, 9300.0, atol=1.0), f'Expected ~9300, got {calibrated.mean():.1f}'

    def test_flat_correction_scales(self):
        from pyTransitPhotometry.calibration import calibrate_image
        raw = flat_image(10000.0)
        bias = flat_image(0.0)
        dark = flat_image(0.0)
        flat = np.full(IMG_SHAPE, 0.5, dtype=np.float32)
        calibrated = calibrate_image(raw, bias, dark, flat)
        assert np.allclose(calibrated, 20000.0, atol=1.0)
