import numpy as np
import pytest
from .helpers import flat_image, gaussian_star

class TestEstimate2DBackground:

    @pytest.fixture
    def star_image(self):
        bg = flat_image(1000.0, noise=30.0)
        star = gaussian_star(64, 64, peak=50000.0, sigma=3.0)
        return (bg + star).astype(np.float32)

    def test_background2d_shape(self, star_image):
        from pyTransitPhotometry.photometry import estimate_2d_background
        bkg, rms = estimate_2d_background(star_image, box_size=32, method='background2d')
        assert bkg.shape == star_image.shape
        assert rms.shape == star_image.shape

    def test_background2d_accuracy(self, star_image):
        from pyTransitPhotometry.photometry import estimate_2d_background
        bkg, _ = estimate_2d_background(star_image, box_size=32, method='background2d')
        assert np.isclose(bkg.mean(), 1000.0, rtol=0.2), f'Background mean {bkg.mean():.1f} far from 1000'

    def test_polynomial_background_shape(self, star_image):
        from pyTransitPhotometry.photometry import estimate_2d_background
        bkg, rms = estimate_2d_background(star_image, method='polynomial')
        assert bkg.shape == star_image.shape
        assert np.all(np.isfinite(bkg))

    def test_invalid_method_raises(self, star_image):
        from pyTransitPhotometry.photometry import estimate_2d_background
        with pytest.raises(ValueError, match='Unknown background method'):
            estimate_2d_background(star_image, method='magic')

class TestMeasureFlux:

    @pytest.fixture
    def star_image(self):
        bg = flat_image(1000.0, noise=20.0)
        star = gaussian_star(64, 64, peak=30000.0, sigma=2.5)
        return ((bg + star).astype(np.float32), (64.0, 64.0))

    def test_flux_positive(self, star_image):
        from pyTransitPhotometry.photometry import measure_flux
        image, position = star_image
        result = measure_flux(image, position, aperture_radius=8.0, annulus_inner=15.0, annulus_outer=25.0)
        assert result['flux'] > 0, 'Flux must be positive'

    def test_flux_error_positive(self, star_image):
        from pyTransitPhotometry.photometry import measure_flux
        image, position = star_image
        result = measure_flux(image, position, aperture_radius=8.0, annulus_inner=15.0, annulus_outer=25.0)
        assert result['flux_err'] > 0, 'Flux error must be positive'

    def test_snr_reasonable(self, star_image):
        from pyTransitPhotometry.photometry import measure_flux
        image, position = star_image
        result = measure_flux(image, position, aperture_radius=8.0, annulus_inner=15.0, annulus_outer=25.0, ccd_gain=1.0)
        assert result['snr'] > 10, f"SNR {result['snr']:.1f} too low"

    def test_off_target_lower_flux(self, star_image):
        from pyTransitPhotometry.photometry import measure_flux
        image, position = star_image
        on_target = measure_flux(image, position, aperture_radius=8.0, annulus_inner=15.0, annulus_outer=25.0)
        off_target = measure_flux(image, (35.0, 35.0), aperture_radius=8.0, annulus_inner=15.0, annulus_outer=25.0)
        assert on_target['flux'] > off_target['flux'], 'On-target flux should exceed off-target flux'
