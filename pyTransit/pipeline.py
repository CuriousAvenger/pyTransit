import numpy as np
from pathlib import Path
from typing import Dict
from .config import PipelineConfig
from .io import load_fits_files, extract_header_value, get_ccd_gain, export_lightcurve, export_fit_results
from .calibration import CalibrationFrames, create_master_frame
from .detection import detect_sources, filter_sources, select_reference_stars
from .photometry import ApertureConfig as PhotConfig, measure_flux, optimize_aperture_radius, refine_centroid
from .background import estimate_2d_background
from .psf import build_epsf, run_psf_photometry
from .lightcurve import LightCurveBuilder
from .detrending import detrend_oot
from .models import TransitFitter

class TransitPipeline:

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.validate()
        Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)
        self.calibration_frames = None
        self.science_data = None
        self.headers = None
        self.calibrated_images = None
        self.sources_list = None
        self.lightcurve = None
        self.detrended_lc = None
        self.fit_result = None
        print('\n' + '=' * 70)
        print('TRANSIT PHOTOMETRY PIPELINE INITIALIZED')
        print('=' * 70)
        self.config.summary()

    def run(self) -> Dict:
        print('\n' + '=' * 70)
        print('STARTING FULL PIPELINE EXECUTION')
        print('=' * 70 + '\n')
        print('\n[STAGE 1/6] CCD Calibration')
        print('-' * 70)
        self.run_calibration()
        print('\n[STAGE 2/6] Star Detection')
        print('-' * 70)
        self.run_detection()
        print('\n[STAGE 3/6] Aperture Photometry')
        print('-' * 70)
        self.run_photometry()
        print('\n[STAGE 4/6] Detrending & Outlier Removal')
        print('-' * 70)
        self.run_detrending()
        print('\n[STAGE 5/6] Transit Model Fitting')
        print('-' * 70)
        self.run_transit_fit()
        print('\n[STAGE 6/6] Exporting Results')
        print('-' * 70)
        self.export_results()
        print('\n' + '=' * 70)
        print('✓ PIPELINE COMPLETED SUCCESSFULLY')
        print('=' * 70 + '\n')
        return {'lightcurve': self.lightcurve, 'detrended_lc': self.detrended_lc, 'fit_result': self.fit_result, 'config': self.config}

    def run_calibration(self):
        print('Loading calibration frames...')
        bias_data, _ = load_fits_files(self.config.paths.bias_dir, self.config.paths.bias_pattern, verbose=self.config.verbose)
        dark_data, _ = load_fits_files(self.config.paths.dark_dir, self.config.paths.dark_pattern, verbose=self.config.verbose)
        flat_data, _ = load_fits_files(self.config.paths.flat_dir, self.config.paths.flat_pattern, verbose=self.config.verbose)
        print('\nCreating master calibration frames...')
        master_bias = create_master_frame(bias_data, method=self.config.calibration.combination_method, sigma_clip=self.config.calibration.sigma_clip)
        master_dark = create_master_frame(dark_data, method=self.config.calibration.combination_method, sigma_clip=self.config.calibration.sigma_clip)
        master_flat = create_master_frame(flat_data, method=self.config.calibration.combination_method, sigma_clip=self.config.calibration.sigma_clip)
        self.calibration_frames = CalibrationFrames(master_bias, master_dark, master_flat, dark_exptime=self.config.calibration.dark_exptime, flat_exptime=self.config.calibration.flat_exptime)
        print('\nLoading science frames...')
        self.science_data, self.headers = load_fits_files(self.config.paths.data_dir, self.config.paths.data_pattern, verbose=self.config.verbose)
        print('\nCalibrating science frames...')
        exptimes = extract_header_value(self.headers, 'EXPTIME', default=self.config.calibration.science_exptime)
        self.calibrated_images = self.calibration_frames.calibrate_batch(self.science_data, exptimes)
        print(f'✓ Calibration complete: {len(self.calibrated_images)} frames')

    def run_detection(self):
        if self.calibrated_images is None:
            raise RuntimeError('Must run calibration first')
        print('Detecting sources in first calibrated frame...')
        self.sources_list = []
        first_frame = self.calibrated_images[0]
        sources_0 = detect_sources(first_frame, fwhm=self.config.detection.fwhm, threshold=self.config.detection.threshold, threshold_type=self.config.detection.threshold_type, exclude_border=self.config.detection.exclude_border)
        sources_0 = filter_sources(sources_0, min_sharpness=self.config.detection.min_sharpness, max_sharpness=self.config.detection.max_sharpness, max_roundness=self.config.detection.max_roundness)
        self.sources_list.append(sources_0)
        n_stars = len(sources_0)
        print(f'Tracking {n_stars} stars across {len(self.calibrated_images) - 1} remaining frames via centroid refinement...')
        for i, frame in enumerate(self.calibrated_images[1:], start=1):
            prev_sources = self.sources_list[-1]
            new_x = []
            new_y = []
            for j in range(n_stars):
                x_prev = float(prev_sources['x_centroid'][j])
                y_prev = float(prev_sources['y_centroid'][j])
                try:
                    x_new, y_new = refine_centroid(frame, (x_prev, y_prev), box_size=51)
                except Exception:
                    x_new, y_new = (x_prev, y_prev)
                new_x.append(x_new)
                new_y.append(y_new)
            tracked = sources_0.copy()
            tracked['x_centroid'] = new_x
            tracked['y_centroid'] = new_y
            self.sources_list.append(tracked)
            if self.config.verbose and (i + 1) % 20 == 0:
                print(f'  Tracked {i + 1}/{len(self.calibrated_images)} frames...')
        print(f'\n✓ Detection & tracking complete: {len(self.sources_list)} frames')
        target_pos, ref_positions, ref_indices = select_reference_stars(sources_0, target_index=self.config.photometry.target_star_index, n_references=len(self.config.photometry.reference_star_indices))

    def run_photometry(self):
        if self.calibrated_images is None or self.sources_list is None:
            raise RuntimeError('Must run calibration and detection first')
        ccd_gain = get_ccd_gain(self.headers[0])
        phot_method = self.config.photometry.method
        bkg_method = self.config.photometry.background_method
        background_maps = None
        if bkg_method in ('background2d', 'polynomial'):
            print(f'Estimating 2D backgrounds ({bkg_method})...')
            background_maps = []
            for frame in self.calibrated_images:
                bkg, _ = estimate_2d_background(frame, box_size=self.config.photometry.background_box_size, filter_size=self.config.photometry.background_filter_size, method=bkg_method)
                background_maps.append(bkg)
        if phot_method == 'psf':
            print('Building empirical PSF from first frame...')
            first_frame = self.calibrated_images[0]
            first_bkg = background_maps[0] if background_maps else None
            first_img_sub = first_frame - first_bkg if first_bkg is not None else first_frame
            n_psf_stars = self.config.photometry.n_psf_stars
            psf_positions = [(float(self.sources_list[0]['x_centroid'][i]), float(self.sources_list[0]['y_centroid'][i])) for i in range(min(n_psf_stars, len(self.sources_list[0])))]
            epsf = build_epsf(first_img_sub, psf_positions, size=self.config.photometry.psf_size, oversampling=self.config.photometry.psf_oversampling, maxiters=self.config.photometry.psf_maxiters)

            def photometry_func(image, star_idx):
                bkg_map = None
                frame_idx = getattr(photometry_func, '_frame_idx', 0)
                if background_maps is not None:
                    bkg_map = background_maps[frame_idx]
                pos = [(float(self.sources_list[0]['x_centroid'][star_idx]), float(self.sources_list[0]['y_centroid'][star_idx]))]
                results = run_psf_photometry(image, pos, epsf, fwhm=self.config.detection.fwhm, fit_shape=self.config.photometry.psf_fit_shape, background_2d=bkg_map, ccd_gain=ccd_gain)
                r = results[0]
                return {'flux': r['flux'], 'flux_err': r['flux_err'], 'background_mean': 0.0, 'background_std': 0.0, 'snr': r['flux'] / (r['flux_err'] + 1e-10), 'aperture_sum': r['flux'], 'centroid': (r['x_fit'], r['y_fit'])}
        else:
            if self.config.photometry.optimize_aperture:
                print('Optimizing aperture radius...')
                radii_test = np.array(self.config.photometry.aperture_radii_test)
                first_frame = self.calibrated_images[0]
                target_pos = (self.sources_list[0]['x_centroid'][self.config.photometry.target_star_index], self.sources_list[0]['y_centroid'][self.config.photometry.target_star_index])
                optimal_r = optimize_aperture_radius(first_frame, target_pos, radii_test, annulus_inner=self.config.photometry.annulus_inner, annulus_outer=self.config.photometry.annulus_outer, ccd_gain=ccd_gain)
                self.config.photometry.aperture_radius = optimal_r
            phot_config = PhotConfig(aperture_radius=self.config.photometry.aperture_radius, annulus_inner=self.config.photometry.annulus_inner, annulus_outer=self.config.photometry.annulus_outer, ccd_gain=ccd_gain)
            print(f'Using {phot_config}')

            def photometry_func(image, star_idx):
                frame_idx = getattr(photometry_func, '_frame_idx', 0)
                sources = self.sources_list[frame_idx]
                position = (sources['x_centroid'][star_idx], sources['y_centroid'][star_idx])
                img_work = image
                if background_maps is not None:
                    img_work = image - background_maps[frame_idx]
                return measure_flux(img_work, position, phot_config.aperture_radius, phot_config.annulus_inner, phot_config.annulus_outer, phot_config.ccd_gain)
        times = extract_header_value(self.headers, 'JD-HELIO', default=0.0)
        times = times - 2400000.5

        def time_extractor(frame_idx):
            return times[frame_idx]
        print(f'\nBuilding differential photometry light curve [method={phot_method}, background={bkg_method}]...')
        builder = LightCurveBuilder(target_index=self.config.photometry.target_star_index, reference_indices=self.config.photometry.reference_star_indices, weighting=self.config.photometry.reference_weighting)
        self.lightcurve = builder.build(self.calibrated_images, self.sources_list, photometry_func, time_extractor, verbose=self.config.verbose)
        print(f"\n✓ Light curve extracted: {len(self.lightcurve['times'])} points")

    def run_detrending(self):
        if self.lightcurve is None:
            raise RuntimeError('Must run photometry first')
        oot_percentile = self.config.detrending.oot_percentile
        sigma_threshold = self.config.detrending.sigma_threshold
        self.detrended_lc = detrend_oot(self.lightcurve['times'], self.lightcurve['fluxes'], self.lightcurve['errors'], oot_percentile=oot_percentile, sigma_threshold=sigma_threshold)
        print(f"\n✓ Detrending complete: {len(self.detrended_lc['times'])} points")

    def run_transit_fit(self):
        if self.detrended_lc is None:
            raise RuntimeError('Must run detrending first')
        fitter = TransitFitter(period=self.config.transit_model.period, t0_guess=self.config.transit_model.t0_guess, limb_dark_u1=self.config.transit_model.limb_dark_u1, limb_dark_u2=self.config.transit_model.limb_dark_u2, ecc=self.config.transit_model.eccentricity, w=self.config.transit_model.omega)
        initial_params = {'rp': self.config.transit_model.rp_guess, 'a': self.config.transit_model.a_guess, 'inc': self.config.transit_model.inc_guess}
        bounds = {'rp': self.config.transit_model.rp_bounds, 'a': (self.config.transit_model.a_guess * (1 - self.config.transit_model.a_bounds_factor), self.config.transit_model.a_guess * (1 + self.config.transit_model.a_bounds_factor)), 'inc': (self.config.transit_model.inc_guess - self.config.transit_model.inc_bounds_offset, self.config.transit_model.inc_guess + self.config.transit_model.inc_bounds_offset)}
        self.fit_result = fitter.fit(self.detrended_lc['times'], self.detrended_lc['fluxes'], self.detrended_lc['errors'], initial_params=initial_params, bounds=bounds, fix_a_rs=self.config.transit_model.fix_a_rs)
        self.fit_result['derived_params'] = fitter.derive_physical_params(self.fit_result, r_star_solar=self.config.transit_model.r_star_solar, m_star_solar=self.config.transit_model.m_star_solar)
        print('\n✓ Transit fit complete')

    def export_results(self):
        output_dir = Path(self.config.paths.output_dir)
        if self.lightcurve is not None:
            export_lightcurve(str(output_dir / 'lightcurve_raw.csv'), self.lightcurve['times'], self.lightcurve['fluxes'], self.lightcurve['errors'])
        if self.detrended_lc is not None:
            export_lightcurve(str(output_dir / 'lightcurve_detrended.csv'), self.detrended_lc['times'], self.detrended_lc['fluxes'], self.detrended_lc['errors'])
        if self.fit_result is not None:
            fit_params = self.fit_result['fitted_params'].copy()
            fit_params.update(self.fit_result['derived_params'])
            export_fit_results(str(output_dir / 'fit_results.json'), fit_params, metadata={'target_star_index': self.config.photometry.target_star_index, 'reference_star_indices': self.config.photometry.reference_star_indices, 'period': self.config.transit_model.period, 't0': self.fit_result['t0'], 'chi_squared': self.fit_result['chi_squared'], 'reduced_chi_squared': self.fit_result['reduced_chi_squared']})
        self.config.to_yaml(str(output_dir / 'config_used.yaml'))
        print(f'\n✓ Results exported to {output_dir}')
