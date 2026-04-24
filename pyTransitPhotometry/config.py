import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
import warnings

@dataclass
class PathConfig:
    data_dir: str
    bias_dir: str
    dark_dir: str
    flat_dir: str
    output_dir: str = './outputs'
    data_pattern: str = '*.fit'
    bias_pattern: str = '*.fit'
    dark_pattern: str = '*.fit'
    flat_pattern: str = '*.fit'

@dataclass
class CalibrationConfig:
    dark_exptime: float = 85.0
    flat_exptime: float = 1.0
    science_exptime: float = 85.0
    combination_method: str = 'median'
    sigma_clip: Optional[float] = None

@dataclass
class DetectionConfig:
    fwhm: float = 5.0
    threshold: float = 10000.0
    threshold_type: str = 'absolute'
    exclude_border: bool = True
    min_sharpness: float = 0.3
    max_sharpness: float = 1.0
    max_roundness: float = 0.5

@dataclass
class PhotometryConfig:
    method: str = 'aperture'
    aperture_radius: float = 6.0
    annulus_inner: float = 40.0
    annulus_outer: float = 60.0
    optimize_aperture: bool = False
    aperture_radii_test: list = field(default_factory=lambda: list(range(3, 20)))
    psf_size: int = 25
    psf_oversampling: int = 4
    psf_maxiters: int = 10
    psf_fit_shape: int = 11
    n_psf_stars: int = 20
    background_method: str = 'background2d'
    background_box_size: int = 64
    background_filter_size: int = 3
    target_star_index: int = 2
    reference_star_indices: list = field(default_factory=lambda: [0, 1])
    reference_weighting: str = 'inverse_variance'

@dataclass
class DetrendingConfig:
    outlier_method: str = 'rolling_mad'
    sigma_threshold: float = 3.0
    max_iterations: int = 5
    window_size: int = 20
    mad_sigma: float = 3.5
    contamination: float = 0.05
    oot_percentile: float = 25.0
    remove_linear_trend: bool = True
    test_airmass: bool = True
    airmass_regression: str = 'huber'
    huber_epsilon: float = 1.35

@dataclass
class TransitModelConfig:
    period: float = 2.4842
    t0_guess: Optional[float] = None
    fix_t0: bool = True
    limb_dark_u1: float = 0.4
    limb_dark_u2: float = 0.26
    eccentricity: float = 0.0
    omega: float = 90.0
    rp_guess: float = 0.103
    a_guess: float = 7.17
    inc_guess: float = 82.0
    rp_bounds: tuple = (0.085, 0.12)
    a_bounds_factor: float = 0.2
    inc_bounds_offset: float = 6.0
    fix_a_rs: bool = False
    r_star_solar: Optional[float] = None
    m_star_solar: Optional[float] = None

@dataclass
class PipelineConfig:
    paths: PathConfig
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    detrending: DetrendingConfig = field(default_factory=DetrendingConfig)
    transit_model: TransitModelConfig = field(default_factory=TransitModelConfig)
    verbose: bool = True
    save_intermediate: bool = True
    plot_diagnostics: bool = True

    @classmethod
    def from_yaml(cls, filepath: str) -> 'PipelineConfig':
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        paths = PathConfig(**data['paths'])
        calibration = CalibrationConfig(**data.get('calibration', {}))
        detection = DetectionConfig(**data.get('detection', {}))
        photometry = PhotometryConfig(**data.get('photometry', {}))
        detrending = DetrendingConfig(**data.get('detrending', {}))
        transit_model = TransitModelConfig(**data.get('transit_model', {}))
        options = data.get('options', {})
        return cls(paths=paths, calibration=calibration, detection=detection, photometry=photometry, detrending=detrending, transit_model=transit_model, verbose=options.get('verbose', True), save_intermediate=options.get('save_intermediate', True), plot_diagnostics=options.get('plot_diagnostics', True))

    def to_yaml(self, filepath: str):
        data = {'paths': asdict(self.paths), 'calibration': asdict(self.calibration), 'detection': asdict(self.detection), 'photometry': asdict(self.photometry), 'detrending': asdict(self.detrending), 'transit_model': asdict(self.transit_model), 'options': {'verbose': self.verbose, 'save_intermediate': self.save_intermediate, 'plot_diagnostics': self.plot_diagnostics}}
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f'✓ Configuration saved to {filepath}')

    def validate(self):
        for dir_name in ['data_dir', 'bias_dir', 'dark_dir', 'flat_dir']:
            dir_path = getattr(self.paths, dir_name)
            if not Path(dir_path).exists():
                warnings.warn(f'Directory not found: {dir_path}')
        if self.photometry.annulus_inner <= self.photometry.aperture_radius:
            raise ValueError(f'annulus_inner must be > aperture_radius (got {self.photometry.annulus_inner} <= {self.photometry.aperture_radius})')
        if self.photometry.annulus_outer <= self.photometry.annulus_inner:
            raise ValueError(f'annulus_outer must be > annulus_inner (got {self.photometry.annulus_outer} <= {self.photometry.annulus_inner})')
        if self.photometry.target_star_index in self.photometry.reference_star_indices:
            raise ValueError('Target cannot be in reference star list')
        if self.transit_model.t0_guess is None:
            warnings.warn("transit_model.t0_guess not set. You'll need to provide it before fitting.")
        print('✓ Configuration validated')

    def summary(self):
        print('\n' + '=' * 70)
        print('PIPELINE CONFIGURATION SUMMARY')
        print('=' * 70)
        print('\n[PATHS]')
        print(f'  Data: {self.paths.data_dir}')
        print(f'  Calibration: {self.paths.bias_dir}, {self.paths.dark_dir}, {self.paths.flat_dir}')
        print(f'  Output: {self.paths.output_dir}')
        print('\n[DETECTION]')
        print(f'  FWHM: {self.detection.fwhm} px')
        print(f'  Threshold: {self.detection.threshold} ({self.detection.threshold_type})')
        print('\n[PHOTOMETRY]')
        print(f'  Aperture: {self.photometry.aperture_radius} px')
        print(f'  Annulus: {self.photometry.annulus_inner}-{self.photometry.annulus_outer} px')
        print(f'  Target: star #{self.photometry.target_star_index}')
        print(f'  References: stars {self.photometry.reference_star_indices}')
        print('\n[DETRENDING]')
        print(f'  Sigma clipping: {self.detrending.sigma_threshold}σ')
        print(f'  Linear trend removal: {self.detrending.remove_linear_trend}')
        print('\n[TRANSIT MODEL]')
        print(f'  Period: {self.transit_model.period} days')
        fix_str = '[FIXED]' if self.transit_model.fix_t0 else '[FREE]'
        print(f'  t0: {self.transit_model.t0_guess} {fix_str}')
        u1 = self.transit_model.limb_dark_u1
        u2 = self.transit_model.limb_dark_u2
        print(f'  Limb darkening: u1={u1}, u2={u2}')
        print('=' * 70 + '\n')

def create_example_config(output_path: str='config_example.yaml'):
    example_config = PipelineConfig(paths=PathConfig(data_dir='./Group8.nosync/data', bias_dir='./Group8.nosync/bias', dark_dir='./Group8.nosync/darks', flat_dir='./Group8.nosync/flats', output_dir='./outputs'), transit_model=TransitModelConfig(period=2.4842, t0_guess=2460000.5, r_star_solar=1.51, m_star_solar=1.24))
    example_config.to_yaml(output_path)
    print(f'\n✓ Example configuration created: {output_path}')
    print('  Edit this file to customize for your data.')
    return example_config
