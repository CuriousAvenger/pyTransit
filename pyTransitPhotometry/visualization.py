"""
Visualization routines for transit photometry.

Publication-quality diagnostic plots and light curve figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Optional, Tuple
import warnings


def plot_calibration_comparison(
    raw_image: np.ndarray,
    calibrated_image: np.ndarray,
    vmin: float = 700,
    vmax: float = 1400,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
):
    """
    Compare raw and calibrated images side-by-side.
    
    Parameters
    ----------
    raw_image : np.ndarray
        Raw uncalibrated image
    calibrated_image : np.ndarray
        Calibrated image
    vmin, vmax : float
        Display range for LogNorm
    figsize : tuple
        Figure size
    save_path : str, optional
        Save figure to path
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Raw image
    im0 = axes[0].imshow(raw_image, cmap='viridis', origin='lower',
                         norm=LogNorm(vmin=vmin, vmax=vmax))
    axes[0].set_title('Raw Image', fontsize=14, weight='bold')
    axes[0].set_xlabel('X (pixels)', fontsize=11)
    axes[0].set_ylabel('Y (pixels)', fontsize=11)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Counts')
    
    # Calibrated image
    im1 = axes[1].imshow(calibrated_image, cmap='viridis', origin='lower',
                         norm=LogNorm(vmin=vmin, vmax=vmax))
    axes[1].set_title('Calibrated Image\n(Bias, Dark, & Flat Corrected)',
                      fontsize=14, weight='bold')
    axes[1].set_xlabel('X (pixels)', fontsize=11)
    axes[1].set_ylabel('Y (pixels)', fontsize=11)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Counts')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_detected_sources(
    image: np.ndarray,
    sources,
    target_index: Optional[int] = None,
    reference_indices: Optional[list] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot detected sources with target and references highlighted.
    
    Parameters
    ----------
    image : np.ndarray
        Calibrated image
    sources : astropy.table.Table
        Detected sources
    target_index : int, optional
        Index of target star
    reference_indices : list, optional
        Indices of reference stars
    figsize : tuple
        Figure size
    save_path : str, optional
        Save figure to path
    """
    from photutils.aperture import CircularAperture
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=30)
    
    im = ax.imshow(image, norm=LogNorm(vmin=700, vmax=1400),
                   cmap='viridis', origin='lower')
    apertures.plot(color='yellow', lw=1.5, alpha=0.5, ax=ax)
    
    # Highlight target and references
    if target_index is not None:
        ax.plot(sources['xcentroid'][target_index],
                sources['ycentroid'][target_index],
                'gx', markersize=15, markeredgewidth=3,
                label='Target')
    
    if reference_indices:
        for i, ref_idx in enumerate(reference_indices):
            ax.plot(sources['xcentroid'][ref_idx],
                    sources['ycentroid'][ref_idx],
                    'r+', markersize=15, markeredgewidth=3,
                    label=f'Reference {i+1}' if i == 0 else '')
    
    ax.set_title(f'Detected Sources (N={len(sources)})', fontsize=14, weight='bold')
    ax.set_xlabel('X (pixels)', fontsize=11)
    ax.set_ylabel('Y (pixels)', fontsize=11)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Counts')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_lightcurve(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    title: str = "Transit Light Curve",
    xlabel: str = "Time (MJD)",
    ylabel: str = "Relative Flux",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot light curve with error bars.
    
    Parameters
    ----------
    times : np.ndarray
        Observation times
    fluxes : np.ndarray
        Flux measurements
    errors : np.ndarray
        Flux uncertainties
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    figsize : tuple
        Figure size
    save_path : str, optional
        Save figure to path
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.errorbar(times, fluxes, yerr=errors,
                fmt='o', capsize=3, alpha=0.7, color='dodgerblue',
                markersize=6, markeredgewidth=0.5, markeredgecolor='navy')
    
    # Add baseline
    baseline = np.median(fluxes)
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=2, alpha=0.5,
               label=f'Median = {baseline:.4f}')
    
    ax.set_xlabel(xlabel, fontsize=13, weight='bold')
    ax.set_ylabel(ylabel, fontsize=13, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    # Statistics box
    textstr = f'Data points: {len(times)}\n'
    textstr += f'Time span: {(times.max()-times.min())*24:.1f} hours\n'
    textstr += f'Mean flux: {np.mean(fluxes):.6f}\n'
    textstr += f'RMS scatter: {np.std(fluxes):.6f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_transit_fit(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    model_times: np.ndarray,
    model_fluxes: np.ndarray,
    residuals: np.ndarray,
    fit_params: dict,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None
):
    """
    Plot transit fit with data, model, and residuals.
    
    Parameters
    ----------
    times : np.ndarray
        Data times
    fluxes : np.ndarray
        Data fluxes
    errors : np.ndarray
        Data errors
    model_times : np.ndarray
        Model evaluation times
    model_fluxes : np.ndarray
        Model fluxes
    residuals : np.ndarray
        Fit residuals
    fit_params : dict
        Fitted parameters
    figsize : tuple
        Figure size
    save_path : str, optional
        Save figure to path
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Data + model
    ax = axes[0]
    ax.errorbar(times, fluxes, yerr=errors,
                fmt='o', capsize=2, alpha=0.6, color='dodgerblue',
                label='Data')
    ax.plot(model_times, model_fluxes, 'r-', lw=2, label='Best-fit model')
    
    rp, a, inc = fit_params['rp'][0], fit_params['a'][0], fit_params['inc'][0]
    depth_pct = (rp**2) * 100
    b = a * np.cos(np.deg2rad(inc))
    
    ax.set_xlabel("Time (MJD)", fontsize=12)
    ax.set_ylabel("Relative Flux", fontsize=12)
    ax.set_title(f"Transit Fit\nRp/Rs={rp:.4f}, depth={depth_pct:.2f}%, b={b:.2f}",
                fontsize=13, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Residuals
    ax = axes[1]
    ax.errorbar(times, residuals, yerr=errors,
                fmt='o', capsize=2, alpha=0.6, color='dodgerblue')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel("Time (MJD)", fontsize=12)
    ax.set_ylabel("Residuals", fontsize=11)
    ax.set_title("Fit Residuals", fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_airmass_correlation(
    airmass: np.ndarray,
    fluxes: np.ndarray,
    times: np.ndarray,
    correlation: float,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None
):
    """
    Plot airmass vs flux correlation and time evolution.
    
    Parameters
    ----------
    airmass : np.ndarray
        Airmass values
    fluxes : np.ndarray
        Flux measurements
    times : np.ndarray
        Observation times
    correlation : float
        Pearson correlation coefficient
    figsize : tuple
        Figure size
    save_path : str, optional
        Save figure to path
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Airmass vs flux
    ax = axes[0]
    ax.scatter(airmass, fluxes, alpha=0.6, s=30, c='coral')
    
    # Fit line
    coeffs = np.polyfit(airmass, fluxes, 1)
    ax.plot(airmass, np.polyval(coeffs, airmass),
            'r--', lw=2, label=f'r={correlation:.3f}')
    
    ax.set_xlabel("Airmass", fontsize=12, weight='bold')
    ax.set_ylabel("Flux Ratio", fontsize=12, weight='bold')
    ax.set_title("Airmass Correlation Test", fontsize=13, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Time evolution
    ax = axes[1]
    ax.plot(times, fluxes, 'o-', alpha=0.6, color='dodgerblue', markersize=4)
    ax.set_xlabel("Time (MJD)", fontsize=12, weight='bold')
    ax.set_ylabel("Flux Ratio", fontsize=12, color='dodgerblue', weight='bold')
    ax.tick_params(axis='y', labelcolor='dodgerblue')
    
    ax2 = ax.twinx()
    ax2.plot(times, airmass, 's-', alpha=0.5, color='coral', markersize=4)
    ax2.set_ylabel("Airmass", fontsize=12, color='coral', weight='bold')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    ax.set_title("Time Evolution", fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_sigma_clipping(
    times_before: np.ndarray,
    fluxes_before: np.ndarray,
    errors_before: np.ndarray,
    times_after: np.ndarray,
    fluxes_after: np.ndarray,
    errors_after: np.ndarray,
    sigma_threshold: float,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None
):
    """
    Show before/after sigma clipping comparison.
    
    Parameters
    ----------
    times_before, fluxes_before, errors_before : np.ndarray
        Data before clipping
    times_after, fluxes_after, errors_after : np.ndarray
        Data after clipping
    sigma_threshold : float
        Sigma threshold used
    figsize : tuple
        Figure size
    save_path : str, optional
        Save figure to path
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Before clipping
    ax = axes[0]
    ax.errorbar(times_before, fluxes_before, yerr=errors_before,
                fmt='o', capsize=2, color='coral', alpha=0.6)
    
    mean = np.mean(fluxes_before)
    std = np.std(fluxes_before)
    ax.axhline(y=mean, color='green', linestyle='--', alpha=0.5)
    ax.axhspan(mean - sigma_threshold*std, mean + sigma_threshold*std,
               alpha=0.2, color='green')
    
    ax.set_xlabel("Time (MJD)", fontsize=11, weight='bold')
    ax.set_ylabel("Flux Ratio", fontsize=11, weight='bold')
    ax.set_title(f"Before Clipping ({len(times_before)} points)",
                fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # After clipping
    ax = axes[1]
    ax.errorbar(times_after, fluxes_after, yerr=errors_after,
                fmt='o', capsize=2, color='dodgerblue', alpha=0.7)
    
    ax.axhline(y=np.mean(fluxes_after), color='darkblue',
               linestyle='--', alpha=0.5)
    
    ax.set_xlabel("Time (MJD)", fontsize=11, weight='bold')
    ax.set_ylabel("Flux Ratio", fontsize=11, weight='bold')
    ax.set_title(f"After {sigma_threshold}σ-Clipping ({len(times_after)} points)",
                fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()
