"""
Command-line interface for transit photometry pipeline.

    transit-pipeline config.yaml
"""

import argparse
import sys
from pathlib import Path

from .pipeline import TransitPipeline
from .config import PipelineConfig, create_example_config


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Transit Photometry Pipeline - Exoplanet transit analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  transit-pipeline config.yaml

  # Create example configuration
  transit-pipeline --create-config my_config.yaml

  # Run with custom output directory
  transit-pipeline config.yaml --output ./my_results

  # Run individual stages
  transit-pipeline config.yaml --stages calibration detection photometry
        """,
    )

    parser.add_argument("config", nargs="?", help="Path to YAML configuration file")

    parser.add_argument(
        "--create-config", metavar="PATH", help="Create an example configuration file at PATH"
    )

    parser.add_argument(
        "--output", "-o", metavar="DIR", help="Override output directory from config"
    )

    parser.add_argument(
        "--stages",
        "-s",
        nargs="+",
        choices=["calibration", "detection", "photometry", "detrending", "fit", "export"],
        help="Run only specified stages (default: all)",
    )

    parser.add_argument("--no-plots", action="store_true", help="Disable diagnostic plots")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    parser.add_argument("--version", action="version", version="transitphotometry 1.0.0")

    args = parser.parse_args()

    if args.create_config:
        create_example_config(args.create_config)
        print(f"\n✓ Example configuration created: {args.create_config}")
        print("  Edit this file to match your data, then run:")
        print(f"    transit-pipeline {args.create_config}")
        return 0

    if not args.config:
        parser.error("config file required (or use --create-config)")

    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
        return 1

    try:
        print(f"Loading configuration from {args.config}...")
        config = PipelineConfig.from_yaml(args.config)

        if args.output:
            config.paths.output_dir = args.output

        if args.no_plots:
            config.plot_diagnostics = False

        if args.verbose:
            config.verbose = True

        pipeline = TransitPipeline(config)

        if args.stages:
            stages = args.stages
        else:
            stages = ["calibration", "detection", "photometry", "detrending", "fit", "export"]

        print(f"\nRunning stages: {', '.join(stages)}\n")

        if "calibration" in stages:
            print("\n[1/6] Running calibration...")
            pipeline.run_calibration()

        if "detection" in stages:
            print("\n[2/6] Running detection...")
            pipeline.run_detection()

        if "photometry" in stages:
            print("\n[3/6] Running photometry...")
            pipeline.run_photometry()

        if "detrending" in stages:
            print("\n[4/6] Running detrending...")
            pipeline.run_detrending()

        if "fit" in stages:
            print("\n[5/6] Running transit fit...")
            pipeline.run_transit_fit()

        if "export" in stages:
            print("\n[6/6] Exporting results...")
            pipeline.export_results()

        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nResults saved to: {config.paths.output_dir}")

        # Print summary
        if "fit" in stages and pipeline.fit_result:
            fit = pipeline.fit_result["fitted_params"]
            print("\nFitted Parameters:")
            print(f"  Rp/Rs = {fit['rp'][0]:.4f} ± {fit['rp'][1]:.4f}")
            print(f"  a/Rs  = {fit['a'][0]:.2f} ± {fit['a'][1]:.2f}")
            print(f"  inc   = {fit['inc'][0]:.2f}° ± {fit['inc'][1]:.2f}°")

            derived = pipeline.fit_result["derived_params"]
            depth = derived["transit_depth_pct"]
            print(f"\n  Transit depth = {depth[0]:.2f}% ± {depth[1]:.2f}%")

            b = derived["impact_parameter"]
            print(f"  Impact parameter = {b[0]:.3f} ± {b[1]:.3f}")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
