#!/usr/bin/env python3
"""Offshore wind farm optimization script."""

import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aerotica.offshore import OffshoreOptimizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Offshore wind farm optimization')
    
    parser.add_argument('--site', required=True,
                       help='Site name')
    parser.add_argument('--lat', type=float, required=True,
                       help='Site latitude')
    parser.add_argument('--lon', type=float, required=True,
                       help='Site longitude')
    parser.add_argument('--depth', type=float, required=True,
                       help='Water depth [m]')
    parser.add_argument('--n-turbines', type=int, required=True,
                       help='Number of turbines')
    parser.add_argument('--area-width', type=float, required=True,
                       help='Farm width [km]')
    parser.add_argument('--area-height', type=float, required=True,
                       help='Farm height [km]')
    parser.add_argument('--layout', default='staggered',
                       choices=['grid', 'staggered', 'random'],
                       help='Initial layout pattern')
    parser.add_argument('--iterations', type=int, default=200,
                       help='Optimization iterations')
    parser.add_argument('--output', type=Path, default='offshore_results',
                       help='Output directory')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("🌊 AEROTICA Offshore Wind Farm Optimization")
    print("="*60)
    print(f"📍 Site: {args.site}")
    print(f"   Coordinates: {args.lat}°N, {args.lon}°E")
    print(f"   Water depth: {args.depth}m")
    print(f"   Area: {args.area_width}km × {args.area_height}km")
    print(f"   Turbines: {args.n_turbines}")
    
    # Convert km to m
    width_m = args.area_width * 1000
    height_m = args.area_height * 1000
    
    # Initialize optimizer
    optimizer = OffshoreOptimizer(
        site_latitude=args.lat,
        site_longitude=args.lon,
        water_depth=args.depth,
        n_turbines=args.n_turbines,
        area_bounds=((0, width_m), (0, height_m))
    )
    
    # Setup with resource data
    print("\n📊 Setting up resource assessment...")
    optimizer.setup(years=[2020, 2021, 2022, 2023, 2024])
    
    # Create initial layout
    print(f"\n🏗️  Creating initial {args.layout} layout...")
    layout = optimizer.create_initial_layout(args.layout)
    
    # Initial evaluation
    print("\n📈 Evaluating initial layout...")
    initial_metrics = optimizer.evaluate_current_layout()
    print(f"   Initial AEP: {initial_metrics['aep_mwh']/1000:.1f} GWh/year")
    print(f"   Wake losses: {initial_metrics['wake_loss_fraction']:.2%}")
    
    # Run optimization
    print(f"\n🚀 Running optimization ({args.iterations} iterations)...")
    results = optimizer.optimize_layout(
        n_iterations=args.iterations,
        verbose=True
    )
    
    # Generate report
    print("\n📊 Generating report...")
    output_dir = args.output / args.site.lower().replace(' ', '_')
    report = optimizer.generate_report(output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\n📊 Final Results:")
    print(f"   Annual Energy Production: {report['performance']['aep_gwh']:.1f} GWh")
    print(f"   Capacity Factor: {report['performance']['capacity_factor']:.2%}")
    print(f"   Wake Losses: {report['performance']['wake_losses_percent']:.1f}%")
    print(f"\n💰 Financial:")
    print(f"   NPV: €{report['financial']['npv_euro']/1e6:.1f}M")
    print(f"   IRR: {report['financial']['irr_percent']:.1f}%")
    print(f"   LCOE: €{report['financial']['lcoe_euro_per_mwh']:.1f}/MWh")
    print(f"\n📁 Reports saved to: {output_dir}")


if __name__ == '__main__':
    main()
