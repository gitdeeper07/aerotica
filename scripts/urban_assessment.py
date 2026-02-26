#!/usr/bin/env python3
"""Urban wind energy assessment script."""

import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aerotica.urban import BuildingWindAssessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Urban wind energy assessment')
    parser.add_argument('--lidar', type=Path, required=True,
                       help='LiDAR DEM file (GeoTIFF)')
    parser.add_argument('--output', type=Path, default='urban_assessment',
                       help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.75,
                       help='AKE threshold for viability')
    parser.add_argument('--min-height', type=float, default=15,
                       help='Minimum building height [m]')
    parser.add_argument('--min-area', type=float, default=50,
                       help='Minimum rooftop area [m²]')
    parser.add_argument('--geojson', action='store_true',
                       help='Output GeoJSON format')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("🌪️  AEROTICA Urban Wind Assessment")
    print("="*50)
    print(f"📍 LiDAR: {args.lidar}")
    print(f"📊 AKE threshold: {args.threshold}")
    print(f"📏 Min height: {args.min_height}m")
    print(f"📐 Min area: {args.min_area}m²")
    
    # Initialize assessor
    assessor = BuildingWindAssessor(
        lidar_dem=args.lidar,
        config={
            'ake_threshold': args.threshold,
            'min_height_m': args.min_height,
            'min_area_m2': args.min_area
        }
    )
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify viable sites
    print("\n🔍 Identifying viable rooftop locations...")
    sites = assessor.identify_sites(ake_threshold=args.threshold)
    
    print(f"\n✅ Found {len(sites)} viable sites")
    
    # Generate report
    assessor.generate_report(output_dir)
    
    # Print summary
    total_yield = sum(s.annual_yield_kwh for s in sites)
    print(f"\n📊 Summary:")
    print(f"   Total viable sites: {len(sites)}")
    print(f"   Total annual yield: {total_yield:.0f} kWh")
    print(f"   Average AKE: {np.mean([s.ake_score for s in sites]):.3f}")
    
    # Classification breakdown
    classes = {}
    for site in sites:
        classes[site.classification] = classes.get(site.classification, 0) + 1
    
    if classes:
        print("\n   Classification:")
        for cls, count in classes.items():
            print(f"     {cls}: {count}")


if __name__ == '__main__':
    main()
