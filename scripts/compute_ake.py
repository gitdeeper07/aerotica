#!/usr/bin/env python3
"""Script to compute AKE index for multiple sites."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aerotica.ake import AKEComposite


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute AKE index for sites')
    parser.add_argument('--sites', nargs='+', required=True,
                       help='Site identifiers')
    parser.add_argument('--config', type=Path,
                       help='Configuration file')
    parser.add_argument('--output', type=Path, default='ake_results.json',
                       help='Output file')
    parser.add_argument('--climate-zone', default='temperate',
                       help='Climate zone for all sites')
    return parser.parse_args()


def load_site_data(site: str) -> Dict[str, float]:
    """Load parameter data for a site.
    
    In production, this would load from databases or files.
    """
    # Sample data for demonstration
    return {
        'KED': 0.85,
        'TII': 0.76,
        'VSR': 0.89,
        'AOD': 0.34,
        'THD': 0.72,
        'PGF': 0.65,
        'HCI': 0.59,
        'ASI': 0.71,
        'LRC': 0.44
    }


def main():
    """Main entry point."""
    args = parse_args()
    
    results = {}
    
    for site in args.sites:
        print(f"🔍 Processing {site}...")
        
        # Initialize AKE
        ake = AKEComposite(
            site_id=site,
            climate_zone=args.climate_zone,
            site_type='unknown'
        )
        
        # Load parameters
        params = load_site_data(site)
        ake.load_parameters(params)
        
        # Compute
        result = ake.compute()
        results[site] = result
        
        print(f"   ✅ AKE: {result['score']:.3f} ({result['classification']})")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Results saved to {args.output}")


if __name__ == '__main__':
    main()
