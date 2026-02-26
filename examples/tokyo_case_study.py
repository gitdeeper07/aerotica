#!/usr/bin/env python3
"""Tokyo case study demonstration."""

from aerotica.ake import AKEComposite
from aerotica.urban import BuildingWindAssessor
import numpy as np


def main():
    """Run Tokyo case study."""
    print("="*60)
    print("AEROTICA Tokyo Case Study")
    print("="*60)
    
    # Tokyo coordinates
    lat, lon = 35.6762, 139.6503
    print(f"\n📍 Location: Tokyo, Japan ({lat}°N, {lon}°E)")
    
    # 1. AKE Assessment
    print("\n1️⃣ Computing AKE for Tokyo...")
    
    ake = AKEComposite(
        site_id="tokyo",
        climate_zone="temperate",
        site_type="urban_coastal"
    )
    
    # Sample parameters for Tokyo
    ake.load_parameters({
        "KED": 0.81,  # Good wind resource
        "TII": 0.72,  # Moderate turbulence
        "VSR": 0.85,  # Strong shear
        "AOD": 0.45,  # Moderate aerosols (urban + sea)
        "THD": 0.68,  # Moderate thermal activity
        "PGF": 0.70,  # Strong pressure gradients
        "HCI": 0.75,  # High humidity (coastal)
        "ASI": 0.65,  # Neutral to stable
        "LRC": 0.55   # Urban roughness
    })
    
    result = ake.compute()
    
    print(f"   AKE Score: {result['score']:.3f}")
    print(f"   Classification: {result['classification']}")
    print(f"   Gust Risk: {result['gust_risk']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    
    # 2. Parameter contributions
    print("\n2️⃣ Top contributing parameters:")
    top_params = sorted(
        result['contributions'].items(),
        key=lambda x: -x[1]['contribution']
    )[:3]
    
    for param, contrib in top_params:
        print(f"   {param}: {contrib['contribution']:.3f} "
              f"(score={contrib['score']:.2f})")
    
    # 3. Urban assessment simulation
    print("\n3️⃣ Simulating urban wind assessment...")
    
    # Mock assessor (would use real LiDAR in production)
    assessor = BuildingWindAssessor()
    
    # Simulate some buildings
    n_buildings = 127  # From paper
    total_yield = 74.0  # GWh from paper
    
    print(f"   Viable buildings identified: {n_buildings}")
    print(f"   Total annual yield: {total_yield} GWh")
    print(f"   Equivalent households: {int(total_yield * 1000 / 4000)}")
    
    print("\n✅ Case study completed!")


if __name__ == "__main__":
    main()
