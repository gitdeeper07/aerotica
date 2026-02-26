#!/usr/bin/env python3
"""Urban wind assessment demonstration."""

import numpy as np
from aerotica.urban import BuildingWindAssessor, UrbanMorphology, RooftopAnalyzer


def create_demo_dem():
    """Create a sample DEM for demonstration."""
    dem = np.zeros((200, 200))
    
    # Add some buildings
    # Building 1: tall building
    dem[30:50, 30:50] = 60
    
    # Building 2: medium building
    dem[80:100, 80:100] = 40
    
    # Building 3: low building complex
    dem[120:150, 120:140] = 20
    dem[120:140, 140:160] = 25
    dem[140:160, 120:140] = 30
    
    # Add height variations
    for i in range(200):
        for j in range(200):
            if dem[i, j] > 0:
                dem[i, j] += np.random.randn() * 2
    
    return dem


def main():
    """Run urban assessment demo."""
    print("="*60)
    print("AEROTICA Urban Wind Assessment Demonstration")
    print("="*60)
    
    # Create sample DEM
    print("\n🏗️  Creating sample urban DEM...")
    dem = create_demo_dem()
    print(f"   DEM shape: {dem.shape}")
    print(f"   Max height: {dem.max():.1f}m")
    
    # 1. Morphology analysis
    print("\n📐 Analyzing urban morphology...")
    morphology = UrbanMorphology(dem, resolution=2.0)
    
    stats = morphology.get_building_statistics()
    print(f"   Buildings detected: {stats['building_count']}")
    print(f"   Mean height: {stats['mean_height']:.1f}m")
    print(f"   Max height: {stats['max_height']:.1f}m")
    print(f"   Building density: {stats['building_density']:.2%}")
    
    # 2. Rooftop analysis
    print("\n🏢 Analyzing rooftops...")
    analyzer = RooftopAnalyzer(dem, resolution=2.0)
    
    rooftops = analyzer.identify_rooftops()
    print(f"   Rooftops identified: {len(rooftops)}")
    
    for i, roof in enumerate(rooftops[:3]):
        print(f"   Roof {i+1}: area={roof.area_m2:.0f}m², "
              f"height={roof.height:.1f}m, type={roof.roof_type}")
    
    # 3. Site assessment
    print("\n🌪️  Assessing wind potential...")
    assessor = BuildingWindAssessor()
    
    # Mock assessment
    viable = [r for r in rooftops if r.area_m2 > 100 and r.slope_degrees < 20]
    
    print(f"   Viable sites: {len(viable)}")
    if viable:
        total_area = sum(r.area_m2 for r in viable)
        print(f"   Total viable area: {total_area:.0f}m²")
        
        # Estimate potential power (simplified)
        avg_wind = 6.0  # m/s
        power_density = 0.5 * 1.225 * avg_wind**3 / 1000  # kW/m²
        total_power = total_area * power_density * 0.25  # 25% efficiency
        print(f"   Estimated potential: {total_power:.0f} kW")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()
