"""Unit tests for urban wind assessment."""

import pytest
import numpy as np
from pathlib import Path

from aerotica.urban import BuildingWindAssessor, UrbanMorphology, RooftopAnalyzer


class TestBuildingWindAssessor:
    """Test building wind assessor."""
    
    def setup_method(self):
        self.assessor = BuildingWindAssessor()
        
        # Mock DEM data
        self.assessor.dem_data = np.random.rand(100, 100) * 50
        self.assessor.dem_transform = None
    
    def test_extract_buildings(self):
        """Test building extraction."""
        # Set some high points as buildings
        self.assessor.dem_data[20:30, 20:30] = 30
        self.assessor.dem_data[60:70, 60:70] = 40
        
        buildings = self.assessor._extract_buildings()
        
        assert len(buildings) > 0
        assert all('height' in b for b in buildings)
        assert all('area' in b for b in buildings)
    
    def test_assess_building(self):
        """Test building assessment."""
        building = {
            'x': 0, 'y': 0,
            'height': 30,
            'area': 200,
            'roof_type': 'flat'
        }
        
        assessment = self.assessor.assess_building(building)
        
        assert 'ked' in assessment
        assert 'ake_score' in assessment
        assert 'classification' in assessment
        assert 'annual_yield_kwh' in assessment
        assert 0 <= assessment['confidence'] <= 1
    
    def test_to_geojson(self):
        """Test GeoJSON conversion."""
        from aerotica.urban.assessor import BuildingSite
        
        sites = [
            BuildingSite(
                id='TEST_001',
                x=100, y=200,
                height=30,
                roof_type='flat',
                area_m2=150,
                ked_w_m2=500,
                ake_score=0.85,
                classification='PREMIUM',
                annual_yield_kwh=5000,
                confidence=0.9
            )
        ]
        
        geojson = self.assessor.to_geojson(sites)
        
        assert geojson['type'] == 'FeatureCollection'
        assert len(geojson['features']) == 1
        assert 'metadata' in geojson


class TestUrbanMorphology:
    """Test urban morphology analysis."""
    
    def setup_method(self):
        self.dem = np.random.rand(200, 200) * 50
        self.morphology = UrbanMorphology(self.dem, resolution=2.0)
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        metrics = self.morphology.compute_metrics(0, 0, 100, 100)
        
        assert hasattr(metrics, 'mean_height')
        assert hasattr(metrics, 'roughness_length')
        assert hasattr(metrics, 'plan_area_density')
        assert 0 <= metrics.plan_area_density <= 1
    
    def test_compute_roughness_map(self):
        """Test roughness map computation."""
        roughness_map = self.morphology.compute_roughness_map()
        
        assert roughness_map.shape == self.dem.shape
        assert np.all(roughness_map >= 0)
    
    def test_classify_urban_zone(self):
        """Test urban zone classification."""
        from aerotica.urban.morphology import UrbanMetrics
        
        metrics = UrbanMetrics(
            mean_height=20,
            std_height=5,
            frontal_area_index=0.2,
            plan_area_density=0.4,
            roughness_length=1.5,
            displacement_height=14,
            building_count=100,
            sky_view_factor=0.3
        )
        
        zone = self.morphology.classify_urban_zone(metrics)
        assert zone in ['HIGH_DENSITY_URBAN', 'HIGH_RISE_URBAN', 
                       'MEDIUM_DENSITY_URBAN', 'LOW_DENSITY_URBAN',
                       'SUBURBAN', 'RURAL']


class TestRooftopAnalyzer:
    """Test rooftop analysis."""
    
    def setup_method(self):
        self.dem = np.zeros((200, 200))
        # Add a building
        self.dem[50:100, 50:100] = 30
        self.analyzer = RooftopAnalyzer(self.dem, resolution=2.0)
    
    def test_identify_rooftops(self):
        """Test rooftop identification."""
        rooftops = self.analyzer.identify_rooftops()
        
        assert len(rooftops) > 0
        assert all(r.area_m2 > 0 for r in rooftops)
        assert all(r.slope_degrees >= 0 for r in rooftops)
    
    def test_rooftop_features(self):
        """Test rooftop feature extraction."""
        rooftops = self.analyzer.identify_rooftops()
        
        if rooftops:
            r = rooftops[0]
            assert 0 <= r.exposure_factor <= 1
            assert 0 <= r.turbulence_potential <= 1
            assert isinstance(r.suitable_for_turbine, bool)
    
    def test_optimal_placement(self):
        """Test optimal turbine placement."""
        rooftops = self.analyzer.identify_rooftops()
        placements = self.analyzer.get_optimal_placement(rooftops)
        
        for p in placements:
            assert 'x' in p
            assert 'y' in p
            assert 'turbine_type' in p
            assert p['estimated_power_kw'] > 0


if __name__ == '__main__':
    pytest.main([__file__])
