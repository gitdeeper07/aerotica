"""End-to-end integration tests."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from aerotica.ake import AKEComposite
from aerotica.parameters import KED, THD
from aerotica.alerts import GustPreAlertEngine
from aerotica.urban import BuildingWindAssessor
from aerotica.offshore import OffshoreOptimizer


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_parameter_to_ake_pipeline(self, sample_wind_data):
        """Test from raw data to AKE score."""
        # Compute parameters from raw data
        ked = KED()
        ked_value = ked.compute(sample_wind_data['wind_speed'].mean())
        
        thd = THD()
        thd_result = thd.compute(thd_value=0.7)
        
        # Create AKE with computed parameters
        ake = AKEComposite("test_site", "temperate", "test")
        ake.load_parameters({
            "KED": ked.normalize(ked_value),
            "THD": thd_result['value'],
            "TII": 0.7,
            "VSR": 0.8,
            "AOD": 0.3,
            "PGF": 0.6,
            "HCI": 0.6,
            "ASI": 0.7,
            "LRC": 0.4
        })
        
        result = ake.compute()
        
        assert result['score'] > 0
        assert result['score'] < 1
    
    def test_ake_to_alert_pipeline(self, sample_parameter_values, sample_wind_data):
        """Test from AKE to gust alert."""
        # Compute AKE
        ake = AKEComposite("test_site", "temperate", "test")
        ake.load_parameters(sample_parameter_values)
        ake_result = ake.compute()
        
        # Use AKE result in alert system
        engine = GustPreAlertEngine(
            site_config={'location': 'test'},
            thd_threshold=0.7
        )
        
        # Mock THD detection
        engine._compute_thd = lambda x: {'value': 0.75, 'normalized': 0.75}
        
        alert = engine.evaluate(sample_wind_data)
        
        if alert:
            assert alert['thd_value'] > 0.7
            assert 'lead_time_seconds' in alert
    
    def test_urban_assessment_with_mock_data(self, sample_dem_data):
        """Test urban assessment with mock DEM."""
        assessor = BuildingWindAssessor()
        assessor.dem_data = sample_dem_data
        
        # Extract buildings
        buildings = assessor._extract_buildings()
        assert len(buildings) > 0
        
        # Assess first building
        if buildings:
            assessment = assessor.assess_building(buildings[0])
            assert 'ked' in assessment
            assert 'ake_score' in assessment
    
    def test_offshore_optimization_small(self):
        """Test offshore optimization with small configuration."""
        optimizer = OffshoreOptimizer(
            site_latitude=55.0,
            site_longitude=-3.0,
            water_depth=50,
            n_turbines=4,
            area_bounds=((0, 2000), (0, 2000))
        )
        
        # Quick setup with minimal data
        optimizer.setup([2020])  # Just one year
        optimizer.create_initial_layout('grid')
        
        # Quick optimization
        results = optimizer.optimize_layout(n_iterations=10)
        
        assert 'initial' in results
        assert 'final' in results
    
    def test_full_workflow_mock(self):
        """Test complete workflow with all components."""
        # 1. Resource assessment
        optimizer = OffshoreOptimizer(55.0, -3.0, 50, 9, ((0, 3000), (0, 3000)))
        optimizer.setup([2020])
        resource = optimizer.resource.assess_site()
        assert 'wind_statistics' in resource
        
        # 2. Layout optimization
        optimizer.create_initial_layout('staggered')
        initial_metrics = optimizer.evaluate_current_layout()
        
        # 3. Financial analysis
        financials = optimizer.calculate_financials()
        assert 'npv_euro' in financials
        assert 'irr_percent' in financials
    
    def test_data_persistence(self, temp_output_dir, sample_parameter_values):
        """Test saving and loading data."""
        import json
        
        # Save AKE result
        ake = AKEComposite("test", "temperate", "test")
        ake.load_parameters(sample_parameter_values)
        result = ake.compute()
        
        output_file = temp_output_dir / "ake_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Load and verify
        assert output_file.exists()
        with open(output_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['score'] == result['score']
        assert loaded['classification'] == result['classification']
    
    def test_error_handling(self):
        """Test error handling in pipeline."""
        ake = AKEComposite("test", "temperate", "test")
        
        # Test with empty parameters
        with pytest.raises(ValueError):
            ake.compute()
        
        # Test with invalid parameter values
        with pytest.raises(ValueError):
            ake.load_parameters({"KED": 1.5})  # > 1.0
        
        # Test with non-existent climate zone
        ake = AKEComposite("test", "invalid_zone", "test")
        ake.load_parameters({"KED": 0.5})
        result = ake.compute()  # Should still work with defaults
        assert result['score'] > 0
