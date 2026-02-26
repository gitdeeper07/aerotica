"""Integration tests for AKE computation pipeline."""

import pytest
import numpy as np
from aerotica.ake import AKEComposite
from aerotica.parameters import KED, TII, VSR, AOD, THD


class TestAKEPipeline:
    """Test complete AKE computation pipeline."""
    
    def test_full_pipeline(self, sample_parameter_values):
        """Test full AKE pipeline from parameters to classification."""
        ake = AKEComposite(
            site_id="test_site",
            climate_zone="temperate",
            site_type="test"
        )
        
        ake.load_parameters(sample_parameter_values)
        result = ake.compute()
        
        assert 'score' in result
        assert 'classification' in result
        assert 'gust_risk' in result
        assert 'confidence' in result
        assert result['score'] > 0
        assert result['score'] < 1
        
    def test_with_missing_parameters(self):
        """Test pipeline with missing parameters."""
        ake = AKEComposite(
            site_id="test_site",
            climate_zone="temperate",
            site_type="test"
        )
        
        # Only provide 5 parameters
        ake.load_parameters({
            "KED": 0.83,
            "TII": 0.76,
            "VSR": 0.89,
            "THD": 0.72,
            "PGF": 0.65
        })
        
        result = ake.compute()
        
        assert result['score'] > 0
        assert len(result['missing_parameters']) == 4
        assert result['weights_used'] == 'renormalized'
    
    def test_parameter_contributions(self, sample_parameter_values):
        """Test parameter contribution calculation."""
        ake = AKEComposite(
            site_id="test_site",
            climate_zone="temperate",
            site_type="test"
        )
        
        ake.load_parameters(sample_parameter_values)
        result = ake.compute()
        
        contributions = result['contributions']
        assert len(contributions) == 9
        
        # Sum of contributions should equal score
        total_contrib = sum(c['contribution'] for c in contributions.values())
        assert abs(total_contrib - result['score']) < 1e-6
    
    def test_climate_zone_impact(self, sample_parameter_values):
        """Test impact of different climate zones."""
        zones = ['tropical', 'arid', 'temperate', 'polar']
        scores = []
        
        for zone in zones:
            ake = AKEComposite(
                site_id="test_site",
                climate_zone=zone,
                site_type="test"
            )
            ake.load_parameters(sample_parameter_values)
            result = ake.compute()
            scores.append(result['score'])
        
        # Different zones should give different scores
        assert len(set(scores)) > 1
    
    def test_ake_thresholds(self):
        """Test AKE classification thresholds."""
        ake = AKEComposite("test", "temperate", "test")
        
        test_cases = [
            (0.90, 'PREMIUM'),
            (0.75, 'VIABLE'),
            (0.60, 'MARGINAL'),
            (0.45, 'CONSTRAINED'),
            (0.30, 'BENIGN')
        ]
        
        for score, expected in test_cases:
            # Create parameters that yield approximately this score
            params = {p: score for p in ['KED', 'TII', 'VSR', 'AOD', 'THD',
                                         'PGF', 'HCI', 'ASI', 'LRC']}
            ake.load_parameters(params)
            result = ake.compute()
            assert result['classification'] == expected
    
    def test_gust_risk_assessment(self, sample_parameter_values):
        """Test gust risk assessment based on THD."""
        ake = AKEComposite("test", "temperate", "test")
        
        # High THD scenario
        high_thd = sample_parameter_values.copy()
        high_thd['THD'] = 0.85
        ake.load_parameters(high_thd)
        result = ake.compute()
        assert result['gust_risk'] in ['SEVERE', 'HIGH']
        
        # Low THD scenario
        low_thd = sample_parameter_values.copy()
        low_thd['THD'] = 0.35
        ake.load_parameters(low_thd)
        result = ake.compute()
        assert result['gust_risk'] in ['LOW', 'MODERATE']
    
    def test_export_formats(self, sample_parameter_values):
        """Test export to different formats."""
        ake = AKEComposite("test", "temperate", "test")
        ake.load_parameters(sample_parameter_values)
        
        # Test dict export
        result_dict = ake.to_dict()
        assert isinstance(result_dict, dict)
        
        # Test JSON export
        json_str = ake.to_json()
        assert isinstance(json_str, str)
        assert '"score"' in json_str
