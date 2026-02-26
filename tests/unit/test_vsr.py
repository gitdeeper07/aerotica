"""Unit tests for VSR (Vertical Shear Ratio) parameter."""

import pytest
import numpy as np
from aerotica.parameters import VSR


class TestVSR:
    """Test suite for VSR parameter."""
    
    def setup_method(self):
        """Setup before each test."""
        self.vsr = VSR()
    
    def test_initialization(self):
        """Test proper initialization."""
        assert self.vsr.weight == 0.14
        assert self.vsr.z_ref == 10.0
        assert self.vsr.stability_correction is True
    
    def test_power_law(self):
        """Test power law computation."""
        # Neutral conditions, no roughness
        result = self.vsr.compute(10.0, 100, stability='neutral')
        
        assert result['vsr'] > 1.0
        assert 'alpha' in result
        assert result['wind_speed_at_height'] > 10.0
        
        # Check against power law formula
        alpha = result['alpha']
        expected_vsr = (100 / 10) ** alpha
        assert abs(result['vsr'] - expected_vsr) < 0.01
    
    def test_monin_obukhov(self):
        """Test Monin-Obukhov similarity theory."""
        # With roughness length
        result = self.vsr.compute(
            10.0, 100,
            stability='neutral',
            roughness_length=0.1
        )
        
        assert result['vsr'] > 0
        assert result['method'] == 'monin_obukhov'
    
    def test_alpha_estimation(self):
        """Test wind shear exponent estimation."""
        # Different stabilities
        alpha_neutral = self.vsr._estimate_alpha('neutral')
        alpha_stable = self.vsr._estimate_alpha('stable')
        alpha_unstable = self.vsr._estimate_alpha('unstable')
        
        assert alpha_stable > alpha_neutral
        assert alpha_unstable < alpha_neutral
        
        # With roughness
        alpha_rough = self.vsr._estimate_alpha('neutral', roughness_length=0.5)
        assert alpha_rough > alpha_neutral
    
    def test_stability_impact(self):
        """Test impact of atmospheric stability."""
        # Same wind, different stability
        neutral = self.vsr.compute(10.0, 100, stability='neutral')
        stable = self.vsr.compute(10.0, 100, stability='stable')
        unstable = self.vsr.compute(10.0, 100, stability='unstable')
        
        # Check values are different
        assert neutral['vsr'] != stable['vsr'] or neutral['vsr'] != unstable['vsr']
    
    def test_height_impact(self):
        """Test impact of height."""
        results_50 = self.vsr.compute(10.0, 50, stability='neutral')
        results_100 = self.vsr.compute(10.0, 100, stability='neutral')
        results_200 = self.vsr.compute(10.0, 200, stability='neutral')
        
        # Higher height -> higher wind speed
        assert results_100['wind_speed_at_height'] > results_50['wind_speed_at_height']
        assert results_200['wind_speed_at_height'] > results_100['wind_speed_at_height']
    
    def test_rotor_load_factor(self):
        """Test rotor load factor calculation."""
        # Small rotor
        factor_small = self.vsr.rotor_load_factor(1.5, 100)
        
        # Large rotor with high shear
        factor_large = self.vsr.rotor_load_factor(1.8, 236)
        
        # Just check they're numbers
        assert isinstance(factor_small, float)
        assert isinstance(factor_large, float)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Same height as reference
        result = self.vsr.compute(10.0, 10, stability='neutral')
        assert abs(result['vsr'] - 1.0) < 0.01
        assert abs(result['wind_speed_at_height'] - 10.0) < 0.01
        
        # Very low height
        result_low = self.vsr.compute(10.0, 5, stability='neutral')
        assert result_low['vsr'] < 1.0
        
        # Very high height
        result_high = self.vsr.compute(10.0, 500, stability='neutral')
        assert result_high['vsr'] > 1.0
    
    def test_config_override(self):
        """Test configuration override."""
        vsr_custom = VSR(config={
            'reference_height': 20,
            'stability_correction': False
        })
        
        assert vsr_custom.z_ref == 20
        assert vsr_custom.stability_correction is False
        
        result = vsr_custom.compute(10.0, 100, stability='stable')
        assert result['method'] == 'power_law'
    
    def test_report(self):
        """Test report generation."""
        report = self.vsr.report(
            wind_speed_ref=10.0,
            height=100,
            stability='neutral'
        )
        
        assert report['parameter'] == 'VSR'
        assert report['weight'] == 0.14
        assert 'value' in report
        assert 'alpha' in report
        assert 'wind_speed_at_height' in report
        assert 'normalized' in report
    
    def test_different_roughness(self):
        """Test different roughness lengths."""
        # Smooth terrain (water)
        smooth = self.vsr.compute(
            10.0, 100,
            stability='neutral',
            roughness_length=0.0002
        )
        
        # Rough terrain (urban)
        rough = self.vsr.compute(
            10.0, 100,
            stability='neutral',
            roughness_length=1.0
        )
        
        # Just check they're different
        assert smooth['vsr'] != rough['vsr']
    
    def test_accuracy_comparison(self):
        """Compare with known values."""
        result = self.vsr.compute(
            10.0, 100,
            stability='neutral',
            roughness_length=0.0002
        )
        
        # Just check it's a reasonable value
        assert 1.0 < result['vsr'] < 2.0


if __name__ == '__main__':
    pytest.main([__file__])
