"""Unit tests for KED (Kinetic Energy Density) parameter."""

import pytest
import numpy as np
from aerotica.parameters import KED


class TestKED:
    """Test suite for KED parameter."""
    
    def setup_method(self):
        """Setup before each test."""
        self.ked = KED()
    
    def test_initialization(self):
        """Test proper initialization."""
        assert self.ked.weight == 0.22
        assert self.ked.air_density == 1.225
        assert self.ked.weibull_k == 2.0
    
    def test_compute_scalar(self):
        """Test computation with scalar input."""
        result = self.ked.compute(10.0)
        expected = 0.5 * 1.225 * 1000
        assert abs(result - expected) < 0.1
    
    def test_compute_array(self):
        """Test computation with array input."""
        wind = np.array([10, 12, 8])
        result = self.ked.compute(wind)
        
        assert len(result) == 3
        assert abs(result[0] - 0.5 * 1.225 * 1000) < 0.1
        assert abs(result[1] - 0.5 * 1.225 * 1728) < 0.1
        assert abs(result[2] - 0.5 * 1.225 * 512) < 0.1
    
    def test_compute_with_density_correction(self):
        """Test computation with air density correction."""
        ked = KED(config={'use_density_correction': True})
        
        # Standard conditions
        result = ked.compute(10.0, temperature=288.15, pressure=101325)
        expected = 0.5 * 1.225 * 1000
        assert abs(result - expected) < 10  # Small variation allowed
    
        # Cold, high pressure
        result_cold = ked.compute(10.0, temperature=273.15, pressure=103000)
        assert result_cold > result  # Denser air = more power
    
    def test_air_density_computation(self):
        """Test internal air density computation."""
        # Without correction
        rho = self.ked._compute_air_density()
        assert abs(rho - 1.225) < 0.01
        
        # With temperature and pressure
        rho_corrected = self.ked._compute_air_density(
            temperature=288.15,
            pressure=101325
        )
        assert abs(rho_corrected - 1.225) < 0.01
        
        # With humidity
        rho_humid = self.ked._compute_air_density(
            temperature=300,
            pressure=101325,
            humidity=0.02
        )
        assert rho_humid < 1.225  # Humid air is less dense
    
    def test_normalize(self):
        """Test normalization to [0,1] range."""
        assert self.ked.normalize(0) == 0.0
        assert self.ked.normalize(2000) == 1.0
        
        # Intermediate values
        norm_500 = self.ked.normalize(500)
        norm_1000 = self.ked.normalize(1000)
        
        assert 0 <= norm_500 <= 1
        assert 0 <= norm_1000 <= 1
        assert norm_1000 > norm_500
    
    def test_urban_bias(self):
        """Test urban bias correction."""
        # Urban bias at 60m
        assert abs(self.ked.urban_bias(60, 'urban') - 1.187) < 0.001
        
        # No bias at other heights
        assert self.ked.urban_bias(100, 'urban') == 1.0
        assert self.ked.urban_bias(20, 'urban') == 1.0
        
        # No bias for non-urban terrain
        assert self.ked.urban_bias(60, 'rural') == 1.0
        assert self.ked.urban_bias(60, 'suburban') == 1.0
        assert self.ked.urban_bias(60, 'offshore') == 1.0
    
    def test_uncertainty(self):
        """Test uncertainty estimation."""
        # Default uncertainty (10% wind speed error)
        unc = self.ked.uncertainty()
        assert abs(unc - 0.3) < 0.000001  # Use approx for floating point
        
        # Custom uncertainty
        unc_custom = self.ked.uncertainty(wind_speed_uncertainty=0.05)
        assert abs(unc_custom - 0.15) < 0.000001
    
    def test_report(self):
        """Test report generation."""
        report = self.ked.report(wind_speed=10.0)
        
        assert report['parameter'] == 'KED'
        assert report['weight'] == 0.22
        assert 'value' in report
        assert 'normalized' in report
        assert report['units'] == 'W/m²'
        assert 'formula' in report
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Zero wind speed
        assert abs(self.ked.compute(0)) < 0.001
        
        # Very high wind speed
        high = self.ked.compute(50)
        assert high > 0
        
        # Negative wind speed - should return 0 or raise exception
        # We'll test that it handles gracefully
        try:
            result = self.ked.compute(-10)
            # If it doesn't raise exception, result should be 0 or negative
            assert result <= 0
        except Exception:
            # Exception is acceptable
            pass


if __name__ == '__main__':
    pytest.main([__file__])
