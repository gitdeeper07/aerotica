"""Unit tests for AKE parameters."""

import pytest
import numpy as np
from aerotica.parameters import (
    KED, TII, VSR, AOD, THD, PGF, HCI, ASI, LRC
)


class TestKED:
    """Test Kinetic Energy Density parameter."""
    
    def test_initialization(self):
        ked = KED()
        assert ked.weight == 0.22
        assert ked.air_density == 1.225
    
    def test_compute_scalar(self):
        ked = KED()
        result = ked.compute(10.0)
        expected = 0.5 * 1.225 * 1000
        assert abs(result - expected) < 0.1
    
    def test_compute_array(self):
        ked = KED()
        wind = np.array([10, 12, 8])
        result = ked.compute(wind)
        assert len(result) == 3
        assert abs(result[0] - 0.5 * 1.225 * 1000) < 0.1
    
    def test_with_density_correction(self):
        ked = KED(config={'use_density_correction': True})
        result = ked.compute(10.0, temperature=288.15, pressure=101325)
        assert result > 0
    
    def test_normalize(self):
        ked = KED()
        assert ked.normalize(0) == 0
        assert ked.normalize(2000) == 1.0
        assert 0 <= ked.normalize(500) <= 1
    
    def test_urban_bias(self):
        ked = KED()
        assert abs(ked.urban_bias(60, 'urban') - 1.187) < 0.001
        assert ked.urban_bias(100, 'urban') == 1.0


class TestTII:
    """Test Turbulence Intensity Index."""
    
    def test_initialization(self):
        tii = TII()
        assert tii.weight == 0.16
    
    def test_scalar_tii(self):
        tii = TII()
        series = [10, 12, 11, 13, 12, 11, 14, 13, 12, 11]
        result = tii.compute(series, method='scalar')
        assert 0 <= result <= 1
    
    def test_spectral_tii(self):
        tii = TII()
        series = np.random.randn(1000) * 2 + 10
        result = tii.compute(series, method='spectral')
        assert 0 <= result <= 1
    
    def test_normalize(self):
        tii = TII()
        assert tii.normalize(0.1) == 1.0
        assert tii.normalize(0.3) == 0.0
    
    def test_fatigue_risk(self):
        tii = TII()
        assert isinstance(tii.fatigue_risk(0.1), str)


class TestVSR:
    """Test Vertical Shear Ratio."""
    
    def test_initialization(self):
        vsr = VSR()
        assert vsr.weight == 0.14
    
    def test_power_law(self):
        vsr = VSR()
        result = vsr.compute(10.0, 100, stability='neutral')
        assert result['vsr'] > 1.0
        assert 'alpha' in result


class TestAOD:
    """Test Aerosol Optical Depth."""
    
    def test_initialization(self):
        aod = AOD()
        assert aod.weight == 0.12
    
    def test_compute_direct(self):
        aod = AOD()
        result = aod.compute(aod_value=0.3)
        assert result == 0.3
    
    def test_solar_reduction(self):
        aod = AOD()
        assert aod.solar_potential_reduction(0.1) > 0.9
        assert aod.solar_potential_reduction(0.5) < 0.7


class TestTHD:
    """Test Thermal Helicity Dynamics."""
    
    def test_initialization(self):
        thd = THD()
        assert thd.weight == 0.10
    
    def test_gust_probability(self):
        thd = THD()
        prob = thd.gust_probability(0.8)
        assert 0 <= prob <= 1
    
    def test_lead_time(self):
        thd = THD()
        lead = thd.estimated_lead_time(0.8)
        assert lead > 0
    
    def test_pre_alert(self):
        thd = THD(config={'threshold': 0.7})
        assert isinstance(thd.pre_alert_triggered(0.8), bool)


class TestPGF:
    """Test Pressure Gradient Force."""
    
    def test_initialization(self):
        pgf = PGF()
        assert pgf.weight == 0.08
    
    def test_compute_direct(self):
        pgf = PGF()
        result = pgf.compute(pressure_gradient=0.003, air_density=1.225)
        expected = 0.003 / 1.225
        assert abs(result - expected) < 1e-6


class TestHCI:
    """Test Humidity-Convection Interaction."""
    
    def test_initialization(self):
        hci = HCI()
        assert hci.weight == 0.07
    
    def test_virtual_temperature(self):
        hci = HCI()
        Tv = hci.virtual_temperature(300, 0.01)
        assert Tv > 300


class TestASI:
    """Test Atmospheric Stability Integration."""
    
    def test_initialization(self):
        asi = ASI()
        assert asi.weight == 0.06
    
    def test_potential_temperature(self):
        asi = ASI()
        # استخدام قيم تزيد مع الارتفاع (الوضع الطبيعي)
        T = np.array([290, 295, 300], dtype=float)  # درجة الحرارة تزيد مع الارتفاع
        z = np.array([0, 100, 200], dtype=float)
        theta = asi.potential_temperature(T, z)
        
        assert len(theta) == 3
        # التحقق أن theta تزيد مع الارتفاع
        assert theta[0] <= theta[1] <= theta[2]
        
        # اختبار إضافي: مع درجة حرارة ثابتة
        T_const = np.array([300, 300, 300], dtype=float)
        theta_const = asi.potential_temperature(T_const, z)
        assert theta_const[0] < theta_const[1] < theta_const[2]
    
    def test_stability_classification(self):
        asi = ASI()
        assert asi.classify_stability(-0.1) == 'UNSTABLE'
        assert asi.classify_stability(0.1) == 'NEUTRAL'
        assert asi.classify_stability(0.5) == 'STABLE'
        assert asi.classify_stability(1.5) == 'VERY_STABLE'


class TestLRC:
    """Test Local Roughness Coefficient."""
    
    def test_initialization(self):
        lrc = LRC()
        assert lrc.weight == 0.05
    
    def test_terrain_roughness(self):
        lrc = LRC()
        assert lrc.roughness_by_terrain('water') < 0.001
        assert lrc.roughness_by_terrain('urban_high') > 1.0
    
    def test_urban_classification(self):
        lrc = LRC()
        assert 'LOW' in lrc.classify_urban_density(0.3)
        assert 'HIGH' in lrc.classify_urban_density(1.5)


if __name__ == '__main__':
    pytest.main([__file__])
