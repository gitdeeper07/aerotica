"""Unit tests for offshore wind farm optimization."""

import pytest
import numpy as np
from pathlib import Path

from aerotica.offshore import (
    OffshoreOptimizer, WakeModel, TurbineLayout, LayoutConfig, OffshoreResource
)
from aerotica.offshore.wake_model import Turbine


class TestWakeModel:
    """Test wake modeling."""
    
    def setup_method(self):
        self.wake_model = WakeModel()
        
        self.t1 = Turbine(
            x=0, y=0,
            hub_height=150,
            rotor_diameter=236,
            rated_power=15000,
            thrust_coefficient=0.8
        )
        
        self.t2 = Turbine(
            x=1000, y=0,
            hub_height=150,
            rotor_diameter=236,
            rated_power=15000,
            thrust_coefficient=0.8
        )
    
    def test_wake_deficit_aligned(self):
        """Test wake deficit for aligned turbines."""
        deficit = self.wake_model.compute_wake_deficit(
            self.t1, self.t2,
            wind_direction=270,  # West wind (aligned)
            ambient_wind_speed=10
        )
        
        assert 0 <= deficit <= 1
        assert deficit < 1.0  # Should have some deficit
    
    def test_wake_deficit_not_aligned(self):
        """Test wake deficit for misaligned turbines."""
        deficit = self.wake_model.compute_wake_deficit(
            self.t1, self.t2,
            wind_direction=0,  # North wind (perpendicular)
            ambient_wind_speed=10
        )
        
        assert deficit == 1.0  # No deficit
    
    def test_multiple_wakes(self):
        """Test multiple wake superposition."""
        turbines = [
            self.t1,
            Turbine(x=1000, y=0, hub_height=150, rotor_diameter=236,
                   rated_power=15000, thrust_coefficient=0.8),
            Turbine(x=2000, y=0, hub_height=150, rotor_diameter=236,
                   rated_power=15000, thrust_coefficient=0.8)
        ]
        
        speeds = self.wake_model.compute_multiple_wakes(
            turbines,
            wind_direction=270,
            ambient_speed=10
        )
        
        assert len(speeds) == 3
        assert speeds[0] == 10  # First turbine no wake
        assert speeds[1] < 10  # Second turbine in wake
        assert speeds[2] < speeds[1]  # Third turbine in multiple wakes
    
    def test_power_calculation(self):
        """Test power calculation."""
        power = self.wake_model.calculate_power(10, self.t1)
        assert power > 0
        
        power_cut_in = self.wake_model.calculate_power(2, self.t1)
        assert power_cut_in == 0
        
        power_cut_out = self.wake_model.calculate_power(26, self.t1)
        assert power_cut_out == 0


class TestTurbineLayout:
    """Test turbine layout optimization."""
    
    def setup_method(self):
        self.config = LayoutConfig(
            n_turbines=9,
            min_spacing=7,
            max_spacing=15,
            boundary_x=(0, 3000),
            boundary_y=(0, 3000),
            rotor_diameter=236,
            hub_height=150,
            rated_power=15000
        )
        
        self.layout = TurbineLayout(self.config)
    
    def test_grid_layout(self):
        """Test grid layout generation."""
        turbines = self.layout.generate_initial_layout('grid')
        
        assert len(turbines) <= self.config.n_turbines
        if turbines:
            # Check spacing
            for i, t1 in enumerate(turbines):
                for j, t2 in enumerate(turbines):
                    if i != j:
                        dist = np.sqrt((t1.x - t2.x)**2 + (t1.y - t2.y)**2)
                        min_allowed = self.config.min_spacing * self.config.rotor_diameter
                        assert dist >= min_allowed or dist == 0
    
    def test_staggered_layout(self):
        """Test staggered layout generation."""
        turbines = self.layout.generate_initial_layout('staggered')
        
        assert len(turbines) <= self.config.n_turbines
    
    def test_random_layout(self):
        """Test random layout generation."""
        turbines = self.layout.generate_initial_layout('random')
        
        assert len(turbines) <= self.config.n_turbines
    
    def test_evaluate_layout(self):
        """Test layout evaluation."""
        self.layout.generate_initial_layout('grid')
        
        wind_rose = {270: 0.5, 0: 0.3, 90: 0.2}
        metrics = self.layout.evaluate_layout(wind_rose)
        
        assert 'aep_mwh' in metrics
        assert 'wake_loss_fraction' in metrics
        assert 'capacity_factor' in metrics
        assert metrics['aep_mwh'] > 0


class TestOffshoreResource:
    """Test offshore resource assessment."""
    
    def setup_method(self):
        self.resource = OffshoreResource(
            latitude=55.0,
            longitude=-3.0,
            water_depth=50
        )
    
    def test_load_data(self):
        """Test data loading."""
        df = self.resource.load_era5_data([2020, 2021])
        
        assert df is not None
        assert 'wind_speed_100m' in df.columns
        assert 'wind_direction' in df.columns
    
    def test_wind_rose(self):
        """Test wind rose computation."""
        self.resource.load_era5_data([2020])
        wind_rose = self.resource.compute_wind_rose()
        
        assert len(wind_rose) > 0
        assert abs(sum(wind_rose.values()) - 1.0) < 0.01
    
    def test_weibull_fit(self):
        """Test Weibull distribution fitting."""
        self.resource.load_era5_data([2020])
        k, c = self.resource.fit_weibull()
        
        assert k > 0
        assert c > 0
    
    def test_extrapolate_height(self):
        """Test height extrapolation."""
        v100 = 10.0
        v150 = self.resource.extrapolate_height(v100, 100, 150)
        
        assert v150 > v100  # Wind increases with height
    
    def test_site_assessment(self):
        """Test site assessment."""
        self.resource.load_era5_data([2020])
        assessment = self.resource.assess_site()
        
        assert 'wind_statistics' in assessment
        assert 'energy_potential' in assessment
        assert assessment['energy_potential']['capacity_factor'] > 0


class TestOffshoreOptimizer:
    """Test offshore optimizer."""
    
    def setup_method(self):
        self.optimizer = OffshoreOptimizer(
            site_latitude=55.0,
            site_longitude=-3.0,
            water_depth=50,
            n_turbines=9,
            area_bounds=((0, 3000), (0, 3000))
        )
    
    def test_setup(self):
        """Test optimizer setup."""
        self.optimizer.setup([2020])
        
        assert self.optimizer.resource.time_series is not None
        assert self.optimizer.resource.wind_rose is not None
    
    def test_create_layout(self):
        """Test layout creation."""
        self.optimizer.setup([2020])
        layout = self.optimizer.create_initial_layout('grid')
        
        assert layout is not None
        assert len(layout.turbines) == 9
    
    def test_evaluate(self):
        """Test layout evaluation."""
        self.optimizer.setup([2020])
        self.optimizer.create_initial_layout('grid')
        
        metrics = self.optimizer.evaluate_current_layout()
        
        assert 'aep_mwh' in metrics
        assert 'site' in metrics
    
    def test_optimization(self):
        """Test optimization (short run)."""
        self.optimizer.setup([2020])
        self.optimizer.create_initial_layout('grid')
        
        results = self.optimizer.optimize_layout(n_iterations=10)
        
        assert 'initial' in results
        assert 'final' in results
        assert 'improvement_percent' in results
    
    def test_financials(self):
        """Test financial calculations."""
        self.optimizer.setup([2020])
        self.optimizer.create_initial_layout('grid')
        
        financials = self.optimizer.calculate_financials()
        
        assert 'npv_euro' in financials
        assert 'irr_percent' in financials
        assert 'lcoe_euro_per_mwh' in financials


if __name__ == '__main__':
    pytest.main([__file__])
