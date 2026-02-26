"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_wind_data():
    """Generate sample wind time series data."""
    np.random.seed(42)
    n_points = 1000
    
    timestamps = pd.date_range('2025-01-01', periods=n_points, freq='10min')
    
    # Generate realistic wind data
    base_wind = 8 + 2 * np.sin(np.linspace(0, 4*np.pi, n_points))
    gusts = np.random.gamma(2, 1, n_points)
    wind_speed = base_wind + gusts
    
    wind_direction = 180 + 30 * np.sin(np.linspace(0, 4*np.pi, n_points))
    temperature = 15 + 5 * np.sin(np.linspace(0, 4*np.pi, n_points))
    pressure = 1013 + 5 * np.random.randn(n_points)
    humidity = 70 + 10 * np.random.randn(n_points)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'temperature': temperature,
        'pressure': pressure,
        'humidity': humidity
    })


@pytest.fixture
def sample_parameter_values():
    """Sample normalized parameter values."""
    return {
        'KED': 0.83,
        'TII': 0.76,
        'VSR': 0.89,
        'AOD': 0.34,
        'THD': 0.72,
        'PGF': 0.65,
        'HCI': 0.59,
        'ASI': 0.71,
        'LRC': 0.44
    }


@pytest.fixture
def sample_dem_data():
    """Generate sample DEM data."""
    np.random.seed(42)
    dem = np.random.rand(100, 100) * 50
    # Add some buildings
    dem[30:40, 30:40] = 40 + np.random.rand(10, 10) * 10
    dem[60:70, 60:70] = 30 + np.random.rand(10, 10) * 10
    return dem


@pytest.fixture
def sample_turbines():
    """Generate sample turbine data."""
    from aerotica.offshore.wake_model import Turbine
    
    return [
        Turbine(
            x=i*1000,
            y=j*1000,
            hub_height=150,
            rotor_diameter=236,
            rated_power=15000,
            thrust_coefficient=0.8
        )
        for i in range(3)
        for j in range(3)
    ]


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_config_file(temp_output_dir):
    """Create a mock configuration file."""
    config = {
        'server': {'host': 'localhost', 'port': 8000},
        'parameters': {'ked': {'weight': 0.22}},
        'alerts': {'enabled': True}
    }
    
    import yaml
    config_file = temp_output_dir / 'config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return config_file


@pytest.fixture
def sample_ake_result():
    """Sample AKE computation result."""
    return {
        'site_id': 'test_site',
        'score': 0.724,
        'classification': 'VIABLE',
        'gust_risk': 'ELEVATED',
        'confidence': 0.94,
        'contributions': {
            'KED': {'score': 0.83, 'weight': 0.22, 'contribution': 0.1826},
            'TII': {'score': 0.76, 'weight': 0.16, 'contribution': 0.1216},
        }
    }
