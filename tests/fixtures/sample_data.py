"""Sample data for testing."""

import numpy as np
import pandas as pd
from pathlib import Path


def create_sample_wind_data(n_points: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create sample wind time series data.
    
    Args:
        n_points: Number of data points
        seed: Random seed
        
    Returns:
        DataFrame with wind data
    """
    np.random.seed(seed)
    
    timestamps = pd.date_range('2025-01-01', periods=n_points, freq='10min')
    
    # Generate realistic wind data with diurnal cycle
    hour_of_day = timestamps.hour
    diurnal = 2 * np.sin(2 * np.pi * hour_of_day / 24)
    
    # Base wind with turbulence
    t = np.linspace(0, 4*np.pi, n_points)
    wind_speed = 8 + 2 * np.sin(t) + diurnal + np.random.randn(n_points) * 1.5
    
    # Add occasional gusts
    gust_indices = np.random.choice(n_points, size=n_points//20, replace=False)
    for idx in gust_indices:
        wind_speed[idx:idx+5] += 8 * np.exp(-np.arange(5))
    
    # Wind direction (prevailing westerly with variability)
    wind_direction = 270 + 30 * np.sin(t/2) + np.random.randn(n_points) * 10
    wind_direction = wind_direction % 360
    
    # Temperature with diurnal cycle
    temperature = 15 + 5 * np.sin(2 * np.pi * timestamps.hour / 24) + np.random.randn(n_points)
    
    # Pressure
    pressure = 1013 + 5 * np.random.randn(n_points)
    
    # Humidity
    humidity = 70 + 10 * np.random.randn(n_points)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'temperature': temperature,
        'pressure': pressure,
        'humidity': humidity
    })


def create_sample_parameter_values(seed: int = 42) -> dict:
    """Create sample normalized parameter values.
    
    Args:
        seed: Random seed
        
    Returns:
        Dictionary with parameter values
    """
    np.random.seed(seed)
    
    return {
        'KED': np.clip(np.random.normal(0.7, 0.15), 0.2, 0.95),
        'TII': np.clip(np.random.normal(0.6, 0.1), 0.2, 0.9),
        'VSR': np.clip(np.random.normal(0.75, 0.1), 0.3, 0.95),
        'AOD': np.clip(np.random.normal(0.4, 0.15), 0.1, 0.8),
        'THD': np.clip(np.random.normal(0.65, 0.1), 0.2, 0.9),
        'PGF': np.clip(np.random.normal(0.6, 0.1), 0.2, 0.9),
        'HCI': np.clip(np.random.normal(0.55, 0.1), 0.2, 0.9),
        'ASI': np.clip(np.random.normal(0.65, 0.1), 0.2, 0.9),
        'LRC': np.clip(np.random.normal(0.45, 0.1), 0.1, 0.8)
    }


def create_sample_dem(size: int = 100, seed: int = 42) -> np.ndarray:
    """Create sample DEM data.
    
    Args:
        size: Size of DEM (size x size)
        seed: Random seed
        
    Returns:
        DEM array
    """
    np.random.seed(seed)
    
    # Base terrain
    dem = np.random.randn(size, size) * 5
    
    # Add some buildings
    for _ in range(10):
        x = np.random.randint(10, size-10)
        y = np.random.randint(10, size-10)
        w = np.random.randint(5, 15)
        h = np.random.randint(20, 60)
        dem[x:x+w, y:y+w] += h * np.exp(-np.arange(w)[:, None]**2 / (w/2)**2)
    
    return dem


def create_sample_ake_result() -> dict:
    """Create sample AKE result."""
    return {
        'site_id': 'test_site',
        'climate_zone': 'temperate',
        'site_type': 'test',
        'score': 0.724,
        'classification': 'VIABLE',
        'gust_risk': 'MODERATE',
        'confidence': 0.94,
        'contributions': {
            'KED': {'score': 0.83, 'weight': 0.22, 'contribution': 0.1826},
            'TII': {'score': 0.76, 'weight': 0.16, 'contribution': 0.1216},
            'VSR': {'score': 0.89, 'weight': 0.14, 'contribution': 0.1246},
            'AOD': {'score': 0.34, 'weight': 0.12, 'contribution': 0.0408},
            'THD': {'score': 0.72, 'weight': 0.10, 'contribution': 0.0720},
            'PGF': {'score': 0.65, 'weight': 0.08, 'contribution': 0.0520},
            'HCI': {'score': 0.59, 'weight': 0.07, 'contribution': 0.0413},
            'ASI': {'score': 0.71, 'weight': 0.06, 'contribution': 0.0426},
            'LRC': {'score': 0.44, 'weight': 0.05, 'contribution': 0.0220}
        },
        'missing_parameters': [],
        'weights_used': 'default'
    }


def create_sample_turbine_data(n_turbines: int = 9) -> list:
    """Create sample turbine data.
    
    Args:
        n_turbines: Number of turbines
        
    Returns:
        List of turbine dictionaries
    """
    from aerotica.offshore.wake_model import Turbine
    
    side = int(np.sqrt(n_turbines))
    spacing = 1000
    
    turbines = []
    for i in range(side):
        for j in range(side):
            if len(turbines) < n_turbines:
                turbines.append(Turbine(
                    x=i * spacing,
                    y=j * spacing,
                    hub_height=150,
                    rotor_diameter=236,
                    rated_power=15000,
                    thrust_coefficient=0.8
                ))
    
    return turbines
