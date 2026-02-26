"""Offshore Wind Resource Assessment Module."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path


class OffshoreResource:
    """Offshore wind resource assessment."""
    
    def __init__(self,
                 latitude: float,
                 longitude: float,
                 water_depth: float,
                 data_source: str = 'era5'):
        """Initialize offshore resource assessment.
        
        Args:
            latitude: Site latitude [degrees]
            longitude: Site longitude [degrees]
            water_depth: Water depth [m]
            data_source: Data source ('era5', 'lidar', 'measurement')
        """
        self.latitude = latitude
        self.longitude = longitude
        self.water_depth = water_depth
        self.data_source = data_source
        
        # Wind rose data
        self.wind_rose: Dict[float, float] = {}
        
        # Time series data
        self.time_series: Optional[pd.DataFrame] = None
        
        # Weibull parameters
        self.weibull_k: float = 2.0
        self.weibull_c: float = 10.0
    
    def load_era5_data(self, years: List[int]) -> pd.DataFrame:
        """Load ERA5 reanalysis data.
        
        Args:
            years: List of years to load
            
        Returns:
            DataFrame with wind data
        """
        # In production, this would download from CDS API
        # Generate synthetic data for demonstration
        np.random.seed(42)
        
        n_hours = len(years) * 8760
        timestamps = pd.date_range(
            start=datetime(years[0], 1, 1),
            periods=n_hours,
            freq='h'
        )
        
        # Generate wind speed (Weibull distribution)
        wind_speed = np.random.weibull(self.weibull_k, n_hours) * self.weibull_c
        
        # Generate wind direction (circular distribution)
        wind_direction = np.random.vonmises(
            np.radians(270), 2.0, n_hours
        )
        wind_direction = np.degrees(wind_direction) % 360
        
        # Generate temperature and pressure
        temperature = 10 + 5 * np.sin(2 * np.pi * np.arange(n_hours) / (365 * 24))
        pressure = 1013 + 10 * np.random.randn(n_hours)
        
        self.time_series = pd.DataFrame({
            'timestamp': timestamps,
            'wind_speed_100m': wind_speed,
            'wind_direction': wind_direction,
            'temperature_2m': temperature,
            'pressure_msl': pressure
        })
        
        return self.time_series
    
    def compute_wind_rose(self, n_directions: int = 16) -> Dict[float, float]:
        """Compute wind rose from time series data.
        
        Args:
            n_directions: Number of direction bins
            
        Returns:
            Dictionary {direction: frequency}
        """
        if self.time_series is None:
            raise ValueError("No data loaded. Call load_era5_data() first.")
        
        # Create direction bins
        bin_width = 360 / n_directions
        bins = np.arange(0, 360 + bin_width, bin_width)
        
        # Calculate frequencies
        hist, _ = np.histogram(
            self.time_series['wind_direction'],
            bins=bins,
            density=True
        )
        
        # Store as dictionary
        centers = bins[:-1] + bin_width / 2
        self.wind_rose = {
            float(center): float(freq)
            for center, freq in zip(centers, hist)
        }
        
        return self.wind_rose
    
    def fit_weibull(self) -> Tuple[float, float]:
        """Fit Weibull distribution to wind speed data.
        
        Returns:
            Tuple of (shape parameter k, scale parameter c)
        """
        if self.time_series is None:
            raise ValueError("No data loaded")
        
        wind_speed = self.time_series['wind_speed_100m'].values
        
        # Method of moments
        mean = np.mean(wind_speed)
        std = np.std(wind_speed)
        
        # Approximate k from mean and std
        self.weibull_k = (std / mean) ** (-1.086)
        self.weibull_c = mean / np.math.gamma(1 + 1/self.weibull_k)
        
        return self.weibull_k, self.weibull_c
    
    def extrapolate_height(self,
                          wind_speed: float,
                          from_height: float,
                          to_height: float,
                          roughness: float = 0.0002) -> float:
        """Extrapolate wind speed to different height using power law.
        
        Args:
            wind_speed: Wind speed at from_height [m/s]
            from_height: Reference height [m]
            to_height: Target height [m]
            roughness: Surface roughness length [m]
            
        Returns:
            Wind speed at target height [m/s]
        """
        # Power law exponent for offshore
        alpha = 0.11  # Open water
        
        return wind_speed * (to_height / from_height) ** alpha
    
    def compute_power_density(self,
                            wind_speed: float,
                            air_density: float = 1.225) -> float:
        """Compute wind power density [W/m²].
        
        P = 0.5 * ρ * v³
        """
        return 0.5 * air_density * wind_speed ** 3
    
    def assess_site(self) -> Dict[str, Any]:
        """Perform comprehensive site assessment.
        
        Returns:
            Dictionary with site assessment results
        """
        if self.time_series is None:
            self.load_era5_data([2020, 2021, 2022, 2023, 2024])
        
        # Compute wind rose
        wind_rose = self.compute_wind_rose()
        
        # Fit Weibull
        k, c = self.fit_weibull()
        
        # Calculate statistics
        wind_speed = self.time_series['wind_speed_100m']
        
        # Calculate power density at hub height (150m)
        wind_speed_150m = self.extrapolate_height(
            wind_speed.values, 100, 150
        )
        power_density = self.compute_power_density(wind_speed_150m)
        
        # Calculate energy yield for a 15MW turbine
        capacity_factor = self._estimate_capacity_factor(wind_speed_150m)
        annual_energy = 15 * 8760 * capacity_factor  # MWh
        
        return {
            'location': {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'water_depth_m': self.water_depth
            },
            'wind_statistics': {
                'mean_speed_100m_ms': float(wind_speed.mean()),
                'max_speed_100m_ms': float(wind_speed.max()),
                'weibull_k': float(k),
                'weibull_c': float(c),
                'power_density_150m_wm2': float(power_density.mean())
            },
            'wind_rose': wind_rose,
            'energy_potential': {
                'capacity_factor': float(capacity_factor),
                'annual_energy_gwh': float(annual_energy / 1000),
                'full_load_hours': float(capacity_factor * 8760)
            },
            'data_source': self.data_source,
            'data_period': f"{self.time_series['timestamp'].min()} to {self.time_series['timestamp'].max()}"
        }
    
    def _estimate_capacity_factor(self, wind_speed: np.ndarray) -> float:
        """Estimate capacity factor from wind speed distribution."""
        rated_speed = 12.0
        cut_in = 3.0
        cut_out = 25.0
        
        # Simplified power curve
        power = np.zeros_like(wind_speed)
        
        mask_cubic = (wind_speed >= cut_in) & (wind_speed < rated_speed)
        mask_rated = (wind_speed >= rated_speed) & (wind_speed < cut_out)
        
        power[mask_cubic] = (wind_speed[mask_cubic] / rated_speed) ** 3
        power[mask_rated] = 1.0
        
        return float(np.mean(power))
    
    def generate_wind_rose_plot(self, output_path: Path):
        """Generate wind rose plot.
        
        Args:
            output_path: Output file path
        """
        import matplotlib.pyplot as plt
        
        if not self.wind_rose:
            self.compute_wind_rose()
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
        
        directions = np.radians(list(self.wind_rose.keys()))
        frequencies = list(self.wind_rose.values())
        
        ax.bar(directions, frequencies, width=np.radians(22.5), alpha=0.7)
        ax.set_title('Wind Rose - Offshore Site')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Wind rose saved to {output_path}")
    
    def generate_time_series_plot(self, output_path: Path, n_days: int = 30):
        """Generate time series plot.
        
        Args:
            output_path: Output file path
            n_days: Number of days to plot
        """
        import matplotlib.pyplot as plt
        
        if self.time_series is None:
            raise ValueError("No data loaded")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Last n_days
        data = self.time_series.tail(n_days * 24)
        
        ax1.plot(data['timestamp'], data['wind_speed_100m'])
        ax1.set_ylabel('Wind Speed [m/s]')
        ax1.set_title(f'Wind Speed Time Series (Last {n_days} days)')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(data['timestamp'], data['wind_direction'], '.', markersize=1)
        ax2.set_ylabel('Wind Direction [°]')
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Time series plot saved to {output_path}")
