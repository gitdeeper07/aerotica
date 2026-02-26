"""Radiosonde profile data preprocessing."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import xarray as xr


class RadiosondeProcessor:
    """Process radiosonde profile data."""
    
    def __init__(self):
        """Initialize processor."""
        self.profiles = []
        self.current_profile = None
    
    def read_file(self, file_path: Path) -> Dict:
        """Read radiosonde file.
        
        Supports standard radiosonde formats.
        
        Args:
            file_path: Path to radiosonde file
            
        Returns:
            Dictionary with profile data
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.nc':
            return self._read_netcdf(file_path)
        elif file_path.suffix == '.csv':
            return self._read_csv(file_path)
        elif file_path.suffix == '.txt':
            return self._read_text(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
    
    def _read_netcdf(self, file_path: Path) -> Dict:
        """Read NetCDF format."""
        ds = xr.open_dataset(file_path)
        
        profile = {
            'time': ds.time.values if 'time' in ds else None,
            'levels': [],
            'pressure': [],
            'height': [],
            'temperature': [],
            'dewpoint': [],
            'wind_speed': [],
            'wind_direction': [],
            'humidity': []
        }
        
        # Extract data
        for var in ['pressure', 'height', 'temperature', 'dewpoint',
                   'wind_speed', 'wind_direction', 'humidity']:
            if var in ds:
                profile[var] = ds[var].values
        
        # Get levels
        if 'level' in ds:
            profile['levels'] = ds.level.values
        elif 'pressure' in ds:
            profile['levels'] = ds.pressure.values
        
        return profile
    
    def _read_csv(self, file_path: Path) -> Dict:
        """Read CSV format."""
        df = pd.read_csv(file_path)
        
        profile = {
            'levels': df.index.values,
            'pressure': df['pressure'].values if 'pressure' in df else None,
            'height': df['height'].values if 'height' in df else None,
            'temperature': df['temperature'].values if 'temperature' in df else None,
            'dewpoint': df['dewpoint'].values if 'dewpoint' in df else None,
            'wind_speed': df['wind_speed'].values if 'wind_speed' in df else None,
            'wind_direction': df['wind_direction'].values if 'wind_direction' in df else None,
            'humidity': df['humidity'].values if 'humidity' in df else None
        }
        
        return profile
    
    def _read_text(self, file_path: Path) -> Dict:
        """Read text format (simplified)."""
        profile = {
            'levels': [],
            'pressure': [],
            'height': [],
            'temperature': [],
            'wind_speed': [],
            'wind_direction': []
        }
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        profile['levels'].append(float(parts[0]))
                        profile['pressure'].append(float(parts[1]))
                        profile['height'].append(float(parts[2]))
                        profile['temperature'].append(float(parts[3]))
                        
                        if len(parts) >= 6:
                            profile['wind_speed'].append(float(parts[4]))
                            profile['wind_direction'].append(float(parts[5]))
        
        return profile
    
    def interpolate_height(self, 
                          profile: Dict,
                          target_heights: np.ndarray) -> Dict:
        """Interpolate profile to target heights.
        
        Args:
            profile: Profile dictionary
            target_heights: Target height levels [m]
            
        Returns:
            Interpolated profile
        """
        heights = profile['height']
        
        interpolated = {
            'height': target_heights,
            'pressure': np.interp(target_heights, heights, profile['pressure']),
            'temperature': np.interp(target_heights, heights, profile['temperature'])
        }
        
        if profile.get('wind_speed') is not None:
            interpolated['wind_speed'] = np.interp(
                target_heights, heights, profile['wind_speed']
            )
        
        if profile.get('wind_direction') is not None:
            # Circular interpolation for wind direction
            wind_dir = profile['wind_direction']
            wind_dir_rad = np.radians(wind_dir)
            sin_dir = np.sin(wind_dir_rad)
            cos_dir = np.cos(wind_dir_rad)
            
            sin_interp = np.interp(target_heights, heights, sin_dir)
            cos_interp = np.interp(target_heights, heights, cos_dir)
            
            interpolated['wind_direction'] = np.degrees(
                np.arctan2(sin_interp, cos_interp)
            ) % 360
        
        return interpolated
    
    def compute_stability_indices(self, profile: Dict) -> Dict:
        """Compute atmospheric stability indices.
        
        Args:
            profile: Profile dictionary
            
        Returns:
            Stability indices
        """
        indices = {}
        
        # Get levels
        heights = profile['height']
        temps = profile['temperature']
        
        if len(heights) < 2:
            return indices
        
        # Lapse rate
        lapse_rate = -(temps[-1] - temps[0]) / (heights[-1] - heights[0]) * 1000
        
        indices['lapse_rate'] = float(lapse_rate)  # K/km
        
        # Stability classification
        if lapse_rate > 9:
            indices['stability'] = 'UNSTABLE'
        elif lapse_rate > 6:
            indices['stability'] = 'NEUTRAL'
        elif lapse_rate > 3:
            indices['stability'] = 'STABLE'
        else:
            indices['stability'] = 'VERY_STABLE'
        
        # Inversion detection
        inversion = False
        inversion_height = None
        inversion_strength = None
        
        for i in range(1, len(heights)):
            if temps[i] > temps[i-1]:
                inversion = True
                inversion_height = heights[i-1]
                inversion_strength = temps[i] - temps[i-1]
                break
        
        indices['inversion'] = inversion
        indices['inversion_height'] = inversion_height
        indices['inversion_strength'] = inversion_strength
        
        return indices
    
    def to_dataframe(self, profile: Dict) -> pd.DataFrame:
        """Convert profile to DataFrame."""
        df = pd.DataFrame(profile)
        
        # Set levels as index
        if 'levels' in df.columns:
            df = df.set_index('levels')
        
        return df
    
    def plot_profile(self, profile: Dict, output_path: Optional[Path] = None):
        """Plot radiosonde profile.
        
        Args:
            profile: Profile dictionary
            output_path: Optional output path for plot
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        
        # Temperature profile
        ax1.plot(profile['temperature'], profile['height'], 'b-', label='Temperature')
        ax1.set_xlabel('Temperature [°C]')
        ax1.set_ylabel('Height [m]')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Higher heights at top
        ax1.legend()
        
        # Wind profile
        if profile.get('wind_speed') is not None:
            ax2.plot(profile['wind_speed'], profile['height'], 'g-', label='Wind Speed')
            ax2.set_xlabel('Wind Speed [m/s]')
            ax2.set_ylabel('Height [m]')
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()
            ax2.legend()
        
        plt.suptitle('Radiosonde Profile')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"RadiosondeProcessor(profiles={len(self.profiles)})"
