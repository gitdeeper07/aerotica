"""Weather radar data preprocessing."""

import numpy as np
import xarray as xr
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import warnings


class RadarProcessor:
    """Process weather radar data."""
    
    def __init__(self):
        """Initialize radar processor."""
        self.scans = []
        self.current_scan = None
    
    def read_file(self, file_path: Path) -> xr.Dataset:
        """Read radar file.
        
        Supports common radar formats.
        
        Args:
            file_path: Path to radar file
            
        Returns:
            xarray Dataset with radar data
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.nc':
            return self._read_netcdf(file_path)
        elif file_path.suffix in ['.h5', '.hdf5']:
            return self._read_hdf5(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
    
    def _read_netcdf(self, file_path: Path) -> xr.Dataset:
        """Read NetCDF format."""
        return xr.open_dataset(file_path)
    
    def _read_hdf5(self, file_path: Path) -> xr.Dataset:
        """Read HDF5 format (simplified)."""
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            # Extract data (simplified)
            data = {}
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data[key] = f[key][()]
            
            # Create coordinates
            if 'azimuth' in data and 'range' in data:
                ds = xr.Dataset(
                    {'reflectivity': (['azimuth', 'range'], data.get('reflectivity', np.zeros((len(data['azimuth']), len(data['range']))))),
                     'velocity': (['azimuth', 'range'], data.get('velocity', np.zeros((len(data['azimuth']), len(data['range'])))))},
                    coords={
                        'azimuth': data['azimuth'],
                        'range': data['range'],
                        'time': data.get('time', 0)
                    }
                )
                return ds
            else:
                # Return as DataArray
                return xr.DataArray(data)
    
    def extract_gust_signature(self, 
                              radar_data: xr.Dataset,
                              threshold: float = 45.0) -> Dict:
        """Extract gust front signature from radar data.
        
        Args:
            radar_data: Radar dataset
            threshold: Velocity threshold for gust detection [m/s]
            
        Returns:
            Dictionary with gust signature information
        """
        signature = {
            'detected': False,
            'gust_front': None,
            'max_velocity': 0.0,
            'area_km2': 0.0
        }
        
        if 'velocity' not in radar_data:
            return signature
        
        velocity = radar_data['velocity'].values
        
        # Find high velocity regions
        gust_mask = velocity > threshold
        
        if np.any(gust_mask):
            signature['detected'] = True
            signature['max_velocity'] = float(np.max(velocity))
            
            # Calculate area (simplified)
            cell_area = 1.0  # km² per grid cell (would be calculated from resolution)
            signature['area_km2'] = float(np.sum(gust_mask) * cell_area)
            
            # Find gust front (leading edge)
            # Simplified: find max gradient
            grad = np.gradient(velocity)
            gust_front_idx = np.unravel_index(np.argmax(np.abs(grad)), velocity.shape)
            signature['gust_front'] = {
                'azimuth': float(radar_data.azimuth[gust_front_idx[0]]),
                'range': float(radar_data.range[gust_front_idx[1]])
            }
        
        return signature
    
    def compute_vad_wind(self, 
                        radar_data: xr.Dataset,
                        height_levels: List[float]) -> Dict:
        """Compute Velocity Azimuth Display (VAD) wind profile.
        
        Args:
            radar_data: Radar dataset
            height_levels: Height levels for profile [m]
            
        Returns:
            VAD wind profile
        """
        # Simplified VAD implementation
        profile = {
            'heights': height_levels,
            'wind_speed': [],
            'wind_direction': []
        }
        
        if 'velocity' not in radar_data:
            return profile
        
        velocity = radar_data['velocity'].values
        
        for height in height_levels:
            # Find closest range gate
            if 'range' in radar_data.coords:
                range_idx = np.argmin(np.abs(radar_data.range.values - height))
                
                # Average over azimuths at this range
                speed = np.mean(np.abs(velocity[:, range_idx]))
                profile['wind_speed'].append(float(speed))
                
                # Direction from azimuth of maximum (simplified)
                max_az_idx = np.argmax(velocity[:, range_idx])
                if 'azimuth' in radar_data.coords:
                    direction = float(radar_data.azimuth[max_az_idx])
                else:
                    direction = 0.0
                profile['wind_direction'].append(direction)
            else:
                profile['wind_speed'].append(0.0)
                profile['wind_direction'].append(0.0)
        
        return profile
    
    def detect_precipitation(self, 
                           radar_data: xr.Dataset,
                           threshold: float = 10.0) -> np.ndarray:
        """Detect precipitation from reflectivity.
        
        Args:
            radar_data: Radar dataset
            threshold: Reflectivity threshold [dBZ]
            
        Returns:
            Boolean mask of precipitation areas
        """
        if 'reflectivity' not in radar_data:
            return np.zeros_like(radar_data.get('velocity', np.array([0])))
        
        reflectivity = radar_data['reflectivity'].values
        return reflectivity > threshold
    
    def estimate_rain_rate(self, 
                          radar_data: xr.Dataset,
                          a: float = 200.0,
                          b: float = 1.6) -> np.ndarray:
        """Estimate rain rate from reflectivity (Z-R relationship).
        
        Z = a * R^b
        R = (Z / a)^(1/b)
        
        Args:
            radar_data: Radar dataset
            a: Z-R coefficient
            b: Z-R exponent
            
        Returns:
            Rain rate array [mm/h]
        """
        if 'reflectivity' not in radar_data:
            return np.zeros_like(radar_data.get('velocity', np.array([0])))
        
        # Convert dBZ to Z
        reflectivity_dbz = radar_data['reflectivity'].values
        Z = 10 ** (reflectivity_dbz / 10)
        
        # Calculate rain rate
        with np.errstate(divide='ignore', invalid='ignore'):
            R = (Z / a) ** (1 / b)
            R[~np.isfinite(R)] = 0
        
        return R
    
    def to_netcdf(self, data: xr.Dataset, output_path: Path):
        """Save radar data to NetCDF."""
        data.to_netcdf(output_path)
    
    def plot_ppi(self, 
                radar_data: xr.Dataset,
                variable: str = 'reflectivity',
                output_path: Optional[Path] = None):
        """Plot Plan Position Indicator (PPI).
        
        Args:
            radar_data: Radar dataset
            variable: Variable to plot
            output_path: Optional output path
        """
        import matplotlib.pyplot as plt
        
        if variable not in radar_data:
            print(f"Variable {variable} not found")
            return
        
        data = radar_data[variable].values
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert to Cartesian for plotting (simplified)
        if 'azimuth' in radar_data.coords and 'range' in radar_data.coords:
            azimuths = np.radians(radar_data.azimuth.values)
            ranges = radar_data.range.values
            
            X = ranges * np.sin(azimuths[:, np.newaxis])
            Y = ranges * np.cos(azimuths[:, np.newaxis])
            
            im = ax.pcolormesh(X, Y, data, shading='auto', cmap='viridis')
        else:
            im = ax.imshow(data, cmap='viridis', origin='lower')
        
        plt.colorbar(im, ax=ax, label=variable)
        ax.set_xlabel('Distance [km]')
        ax.set_ylabel('Distance [km]')
        ax.set_title(f'Radar PPI - {variable}')
        ax.set_aspect('equal')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"RadarProcessor(scans={len(self.scans)})"
