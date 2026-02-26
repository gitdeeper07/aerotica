"""MODIS satellite data processor."""

import numpy as np
import xarray as xr
from typing import Optional, Dict, List, Union
from pathlib import Path
from datetime import datetime, timedelta


class MODISProcessor:
    """Process MODIS satellite data for AOD and atmospheric parameters."""
    
    def __init__(self):
        """Initialize MODIS processor."""
        self.products = []
    
    def read_hdf(self, file_path: Path) -> xr.Dataset:
        """Read MODIS HDF file.
        
        Args:
            file_path: Path to HDF file
            
        Returns:
            xarray Dataset with MODIS data
        """
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            # Extract AOD data (simplified)
            data = {}
            
            # Common MODIS AOD datasets
            aod_paths = [
                '/MODIS_Grid_Daily_1km_2D/Aerosol_Optical_Depth_Land',
                'Optical_Depth_Land_And_Ocean',
                'Aerosol_Optical_Depth'
            ]
            
            for path in aod_paths:
                try:
                    # Navigate through groups
                    parts = path.split('/')
                    current = f
                    for part in parts:
                        current = current[part]
                    data['aod'] = current[()]
                    break
                except (KeyError, ValueError):
                    continue
            
            # Extract coordinates
            try:
                lat = f['/MOD_Grid_Daily_1km_2D/lat'][()]
                lon = f['/MOD_Grid_Daily_1km_2D/lon'][()]
                data['lat'] = lat
                data['lon'] = lon
            except KeyError:
                # Create dummy coordinates
                if 'aod' in data:
                    ny, nx = data['aod'].shape
                    data['lat'] = np.linspace(-90, 90, ny)
                    data['lon'] = np.linspace(-180, 180, nx)
            
            # Create xarray dataset
            ds = xr.Dataset(
                {'aod': (['y', 'x'], data.get('aod', np.zeros((10, 10))))},
                coords={
                    'lat': (['y', 'x'], data.get('lat', np.zeros((10, 10)))),
                    'lon': (['y', 'x'], data.get('lon', np.zeros((10, 10))))
                }
            )
            
            return ds
    
    def extract_aod(self, 
                   modis_data: xr.Dataset,
                   lat: float,
                   lon: float,
                   radius_km: float = 10.0) -> float:
        """Extract AOD at specific location.
        
        Args:
            modis_data: MODIS dataset
            lat: Target latitude
            lon: Target longitude
            radius_km: Search radius in km
            
        Returns:
            AOD value at location
        """
        if 'aod' not in modis_data:
            return 0.15  # Default value
        
        # Calculate distances
        if 'lat' in modis_data.coords and 'lon' in modis_data.coords:
            lat_grid = modis_data.lat.values
            lon_grid = modis_data.lon.values
            
            # Simple distance calculation (ignoring Earth curvature)
            dist = np.sqrt((lat_grid - lat)**2 + (lon_grid - lon)**2)
            dist_km = dist * 111  # Rough conversion to km
            
            # Get points within radius
            mask = dist_km <= radius_km
            
            if np.any(mask):
                aod_values = modis_data.aod.values[mask]
                return float(np.nanmean(aod_values))
        
        return 0.15
    
    def classify_aerosol_type(self, aod: float, angstrom: float) -> str:
        """Classify aerosol type based on AOD and Angstrom exponent.
        
        Args:
            aod: Aerosol Optical Depth
            angstrom: Angstrom exponent
            
        Returns:
            Aerosol type classification
        """
        if aod < 0.1:
            return 'CLEAN'
        elif aod < 0.2:
            if angstrom > 1.5:
                return 'URBAN/INDUSTRIAL'
            else:
                return 'MARITIME'
        elif aod < 0.5:
            if angstrom > 1.5:
                return 'BIOMASS_BURNING'
            else:
                return 'DUST'
        else:
            if angstrom < 1.0:
                return 'HEAVY_DUST'
            else:
                return 'POLLUTED'
    
    def get_aod_climatology(self, 
                           lat: float,
                           lon: float,
                           month: int) -> float:
        """Get climatological AOD for location and month.
        
        Args:
            lat: Latitude
            lon: Longitude
            month: Month (1-12)
            
        Returns:
            Climatological AOD
        """
        # Simplified climatology based on location
        if -30 < lat < 30:  # Tropics
            base = 0.25
        elif abs(lat) > 60:  # Polar
            base = 0.08
        else:  # Mid-latitudes
            base = 0.15
        
        # Seasonal variation
        seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * (month - 3) / 12)
        
        # Regional adjustments
        if 20 < lon < 50 and 15 < lat < 30:  # Sahara
            base = 0.4
        elif 100 < lon < 120 and 20 < lat < 40:  # East Asia
            base = 0.35
        
        return float(base * seasonal_factor)
    
    def process_daily_product(self, 
                            file_paths: List[Path],
                            output_path: Path):
        """Process multiple daily files into composite.
        
        Args:
            file_paths: List of daily files
            output_path: Output path for composite
        """
        datasets = []
        
        for file_path in file_paths:
            try:
                ds = self.read_hdf(file_path)
                datasets.append(ds)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if datasets:
            # Combine along time dimension
            combined = xr.concat(datasets, dim='time')
            combined.to_netcdf(output_path)
            print(f"Saved composite to {output_path}")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MODISProcessor(products={len(self.products)})"
