"""Aerosol Optical Depth (AOD) Parameter.

.. math::
    AOD(\\lambda) = -\\ln(I/I_0) / \\cos(\\theta_z)

where:
    - :math:`I` is measured solar irradiance
    - :math:`I_0` is extraterrestrial irradiance
    - :math:`\\theta_z` is solar zenith angle
"""

import numpy as np
from typing import Union, Optional, Dict, Any
from datetime import datetime


class AOD:
    """Aerosol Optical Depth parameter.
    
    Weight in AKE index: 12%
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize AOD parameter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.weight = 0.12
        self.wavelength = self.config.get('wavelength', 550)  # nm
        self.source = self.config.get('source', 'MODIS')
    
    def compute(self,
               aod_value: Optional[float] = None,
               latitude: Optional[float] = None,
               longitude: Optional[float] = None,
               date: Optional[datetime] = None) -> float:
        """Compute or retrieve AOD value.
        
        Args:
            aod_value: Direct AOD value if available
            latitude: Latitude for retrieval
            longitude: Longitude for retrieval
            date: Date for retrieval
            
        Returns:
            AOD value at 550nm
        """
        if aod_value is not None:
            return float(aod_value)
        
        # In production, this would query MODIS or other satellite data
        # Simplified model for demonstration
        if latitude is not None and longitude is not None:
            # Crude estimate based on location
            if -30 < latitude < 30:  # Tropics
                base_aod = 0.3
            elif abs(latitude) > 60:  # Polar
                base_aod = 0.1
            else:  # Mid-latitudes
                base_aod = 0.2
            
            # Adjust for known aerosol sources
            if 20 < longitude < 50 and 15 < latitude < 30:  # Sahara
                base_aod *= 3.0
            elif 100 < longitude < 120 and 20 < latitude < 40:  # East Asia
                base_aod *= 2.5
            
            return float(base_aod)
        
        # Default value
        return 0.15
    
    def solar_potential_reduction(self, aod: float) -> float:
        """Estimate reduction in solar power potential due to aerosols.
        
        Args:
            aod: Aerosol Optical Depth
            
        Returns:
            Reduction factor (1.0 = no reduction, 0.5 = 50% reduction)
        """
        # Simplified model: reduction ~ exp(-AOD)
        return float(np.exp(-aod))
    
    def gust_suppression_factor(self, aod: float, time_of_day: str = 'day') -> float:
        """Estimate suppression of convective gusts by aerosols.
        
        High AOD reduces surface heating, suppressing convective gusts.
        
        Args:
            aod: Aerosol Optical Depth
            time_of_day: 'day' or 'night'
            
        Returns:
            Suppression factor (1.0 = no suppression, 0.6 = 40% reduction)
        """
        if time_of_day == 'night':
            return 1.0  # No effect at night
        
        if aod > 0.5:
            return 0.6  # 40% reduction in gust probability
        elif aod > 0.3:
            return 0.8
        elif aod > 0.15:
            return 0.95
        else:
            return 1.0
    
    def air_density_correction(self, aod: float) -> float:
        """Estimate air density modification due to aerosols.
        
        Args:
            aod: Aerosol Optical Depth
            
        Returns:
            Density correction factor
        """
        # Minor effect, but included for completeness
        return 1.0 + 0.01 * aod
    
    def normalize(self, value: float) -> float:
        """Normalize AOD value to [0, 1] range.
        
        Lower AOD is better for solar and gust prediction.
        """
        AOD_MIN = 0.0
        AOD_MAX = 1.0
        
        # Invert so low AOD gives high score
        normalized = 1.0 - np.clip((value - AOD_MIN) / (AOD_MAX - AOD_MIN), 0, 1)
        
        return float(normalized)
    
    def classify(self, aod: float) -> str:
        """Classify AOD level."""
        if aod < 0.05:
            return 'PRISTINE'
        elif aod < 0.15:
            return 'CLEAN'
        elif aod < 0.3:
            return 'MODERATE'
        elif aod < 0.5:
            return 'HAZY'
        else:
            return 'SEVERE'
    
    def report(self, **kwargs) -> Dict[str, Any]:
        """Generate full diagnostic report."""
        aod = self.compute(**kwargs)
        
        return {
            'parameter': 'AOD',
            'weight': self.weight,
            'value': aod,
            'normalized': self.normalize(aod),
            'classification': self.classify(aod),
            'solar_reduction': self.solar_potential_reduction(aod),
            'gust_suppression': self.gust_suppression_factor(aod),
            'units': 'dimensionless',
            'wavelength_nm': self.wavelength,
            'source': self.source,
            'description': 'Aerosol Optical Depth',
            'config': self.config
        }
