"""Humidity-Convection Interaction (HCI) Parameter.

Quantifies the role of atmospheric moisture in modifying
the kinetic energy budget through:
1. Virtual temperature effect (density reduction)
2. Latent heat release in convective updrafts
"""

import numpy as np
from typing import Union, Optional, Dict, Any


class HCI:
    """Humidity-Convection Interaction parameter.
    
    Weight in AKE index: 7%
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize HCI parameter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.weight = 0.07
    
    def compute(self,
               specific_humidity: float,
               temperature: float,
               pressure: float,
               cape: Optional[float] = None) -> Dict[str, Any]:
        """Compute HCI components.
        
        Args:
            specific_humidity: Specific humidity [kg/kg]
            temperature: Air temperature [K]
            pressure: Atmospheric pressure [Pa]
            cape: Convective Available Potential Energy [J/kg]
            
        Returns:
            Dictionary with HCI components
        """
        # Virtual temperature effect
        virtual_temp = self.virtual_temperature(temperature, specific_humidity)
        density_ratio = temperature / virtual_temp
        
        # Density reduction factor
        density_reduction = 1.0 - density_ratio
        
        # Latent heat enhancement (if CAPE available)
        if cape is not None:
            latent_enhancement = self.latent_heat_enhancement(cape)
        else:
            latent_enhancement = 1.0
        
        # Combined HCI score
        hci_score = density_reduction * 2 + (latent_enhancement - 1.0) * 5
        hci_score = float(np.clip(hci_score, 0, 1))
        
        return {
            'hci': hci_score,
            'virtual_temperature': float(virtual_temp),
            'density_reduction': float(density_reduction),
            'latent_enhancement': float(latent_enhancement)
        }
    
    def virtual_temperature(self, T: float, q: float) -> float:
        """Compute virtual temperature.
        
        T_v = T(1 + 0.608q)
        """
        return T * (1 + 0.608 * q)
    
    def latent_heat_enhancement(self, cape: float) -> float:
        """Estimate wind enhancement from latent heat release.
        
        Args:
            cape: Convective Available Potential Energy [J/kg]
            
        Returns:
            Enhancement factor (1.0 = no enhancement)
        """
        if cape < 500:
            return 1.0
        elif cape < 1000:
            return 1.05
        elif cape < 2000:
            return 1.10
        elif cape < 3000:
            return 1.15
        else:
            return 1.20
    
    def mixing_ratio_to_specific(self, mixing_ratio: float) -> float:
        """Convert mixing ratio to specific humidity.
        
        q = w / (1 + w)
        """
        return mixing_ratio / (1 + mixing_ratio)
    
    def dewpoint_to_humidity(self, T: float, Td: float, pressure: float) -> float:
        """Estimate specific humidity from dewpoint.
        
        Simplified approximation.
        """
        # Saturation vapor pressure (Tetens formula)
        es = 611 * np.exp(17.67 * (Td - 273.15) / (Td - 29.65))
        
        # Specific humidity
        q = 0.622 * es / pressure
        
        return float(q)
    
    def normalize(self, value: float) -> float:
        """Normalize HCI value to [0, 1] range."""
        return float(np.clip(value, 0, 1))
    
    def report(self, **kwargs) -> Dict[str, Any]:
        """Generate full diagnostic report."""
        result = self.compute(**kwargs)
        
        return {
            'parameter': 'HCI',
            'weight': self.weight,
            'value': result['hci'],
            'normalized': self.normalize(result['hci']),
            'virtual_temperature': result['virtual_temperature'],
            'density_reduction': result['density_reduction'],
            'latent_enhancement': result['latent_enhancement'],
            'units': 'dimensionless',
            'description': 'Humidity-Convection Interaction',
            'config': self.config
        }
