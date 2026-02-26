"""Vertical Shear Ratio (VSR) Parameter.

.. math::
    VSR = \\frac{v(z)}{v(z_{ref})} = \\left(\\frac{z}{z_{ref}}\\right)^\\alpha

where:
    - :math:`\\alpha` is the wind shear exponent
    - :math:`z` is height above ground
    - :math:`z_{ref}` is reference height (typically 10m)
"""

import numpy as np
from typing import Union, Optional, Dict, Any


class VSR:
    """Vertical Shear Ratio parameter.
    
    Weight in AKE index: 14%
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize VSR parameter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.weight = 0.14
        self.z_ref = self.config.get('reference_height', 10.0)  # meters
        self.stability_correction = self.config.get('stability_correction', True)
    
    def compute(self,
               wind_speed_ref: float,
               height: float,
               stability: Optional[str] = 'neutral',
               roughness_length: Optional[float] = None) -> Dict[str, Any]:
        """Compute Vertical Shear Ratio.
        
        Args:
            wind_speed_ref: Wind speed at reference height [m/s]
            height: Target height above ground [m]
            stability: Atmospheric stability ('neutral', 'stable', 'unstable')
            roughness_length: Surface roughness length [m] (for Monin-Obukhov)
            
        Returns:
            Dictionary with VSR results
        """
        if self.stability_correction and roughness_length is not None:
            # Use Monin-Obukhov similarity theory
            vsr = self._compute_monin_obukhov(
                wind_speed_ref, height, stability, roughness_length
            )
            alpha = self._estimate_alpha(stability, roughness_length)
        else:
            # Use simple power law
            alpha = self._estimate_alpha(stability)
            vsr = (height / self.z_ref) ** alpha
        
        wind_speed_at_height = wind_speed_ref * vsr
        
        return {
            'vsr': float(vsr),
            'alpha': float(alpha),
            'wind_speed_at_height': float(wind_speed_at_height),
            'height': height,
            'stability': stability,
            'method': 'monin_obukhov' if roughness_length else 'power_law'
        }
    
    def _estimate_alpha(self, 
                        stability: str,
                        roughness_length: Optional[float] = None) -> float:
        """Estimate wind shear exponent alpha.
        
        Typical values:
        - Open water, neutral: 0.11
        - Open terrain: 0.14-0.20
        - Urban: 0.25-0.40
        - Stable conditions: higher values
        - Unstable conditions: lower values
        """
        base_alpha = {
            'neutral': 0.14,
            'stable': 0.25,
            'unstable': 0.10
        }.get(stability, 0.14)
        
        # Adjust for roughness
        if roughness_length is not None:
            # Rougher terrain increases shear
            roughness_factor = 1.0 + 0.5 * np.log10(1 + roughness_length * 100)
            base_alpha *= roughness_factor
        
        return float(np.clip(base_alpha, 0.05, 0.60))
    
    def _compute_monin_obukhov(self,
                               wind_speed_ref: float,
                               height: float,
                               stability: str,
                               roughness_length: float) -> float:
        """Compute VSR using Monin-Obukhov similarity theory.
        
        v(z) = (u*/κ) [ln(z/z₀) - ψ_m(z/L)]
        """
        kappa = 0.41  # von Kármán constant
        
        # Stability correction functions
        if stability == 'stable':
            psi_m = -5 * (height - self.z_ref) / 100  # Simplified
        elif stability == 'unstable':
            psi_m = 2 * np.log((1 + (1 - 16*(height/100))**0.5) / 2)  # Simplified
        else:  # neutral
            psi_m = 0
        
        # Compute ratio
        vsr = (np.log(height / roughness_length) - psi_m) / \
              (np.log(self.z_ref / roughness_length))
        
        return float(np.clip(vsr, 0.5, 3.0))
    
    def normalize(self, value: float) -> float:
        """Normalize VSR value to [0, 1] range.
        
        Optimal VSR depends on application:
        - For large rotors: moderate shear (α ~ 0.14-0.20)
        - Too high shear causes fatigue loading
        - Too low shear indicates low resource
        """
        # Convert vsr to alpha estimate
        if value <= 1.2:  # Low shear
            return 0.7
        elif value <= 1.5:  # Moderate shear - optimal
            return 1.0
        elif value <= 2.0:  # High shear
            return 0.5
        else:  # Extreme shear
            return 0.2
    
    def rotor_load_factor(self, vsr: float, rotor_diameter: float) -> float:
        """Estimate fatigue loading factor for large rotors.
        
        Args:
            vsr: Vertical Shear Ratio
            rotor_diameter: Rotor diameter [m]
            
        Returns:
            Load factor (1.0 = design load)
        """
        # For 236m rotor, shear creates cyclic loading
        if rotor_diameter > 200:
            # Large rotors more sensitive to shear
            if vsr > 1.8:
                return 1.4  # 40% higher fatigue load
            elif vsr > 1.5:
                return 1.2
        return 1.0
    
    def report(self, **kwargs) -> Dict[str, Any]:
        """Generate full diagnostic report."""
        result = self.compute(**kwargs)
        
        return {
            'parameter': 'VSR',
            'weight': self.weight,
            'value': result['vsr'],
            'alpha': result['alpha'],
            'wind_speed_at_height': result['wind_speed_at_height'],
            'normalized': self.normalize(result['vsr']),
            'load_factor': self.rotor_load_factor(
                result['vsr'], 
                kwargs.get('rotor_diameter', 150)
            ),
            'units': 'dimensionless',
            'description': 'Vertical Shear Ratio',
            'formula': '(z/z_ref)^α',
            'stability': kwargs.get('stability', 'neutral'),
            'config': self.config
        }
