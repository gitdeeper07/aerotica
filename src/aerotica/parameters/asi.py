"""Atmospheric Stability Integration (ASI) Parameter.

.. math::
    Ri_B = \\frac{g}{\\theta} \\cdot \\frac{\\Delta\\theta/\\Delta z}{(\\Delta v/\\Delta z)^2}

where:
    - :math:`Ri_B` is bulk Richardson number
    - :math:`g` is gravity
    - :math:`\\theta` is potential temperature
"""

import numpy as np
from typing import Union, Optional, Dict, Any


class ASI:
    """Atmospheric Stability Integration parameter.
    
    Weight in AKE index: 6%
    Determines whether turbulence is mechanically generated
    or buoyantly suppressed.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ASI parameter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.weight = 0.06
        self.g = 9.81  # gravity [m/s²]
    
    def compute(self,
               temperature_profile: np.ndarray,
               wind_profile: np.ndarray,
               height_profile: np.ndarray) -> Dict[str, Any]:
        """Compute Atmospheric Stability Integration.
        
        Args:
            temperature_profile: Temperature at each level [K]
            wind_profile: Wind speed at each level [m/s]
            height_profile: Height at each level [m]
            
        Returns:
            Dictionary with stability metrics
        """
        # Compute potential temperature
        theta = self.potential_temperature(temperature_profile, height_profile)
        
        # Compute gradients
        dtheta_dz = np.gradient(theta, height_profile)
        dv_dz = np.gradient(wind_profile, height_profile)
        
        # Mean values
        theta_mean = np.mean(theta)
        
        # Richardson number profile (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            Ri = (self.g / theta_mean) * dtheta_dz / (dv_dz**2 + 1e-6)
            Ri = np.where(np.isfinite(Ri), Ri, 0)
        
        # Bulk Richardson number
        Ri_B = float(np.mean(Ri))
        
        # Stability classification
        stability = self.classify_stability(Ri_B)
        
        # Nocturnal jet potential
        jet_potential = self.nocturnal_jet_potential(Ri_B, wind_profile)
        
        return {
            'ri_bulk': Ri_B,
            'stability': stability,
            'jet_potential': jet_potential,
            'ri_profile': Ri.tolist()
        }
    
    def potential_temperature(self, T: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute potential temperature.
        
        θ = T (p0/p)^(R/cp)
        
        Simplified: θ ≈ T + Γ_d * z
        """
        gamma_d = 0.0098  # Dry adiabatic lapse rate [K/m]
        return T + gamma_d * z
    
    def classify_stability(self, Ri: float) -> str:
        """Classify atmospheric stability based on Richardson number."""
        if Ri < 0:
            return 'UNSTABLE'
        elif Ri < 0.25:
            return 'NEUTRAL'
        elif Ri < 1.0:
            return 'STABLE'
        else:
            return 'VERY_STABLE'
    
    def nocturnal_jet_potential(self, Ri: float, wind_profile: np.ndarray) -> str:
        """Assess potential for low-level nocturnal jet."""
        if Ri > 1.0 and len(wind_profile) > 10:
            # Check for wind max below 500m
            if np.argmax(wind_profile) < 10:  # Low-level maximum
                return 'HIGH'
            else:
                return 'MODERATE'
        return 'LOW'
    
    def gust_probability_modifier(self, stability: str) -> float:
        """Modify gust probability based on stability."""
        modifiers = {
            'UNSTABLE': 1.3,    # 30% higher gust probability
            'NEUTRAL': 1.0,
            'STABLE': 0.7,       # 30% lower
            'VERY_STABLE': 0.5    # 50% lower
        }
        return modifiers.get(stability, 1.0)
    
    def normalize(self, value: float) -> float:
        """Normalize ASI value to [0, 1] range."""
        # Optimal stability is neutral (Ri ~ 0.1)
        if value < 0:
            # Unstable: good for convection, moderate score
            return 0.7
        elif value < 0.25:
            # Neutral: optimal
            return 1.0
        elif value < 1.0:
            # Stable: decreasing score
            return 1.0 - (value - 0.25) * 0.4
        else:
            # Very stable: low score
            return 0.3
    
    def report(self, **kwargs) -> Dict[str, Any]:
        """Generate full diagnostic report."""
        # This would normally compute from data
        # Simplified for demonstration
        result = {
            'ri_bulk': 0.3,
            'stability': 'NEUTRAL',
            'jet_potential': 'LOW'
        }
        
        return {
            'parameter': 'ASI',
            'weight': self.weight,
            'value': result['ri_bulk'],
            'normalized': self.normalize(result['ri_bulk']),
            'stability': result['stability'],
            'jet_potential': result['jet_potential'],
            'gust_modifier': self.gust_probability_modifier(result['stability']),
            'units': 'dimensionless',
            'description': 'Atmospheric Stability Integration',
            'formula': 'Ri_B = (g/θ) · (Δθ/Δz)/(Δv/Δz)²',
            'config': self.config
        }
