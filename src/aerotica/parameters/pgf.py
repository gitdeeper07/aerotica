"""Pressure Gradient Force (PGF) Parameter.

.. math::
    PGF = -\\frac{1}{\\rho}\\nabla p \\quad [m/s^2]

where:
    - :math:`\\rho` is air density
    - :math:`\\nabla p` is pressure gradient
"""

import numpy as np
from typing import Union, Optional, Dict, Any


class PGF:
    """Pressure Gradient Force parameter.
    
    Weight in AKE index: 8%
    Primary kinematic driver of atmospheric motion.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize PGF parameter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.weight = 0.08
    
    def compute(self,
               pressure_gradient: Optional[float] = None,
               pressure_field: Optional[np.ndarray] = None,
               dx: Optional[float] = None,
               air_density: float = 1.225) -> float:
        """Compute Pressure Gradient Force.
        
        Args:
            pressure_gradient: Direct pressure gradient value [Pa/m]
            pressure_field: 2D pressure field [Pa]
            dx: Grid spacing [m]
            air_density: Air density [kg/m³]
            
        Returns:
            PGF magnitude [m/s²]
        """
        if pressure_gradient is not None:
            pgf = pressure_gradient / air_density
            return float(pgf)
        
        if pressure_field is not None and dx is not None:
            # Compute gradient numerically
            dp_dx = np.gradient(pressure_field, dx, axis=1)
            dp_dy = np.gradient(pressure_field, dx, axis=0)
            grad_mag = np.sqrt(dp_dx**2 + dp_dy**2).mean()
            pgf = grad_mag / air_density
            return float(pgf)
        
        # Default value (moderate gradient)
        return 0.003  # m/s²
    
    def geostrophic_wind(self, pgf: float, latitude: float) -> float:
        """Compute geostrophic wind speed.
        
        v_g = PGF / f, where f is Coriolis parameter
        
        Args:
            pgf: Pressure Gradient Force [m/s²]
            latitude: Latitude [degrees]
            
        Returns:
            Geostrophic wind speed [m/s]
        """
        omega = 7.2921e-5  # Earth's angular velocity [rad/s]
        f = 2 * omega * np.sin(np.radians(latitude))
        
        if abs(f) < 1e-6:  # Near equator
            return pgf * 1000  # Approximate
        else:
            return pgf / f
    
    def gradient_correction(self, pgf: float, curvature_radius: float) -> float:
        """Apply curvature correction for gradient wind.
        
        Args:
            pgf: Pressure Gradient Force
            curvature_radius: Radius of curvature [m]
            
        Returns:
            Corrected wind speed
        """
        # Simplified gradient wind correction
        if curvature_radius > 0:
            return pgf * (1 + pgf / curvature_radius)
        return pgf
    
    def normalize(self, value: float) -> float:
        """Normalize PGF value to [0, 1] range."""
        PGF_MIN = 0.0005  # Very weak gradient
        PGF_MAX = 0.01    # Very strong gradient (hurricane)
        
        normalized = np.clip((value - PGF_MIN) / (PGF_MAX - PGF_MIN), 0, 1)
        
        return float(normalized)
    
    def intensity_category(self, pgf: float) -> str:
        """Categorize pressure gradient intensity."""
        if pgf < 0.001:
            return 'WEAK'
        elif pgf < 0.003:
            return 'MODERATE'
        elif pgf < 0.006:
            return 'STRONG'
        else:
            return 'EXTREME'
    
    def report(self, **kwargs) -> Dict[str, Any]:
        """Generate full diagnostic report."""
        pgf = self.compute(**kwargs)
        latitude = kwargs.get('latitude', 45.0)
        
        return {
            'parameter': 'PGF',
            'weight': self.weight,
            'value': pgf,
            'normalized': self.normalize(pgf),
            'geostrophic_wind': self.geostrophic_wind(pgf, latitude),
            'intensity': self.intensity_category(pgf),
            'units': 'm/s²',
            'description': 'Pressure Gradient Force',
            'formula': '-(1/ρ)∇p',
            'config': self.config
        }
