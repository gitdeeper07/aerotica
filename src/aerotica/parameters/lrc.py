"""Local Roughness Coefficient (LRC) Parameter.

.. math::
    z_0 = f(\\sigma_h, \\lambda_f, \\lambda_p)

where:
    - :math:`\\sigma_h` is building height standard deviation
    - :math:`\\lambda_f` is frontal area index
    - :math:`\\lambda_p` is plan area density
"""

import numpy as np
from typing import Union, Optional, Dict, Any


class LRC:
    """Local Roughness Coefficient parameter.
    
    Weight in AKE index: 5%
    Derived from high-resolution LiDAR topographic surveys.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LRC parameter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.weight = 0.05
        self.resolution = self.config.get('resolution', 2.0)  # meters
    
    def compute(self,
               building_heights: Optional[np.ndarray] = None,
               frontal_area: Optional[float] = None,
               plan_area: Optional[float] = None,
               total_area: Optional[float] = None,
               terrain_type: Optional[str] = None) -> Dict[str, Any]:
        """Compute Local Roughness Coefficient.
        
        Args:
            building_heights: Array of building heights [m]
            frontal_area: Total frontal area of obstacles [m²]
            plan_area: Total plan area of obstacles [m²]
            total_area: Total ground area [m²]
            terrain_type: Type of terrain ('urban', 'suburban', 'rural')
            
        Returns:
            Dictionary with roughness parameters
        """
        if terrain_type is not None:
            z0 = self.roughness_by_terrain(terrain_type)
            method = 'terrain_classification'
        elif building_heights is not None:
            z0 = self.compute_macdonald(building_heights)
            method = 'macdonald_1998'
        elif all(v is not None for v in [frontal_area, plan_area, total_area]):
            z0 = self.compute_morphological(frontal_area, plan_area, total_area)
            method = 'morphological'
        else:
            # Default urban value
            z0 = 0.5
            method = 'default'
        
        # Urban subclassification
        if terrain_type == 'urban' or z0 > 0.3:
            urban_class = self.classify_urban_density(z0)
        else:
            urban_class = 'N/A'
        
        return {
            'z0': float(z0),
            'method': method,
            'urban_class': urban_class,
            'roughness_class': self.roughness_class(z0)
        }
    
    def compute_macdonald(self, building_heights: np.ndarray) -> float:
        """Compute roughness using Macdonald et al. (1998) method.
        
        z0 = (h - d) exp(-κ / √(0.5 λ_f))
        """
        kappa = 0.41  # von Kármán constant
        
        h_mean = np.mean(building_heights)
        h_std = np.std(building_heights)
        
        # Simplified: assume λ_f ≈ h_std/h_mean
        lambda_f = min(h_std / (h_mean + 1e-6), 0.3)
        
        # Displacement height
        d = 0.7 * h_mean
        
        # Roughness length
        z0 = (h_mean - d) * np.exp(-kappa / np.sqrt(0.5 * lambda_f))
        
        return float(np.clip(z0, 0.01, 5.0))
    
    def compute_morphological(self,
                             frontal_area: float,
                             plan_area: float,
                             total_area: float) -> float:
        """Compute roughness from morphological parameters."""
        lambda_f = frontal_area / total_area
        lambda_p = plan_area / total_area
        
        # Empirical relationship
        z0 = 0.1 * lambda_f + 0.05 * lambda_p
        
        return float(np.clip(z0, 0.01, 2.0))
    
    def roughness_by_terrain(self, terrain_type: str) -> float:
        """Estimate roughness length by terrain type."""
        roughness_map = {
            'water': 0.0002,
            'ice': 0.0001,
            'sand': 0.0003,
            'grass': 0.03,
            'cropland': 0.1,
            'suburban': 0.4,
            'urban_low': 0.8,
            'urban_medium': 1.5,
            'urban_high': 2.5,
            'forest': 1.0,
            'mountain': 0.5
        }
        return roughness_map.get(terrain_type, 0.5)
    
    def classify_urban_density(self, z0: float) -> str:
        """Classify urban density based on roughness length."""
        if z0 < 0.5:
            return 'LOW_DENSITY'
        elif z0 < 1.0:
            return 'MEDIUM_DENSITY'
        elif z0 < 2.0:
            return 'HIGH_DENSITY'
        else:
            return 'VERY_HIGH_DENSITY'
    
    def roughness_class(self, z0: float) -> str:
        """Davenport roughness classification."""
        if z0 < 0.0002:
            return '0 - Sea'
        elif z0 < 0.03:
            return '1 - Smooth'
        elif z0 < 0.1:
            return '2 - Open'
        elif z0 < 0.25:
            return '3 - Rough'
        elif z0 < 0.5:
            return '4 - Very Rough'
        elif z0 < 1.0:
            return '5 - Closed'
        else:
            return '6 - Chaotic'
    
    def wind_speedup_factor(self, z0: float, height: float) -> float:
        """Estimate wind speedup factor at given height."""
        # Simplified: higher roughness = more acceleration above canopy
        if height < 50:
            return 1.0 + 0.2 * np.log(1 + z0)
        else:
            return 1.0 + 0.1 * np.log(1 + z0)
    
    def normalize(self, value: float) -> float:
        """Normalize roughness length to [0, 1] range.
        
        Lower roughness is better for wind energy.
        """
        Z0_MIN = 0.0001
        Z0_MAX = 2.5
        
        # Invert so low roughness gives high score
        normalized = 1.0 - np.clip((np.log10(value) - np.log10(Z0_MIN)) / 
                                   (np.log10(Z0_MAX) - np.log10(Z0_MIN)), 0, 1)
        
        return float(normalized)
    
    def report(self, **kwargs) -> Dict[str, Any]:
        """Generate full diagnostic report."""
        result = self.compute(**kwargs)
        
        return {
            'parameter': 'LRC',
            'weight': self.weight,
            'value': result['z0'],
            'normalized': self.normalize(result['z0']),
            'roughness_class': result['roughness_class'],
            'urban_class': result['urban_class'],
            'method': result['method'],
            'units': 'm',
            'description': 'Local Roughness Coefficient',
            'formula': 'z₀ from morphological parameters',
            'config': self.config
        }
