"""AKE Parameter Weights Module.

Bayesian-optimized weights for the nine AKE parameters.
"""

from typing import Dict, Optional
import numpy as np
import json
from pathlib import Path


class AKEWeights:
    """Bayesian-optimized weights for AKE parameters."""
    
    # Default weights from global optimization
    DEFAULT_WEIGHTS = {
        'KED': 0.22,   # Kinetic Energy Density
        'TII': 0.16,   # Turbulence Intensity Index
        'VSR': 0.14,   # Vertical Shear Ratio
        'AOD': 0.12,   # Aerosol Optical Depth
        'THD': 0.10,   # Thermal Helicity Dynamics
        'PGF': 0.08,   # Pressure Gradient Force
        'HCI': 0.07,   # Humidity-Convection Interaction
        'ASI': 0.06,   # Atmospheric Stability Integration
        'LRC': 0.05    # Local Roughness Coefficient
    }
    
    # Uncertainty ranges (95% credible intervals)
    UNCERTAINTY = {
        'KED': 0.012,
        'TII': 0.015,
        'VSR': 0.018,
        'AOD': 0.020,
        'THD': 0.022,
        'PGF': 0.025,
        'HCI': 0.028,
        'ASI': 0.030,
        'LRC': 0.031
    }
    
    # Climate zone specific adjustments
    CLIMATE_ADJUSTMENTS = {
        'tropical': {
            'KED': 1.05,   # +5% for KED in tropics
            'THD': 1.10,   # +10% for THD (more convection)
            'HCI': 1.15,   # +15% for humidity
        },
        'arid': {
            'AOD': 1.20,   # +20% for AOD (dust)
            'HCI': 0.80,   # -20% for humidity
        },
        'temperate': {
            'VSR': 1.05,   # +5% for shear
            'ASI': 1.10,   # +10% for stability
        },
        'continental': {
            'ASI': 1.15,   # +15% for stability
            'VSR': 1.10,   # +10% for shear
        },
        'polar': {
            'AOD': 0.70,   # -30% for AOD (clean air)
            'THD': 0.80,   # -20% for thermal effects
        },
        'high_altitude': {
            'KED': 1.10,   # +10% for KED
            'LRC': 1.20,   # +20% for roughness
        }
    }
    
    def __init__(self, 
                 weights_file: Optional[Path] = None,
                 climate_zone: str = 'temperate'):
        """Initialize weights.
        
        Args:
            weights_file: Optional custom weights file
            climate_zone: Climate zone for adjustments
        """
        self.climate_zone = climate_zone
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.uncertainty = self.UNCERTAINTY.copy()
        
        # Load custom weights if provided
        if weights_file and weights_file.exists():
            self.load_weights(weights_file)
        
        # Apply climate zone adjustments
        self._apply_climate_adjustments()
        
        # Renormalize to sum to 1
        self._renormalize()
    
    def _apply_climate_adjustments(self):
        """Apply climate zone specific weight adjustments."""
        if self.climate_zone in self.CLIMATE_ADJUSTMENTS:
            adjustments = self.CLIMATE_ADJUSTMENTS[self.climate_zone]
            
            for param, factor in adjustments.items():
                if param in self.weights:
                    self.weights[param] *= factor
    
    def _renormalize(self):
        """Renormalize weights to sum to 1."""
        total = sum(self.weights.values())
        for param in self.weights:
            self.weights[param] /= total
    
    def get_weight(self, parameter: str) -> float:
        """Get weight for a parameter."""
        return self.weights.get(parameter, 0.0)
    
    def get_weights(self) -> Dict[str, float]:
        """Get all weights."""
        return self.weights.copy()
    
    def get_uncertainty(self, parameter: str) -> float:
        """Get uncertainty for a parameter weight."""
        return self.uncertainty.get(parameter, 0.0)
    
    def load_weights(self, weights_file: Path):
        """Load custom weights from file."""
        with open(weights_file, 'r') as f:
            custom_weights = json.load(f)
        
        for param, weight in custom_weights.items():
            if param in self.weights:
                self.weights[param] = weight
    
    def save_weights(self, weights_file: Path):
        """Save current weights to file."""
        with open(weights_file, 'w') as f:
            json.dump(self.weights, f, indent=2)
    
    def sample_posterior(self, n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Sample from posterior distribution of weights.
        
        Uses uncertainty estimates to generate samples.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Dictionary of sampled weights
        """
        samples = {param: [] for param in self.weights}
        
        for _ in range(n_samples):
            # Generate Dirichlet-like samples
            alphas = [self.weights[p] / self.uncertainty[p] 
                     for p in self.weights]
            dirichlet_sample = np.random.dirichlet(alphas)
            
            for i, param in enumerate(self.weights):
                samples[param].append(dirichlet_sample[i])
        
        # Convert to arrays
        for param in samples:
            samples[param] = np.array(samples[param])
        
        return samples
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'weights': self.weights,
            'uncertainty': self.uncertainty,
            'climate_zone': self.climate_zone,
            'sum': sum(self.weights.values())
        }
    
    def __repr__(self):
        """String representation."""
        lines = ["AKEWeights:"]
        for param, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
            unc = self.uncertainty[param]
            lines.append(f"  {param}: {weight:.3f} ± {unc:.3f}")
        lines.append(f"  Climate zone: {self.climate_zone}")
        return "\n".join(lines)
