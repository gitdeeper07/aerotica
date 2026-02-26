"""Turbulence Intensity Index (TII) Parameter.

.. math::
    TII = \\frac{\\sigma_v}{\\bar{v}}

where:
    - :math:`\\sigma_v` is standard deviation of wind speed
    - :math:`\\bar{v}` is mean wind speed
"""

import numpy as np
from typing import Union, Optional, Dict, Any, List


class TII:
    """Turbulence Intensity Index parameter.
    
    Weight in AKE index: 16%
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TII parameter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.weight = 0.16
        self.sampling_freq = self.config.get('sampling_freq', 1.0)  # Hz
        self.avg_period = self.config.get('avg_period', 600)  # seconds (10 min)
    
    def compute(self, 
                wind_speed_series: Union[List[float], np.ndarray],
                method: str = 'scalar') -> float:
        """Compute Turbulence Intensity Index.
        
        Args:
            wind_speed_series: Time series of wind speed measurements [m/s]
            method: 'scalar' for basic TII, 'spectral' for STII
            
        Returns:
            TII value
        """
        if len(wind_speed_series) == 0:
            return 0.0
            
        if method == 'scalar':
            return self._compute_scalar_tii(wind_speed_series)
        elif method == 'spectral':
            return self._compute_spectral_tii(wind_speed_series)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _compute_scalar_tii(self, series: Union[List[float], np.ndarray]) -> float:
        """Compute basic scalar TII.
        
        TII = σ_v / v̄
        """
        series = np.array(series)
        if len(series) == 0:
            return 0.0
            
        mean_v = np.mean(series)
        if mean_v == 0 or np.isnan(mean_v):
            return 0.0
            
        std_v = np.std(series)
        
        return float(std_v / mean_v)
    
    def _compute_spectral_tii(self, series: Union[List[float], np.ndarray]) -> float:
        """Compute spectral TII using von Kármán turbulence model.
        
        Focuses on frequency range 0.05-10 Hz that drives structural loading.
        """
        series = np.array(series)
        if len(series) < 10:
            return self._compute_scalar_tii(series)
            
        n = len(series)
        
        # Remove mean
        series = series - np.mean(series)
        
        # Compute FFT
        fft = np.fft.fft(series)
        freq = np.fft.fftfreq(n, d=1/self.sampling_freq)
        
        # Get positive frequencies
        pos_mask = freq > 0
        freq_pos = freq[pos_mask]
        fft_pos = np.abs(fft[pos_mask])
        
        # Calculate energy in structural loading range (0.05-10 Hz)
        structural_mask = (freq_pos >= 0.05) & (freq_pos <= 10.0)
        
        if not np.any(structural_mask):
            return self._compute_scalar_tii(series)
        
        # Compute spectral TII
        total_energy = np.sum(fft_pos ** 2)
        if total_energy == 0:
            return 0.0
            
        structural_energy = np.sum(fft_pos[structural_mask] ** 2)
        stii = np.sqrt(structural_energy / total_energy)
        
        return float(stii)
    
    def normalize(self, value: float) -> float:
        """Normalize TII value to [0, 1] range.
        
        Args:
            value: Raw TII value
            
        Returns:
            Normalized score (lower TII is better)
        """
        # IEC 61400-1 classes:
        # Class A (high turbulence): TII = 0.16 at 15 m/s
        # Class B (medium): TII = 0.14
        # Class C (low): TII = 0.12
        
        TII_GOOD = 0.12
        TII_POOR = 0.25
        
        if value <= TII_GOOD:
            return 1.0
        elif value >= TII_POOR:
            return 0.0
        else:
            # Linear interpolation between GOOD and POOR
            return 1.0 - (value - TII_GOOD) / (TII_POOR - TII_GOOD)
    
    def fatigue_risk(self, tii: float, turbine_class: str = 'IEC Class A') -> str:
        """Assess fatigue loading risk based on TII.
        
        Args:
            tii: TII value
            turbine_class: Turbine design class
            
        Returns:
            Risk level: 'LOW', 'MODERATE', 'HIGH', 'SEVERE'
        """
        if tii < 0.12:
            return 'LOW'
        elif tii < 0.16:
            return 'MODERATE'
        elif tii < 0.20:
            return 'HIGH'
        else:
            return 'SEVERE'
    
    def report(self, wind_speed_series=None, **kwargs) -> Dict[str, Any]:
        """Generate full diagnostic report."""
        # Handle both positional and keyword arguments
        if wind_speed_series is not None:
            series = wind_speed_series
        elif 'wind_speed_series' in kwargs:
            series = kwargs['wind_speed_series']
        else:
            series = []
            
        tii = self.compute(series)
        
        report = {
            'parameter': 'TII',
            'weight': self.weight,
            'value': tii,
            'normalized': self.normalize(tii),
            'fatigue_risk': self.fatigue_risk(tii),
            'units': 'dimensionless',
            'description': 'Turbulence Intensity Index',
            'formula': 'σ_v / v̄',
            'config': self.config
        }
        
        return report
