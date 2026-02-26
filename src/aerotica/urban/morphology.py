"""Urban Morphology Analysis Module."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import rasterio
from rasterio.transform import from_origin


@dataclass
class UrbanMetrics:
    """Urban morphological metrics."""
    mean_height: float
    std_height: float
    frontal_area_index: float
    plan_area_density: float
    roughness_length: float
    displacement_height: float
    building_count: int
    sky_view_factor: float


class UrbanMorphology:
    """Analyze urban morphology from LiDAR and GIS data."""
    
    def __init__(self,
                 dem_data: Optional[np.ndarray] = None,
                 building_footprints: Optional[np.ndarray] = None,
                 resolution: float = 2.0):
        """Initialize urban morphology analyzer.
        
        Args:
            dem_data: Digital Elevation Model data
            building_footprints: Building footprint mask
            resolution: Spatial resolution in meters
        """
        self.dem = dem_data
        self.footprints = building_footprints
        self.resolution = resolution
        self.kappa = 0.41  # von Kármán constant
    
    def compute_metrics(self, 
                       x_min: float, 
                       y_min: float,
                       width: int,
                       height: int) -> UrbanMetrics:
        """Compute urban morphology metrics for a region.
        
        Args:
            x_min: Minimum x coordinate
            y_min: Minimum y coordinate
            width: Region width in pixels
            height: Region height in pixels
            
        Returns:
            UrbanMetrics object
        """
        if self.dem is None:
            raise ValueError("DEM data not loaded")
        
        # Extract region
        region_dem = self.dem[y_min:y_min+height, x_min:x_min+width]
        
        # Building detection (height > 5m)
        building_mask = region_dem > 5.0
        if self.footprints is not None:
            building_mask &= self.footprints[y_min:y_min+height, x_min:x_min+width]
        
        building_heights = region_dem[building_mask]
        
        # Basic statistics
        if len(building_heights) > 0:
            mean_h = np.mean(building_heights)
            std_h = np.std(building_heights)
            building_count = len(building_heights)
        else:
            mean_h = 0
            std_h = 0
            building_count = 0
        
        # Plan area density (λ_p)
        total_area = width * height * self.resolution**2
        built_area = np.sum(building_mask) * self.resolution**2
        lambda_p = built_area / total_area if total_area > 0 else 0
        
        # Frontal area index (λ_f) - simplified
        # In production, compute from building orientations
        lambda_f = lambda_p * 0.3  # Rough approximation
        
        # Roughness length (Macdonald et al. 1998)
        if lambda_f > 0:
            z0 = (mean_h - 0.7 * mean_h) * np.exp(
                -self.kappa / np.sqrt(0.5 * lambda_f)
            )
        else:
            z0 = 0.03  # Open terrain
        
        # Displacement height
        d = 0.7 * mean_h
        
        # Sky view factor (simplified)
        if mean_h > 0:
            svf = np.exp(-lambda_p * 2)
        else:
            svf = 1.0
        
        return UrbanMetrics(
            mean_height=float(mean_h),
            std_height=float(std_h),
            frontal_area_index=float(lambda_f),
            plan_area_density=float(lambda_p),
            roughness_length=float(z0),
            displacement_height=float(d),
            building_count=building_count,
            sky_view_factor=float(svf)
        )
    
    def compute_roughness_map(self) -> np.ndarray:
        """Compute roughness length map."""
        if self.dem is None:
            raise ValueError("DEM data not loaded")
        
        h, w = self.dem.shape
        roughness_map = np.zeros_like(self.dem)
        
        # Window size for local statistics
        window_size = 50  # pixels
        half = window_size // 2
        
        for i in range(half, h - half, window_size):
            for j in range(half, w - half, window_size):
                y_start = max(0, i - half)
                y_end = min(h, i + half)
                x_start = max(0, j - half)
                x_end = min(w, j + half)
                
                metrics = self.compute_metrics(
                    x_start, y_start,
                    x_end - x_start,
                    y_end - y_start
                )
                
                roughness_map[y_start:y_end, x_start:x_end] = metrics.roughness_length
        
        return roughness_map
    
    def classify_urban_zone(self, metrics: UrbanMetrics) -> str:
        """Classify urban zone type based on metrics."""
        if metrics.plan_area_density > 0.5:
            return 'HIGH_DENSITY_URBAN'
        elif metrics.plan_area_density > 0.3:
            if metrics.mean_height > 30:
                return 'HIGH_RISE_URBAN'
            else:
                return 'MEDIUM_DENSITY_URBAN'
        elif metrics.plan_area_density > 0.1:
            return 'LOW_DENSITY_URBAN'
        elif metrics.plan_area_density > 0.05:
            return 'SUBURBAN'
        else:
            return 'RURAL'
    
    def compute_wind_corridor_potential(self, 
                                       metrics: UrbanMetrics,
                                       wind_direction: float) -> float:
        """Compute potential for wind corridors.
        
        Args:
            metrics: Urban metrics
            wind_direction: Prevailing wind direction [degrees]
            
        Returns:
            Corridor potential score [0-1]
        """
        # Higher potential with aligned streets and lower roughness
        aligned_score = 0.5  # Would need street orientation data
        
        # Lower density = higher potential
        density_score = 1.0 - metrics.plan_area_density
        
        # Taller buildings can create stronger corridors
        height_score = np.clip(metrics.mean_height / 50, 0, 1)
        
        # Combined score
        potential = 0.4 * aligned_score + 0.4 * density_score + 0.2 * height_score
        
        return float(potential)
    
    def get_building_statistics(self) -> Dict[str, float]:
        """Get overall building statistics."""
        if self.dem is None:
            return {}
        
        building_mask = self.dem > 5.0
        building_heights = self.dem[building_mask]
        
        if len(building_heights) == 0:
            return {
                'building_count': 0,
                'mean_height': 0,
                'max_height': 0,
                'total_built_area': 0
            }
        
        total_area = self.dem.shape[0] * self.dem.shape[1] * self.resolution**2
        built_area = np.sum(building_mask) * self.resolution**2
        
        return {
            'building_count': int(len(building_heights)),
            'mean_height': float(np.mean(building_heights)),
            'std_height': float(np.std(building_heights)),
            'max_height': float(np.max(building_heights)),
            'min_height': float(np.min(building_heights)),
            'total_built_area_km2': float(built_area / 1e6),
            'building_density': float(built_area / total_area) if total_area > 0 else 0
        }
