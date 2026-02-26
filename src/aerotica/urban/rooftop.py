"""Rooftop Analysis Module."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import cv2
from scipy import ndimage


@dataclass
class RooftopFeatures:
    """Rooftop features for wind energy assessment."""
    x: float
    y: float
    height: float
    area_m2: float
    slope_degrees: float
    aspect_degrees: float
    roof_type: str
    exposure_factor: float
    turbulence_potential: float
    suitable_for_turbine: bool
    recommended_turbine_type: str


class RooftopAnalyzer:
    """Analyze rooftop characteristics for wind turbine placement."""
    
    def __init__(self,
                 dem_data: np.ndarray,
                 resolution: float = 2.0,
                 min_area: float = 50.0,
                 max_slope: float = 30.0):
        """Initialize rooftop analyzer.
        
        Args:
            dem_data: Digital Elevation Model data
            resolution: Spatial resolution in meters
            min_area: Minimum rooftop area for turbine [m²]
            max_slope: Maximum roof slope for turbine [degrees]
        """
        self.dem = dem_data
        self.resolution = resolution
        self.min_area = min_area
        self.max_slope = max_slope
        
        # Pre-compute slope and aspect
        self.slope, self.aspect = self._compute_slope_aspect()
    
    def _compute_slope_aspect(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute slope and aspect from DEM."""
        # Gradient in x and y directions
        gy, gx = np.gradient(self.dem, self.resolution)
        
        # Slope in degrees
        slope = np.arctan(np.sqrt(gx**2 + gy**2)) * 180 / np.pi
        
        # Aspect in degrees (0-360, clockwise from north)
        aspect = np.arctan2(-gx, gy) * 180 / np.pi
        aspect = (90 - aspect) % 360
        
        return slope, aspect
    
    def identify_rooftops(self) -> List[RooftopFeatures]:
        """Identify and analyze individual rooftops."""
        # Find building rooftops (local maxima)
        building_mask = self.dem > 5.0
        
        # Label connected components
        labeled, num_features = ndimage.label(building_mask)
        
        rooftops = []
        
        for label_id in range(1, num_features + 1):
            # Get mask for this building
            mask = labeled == label_id
            
            # Calculate rooftop features
            features = self._analyze_rooftop(mask, label_id)
            
            if features is not None:
                rooftops.append(features)
        
        return rooftops
    
    def _analyze_rooftop(self, mask: np.ndarray, label_id: int) -> Optional[RooftopFeatures]:
        """Analyze a single rooftop."""
        # Get coordinates
        y_idxs, x_idxs = np.where(mask)
        
        if len(x_idxs) == 0:
            return None
        
        # Calculate centroid
        x_center = np.mean(x_idxs) * self.resolution
        y_center = np.mean(y_idxs) * self.resolution
        
        # Get height at centroid
        height = np.mean(self.dem[mask])
        
        # Calculate area
        area = len(x_idxs) * self.resolution**2
        
        if area < self.min_area:
            return None
        
        # Calculate mean slope on rooftop
        mean_slope = np.mean(self.slope[mask])
        
        # Determine roof type based on slope
        if mean_slope < 5:
            roof_type = 'flat'
        elif mean_slope < 15:
            roof_type = 'low_pitch'
        elif mean_slope < 30:
            roof_type = 'medium_pitch'
        else:
            roof_type = 'steep_pitch'
        
        # Calculate exposure factor
        exposure = self._calculate_exposure(x_center, y_center, height)
        
        # Calculate turbulence potential
        turbulence = self._calculate_turbulence_potential(mask, mean_slope)
        
        # Determine suitability
        suitable = mean_slope <= self.max_slope and area >= self.min_area
        
        # Recommend turbine type
        if suitable:
            if area > 500:
                turbine_type = 'large'
            elif area > 200:
                turbine_type = 'medium'
            else:
                turbine_type = 'small'
        else:
            turbine_type = 'none'
        
        return RooftopFeatures(
            x=float(x_center),
            y=float(y_center),
            height=float(height),
            area_m2=float(area),
            slope_degrees=float(mean_slope),
            aspect_degrees=float(np.mean(self.aspect[mask])),
            roof_type=roof_type,
            exposure_factor=float(exposure),
            turbulence_potential=float(turbulence),
            suitable_for_turbine=suitable,
            recommended_turbine_type=turbine_type
        )
    
    def _calculate_exposure(self, x: float, y: float, height: float) -> float:
        """Calculate wind exposure factor for rooftop.
        
        Higher values indicate better exposure.
        """
        # Convert to pixel coordinates
        x_px = int(x / self.resolution)
        y_px = int(y / self.resolution)
        
        # Look at surrounding area
        radius = 10  # pixels
        y_start = max(0, y_px - radius)
        y_end = min(self.dem.shape[0], y_px + radius + 1)
        x_start = max(0, x_px - radius)
        x_end = min(self.dem.shape[1], x_px + radius + 1)
        
        surrounding = self.dem[y_start:y_end, x_start:x_end]
        
        # Calculate how much higher this point is than surroundings
        height_diff = height - np.mean(surrounding)
        
        # Normalize exposure
        exposure = np.clip(height_diff / 20, 0, 1)
        
        return exposure
    
    def _calculate_turbulence_potential(self, mask: np.ndarray, slope: float) -> float:
        """Calculate turbulence potential for rooftop.
        
        Higher values indicate more turbulent flow.
        """
        # Turbulence increases with slope and edge effects
        edge_fraction = 1.0 - (np.sum(mask) / (mask.shape[0] * mask.shape[1]))
        
        # Combine factors
        turbulence = 0.3 * (slope / 30) + 0.7 * edge_fraction
        
        return float(np.clip(turbulence, 0, 1))
    
    def get_optimal_placement(self, rooftops: List[RooftopFeatures]) -> List[Dict]:
        """Get optimal turbine placement for each rooftop."""
        placements = []
        
        for rooftop in rooftops:
            if not rooftop.suitable_for_turbine:
                continue
            
            # Find optimal position within rooftop
            # (simplified - place at highest point with good exposure)
            
            placement = {
                'x': rooftop.x,
                'y': rooftop.y,
                'height_m': rooftop.height,
                'roof_type': rooftop.roof_type,
                'turbine_type': rooftop.recommended_turbine_type,
                'exposure_score': rooftop.exposure_factor,
                'turbulence_score': rooftop.turbulence_potential,
                'estimated_power_kw': self._estimate_power(rooftop)
            }
            
            placements.append(placement)
        
        return placements
    
    def _estimate_power(self, rooftop: RooftopFeatures) -> float:
        """Estimate power generation potential [kW]."""
        # Simplified power estimation
        # In production, use detailed wind resource assessment
        
        base_power = 10  # kW base
        
        # Adjust for exposure
        exposure_factor = 0.5 + 0.5 * rooftop.exposure_factor
        
        # Adjust for turbulence (reduces efficiency)
        turbulence_factor = 1.0 - 0.3 * rooftop.turbulence_potential
        
        # Scale with area
        area_factor = np.sqrt(rooftop.area_m2 / 100)
        
        power = base_power * exposure_factor * turbulence_factor * area_factor
        
        return float(power)
    
    def generate_heatmap(self, output_path: str):
        """Generate suitability heatmap."""
        h, w = self.dem.shape
        heatmap = np.zeros_like(self.dem)
        
        rooftops = self.identify_rooftops()
        
        for rooftop in rooftops:
            if rooftop.suitable_for_turbine:
                # Create Gaussian blob at rooftop location
                x_px = int(rooftop.x / self.resolution)
                y_px = int(rooftop.y / self.resolution)
                
                radius = int(np.sqrt(rooftop.area_m2) / self.resolution / 2)
                
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if 0 <= y_px + dy < h and 0 <= x_px + dx < w:
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist <= radius:
                                value = np.exp(-dist**2 / (2 * (radius/3)**2))
                                heatmap[y_px + dy, x_px + dx] = max(
                                    heatmap[y_px + dy, x_px + dx],
                                    value
                                )
        
        # Normalize and save
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = (heatmap * 255).astype(np.uint8)
        
        cv2.imwrite(output_path, heatmap)
        print(f"✅ Heatmap saved to {output_path}")
