"""Building-Integrated Wind Energy Assessor.

Assesses wind energy potential at rooftop and façade level
using high-resolution LiDAR data and morphological analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from dataclasses import dataclass, field
import rasterio
from rasterio.transform import from_origin

from aerotica.parameters.ked import KED
from aerotica.parameters.lrc import LRC
from aerotica.pinn.inference import AeroticaPINN


@dataclass
class BuildingSite:
    """Building site information."""
    id: str
    x: float
    y: float
    height: float
    roof_type: str
    area_m2: float
    ked_w_m2: float
    ake_score: float
    classification: str
    annual_yield_kwh: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BuildingWindAssessor:
    """Assessor for building-integrated wind energy potential."""
    
    def __init__(self,
                 lidar_dem: Optional[Path] = None,
                 climate_ref: Optional[Path] = None,
                 pinn_model: Optional[AeroticaPINN] = None,
                 config: Optional[Dict] = None):
        """Initialize building wind assessor.
        
        Args:
            lidar_dem: Path to LiDAR DEM file (GeoTIFF)
            climate_ref: Path to climate reference data
            pinn_model: Pre-trained PINN model
            config: Configuration dictionary
        """
        self.lidar_dem = lidar_dem
        self.climate_ref = climate_ref
        self.pinn = pinn_model
        self.config = config or {}
        
        # Initialize parameters
        self.ked = KED()
        self.lrc = LRC()
        
        # Load LiDAR data if provided
        self.dem_data = None
        self.dem_transform = None
        if lidar_dem and lidar_dem.exists():
            self._load_lidar()
        
        # Storage for identified sites
        self.sites: List[BuildingSite] = []
        
        # Thresholds
        self.ake_threshold = self.config.get('ake_threshold', 0.75)
        self.min_height_m = self.config.get('min_height_m', 15)
        self.min_area_m2 = self.config.get('min_area_m2', 50)
    
    def _load_lidar(self):
        """Load LiDAR DEM data."""
        with rasterio.open(self.lidar_dem) as src:
            self.dem_data = src.read(1)
            self.dem_transform = src.transform
            self.dem_bounds = src.bounds
            self.dem_resolution = src.res
    
    def identify_sites(self,
                      ake_threshold: Optional[float] = None,
                      return_geojson: bool = False) -> List[BuildingSite]:
        """Identify viable rooftop locations.
        
        Args:
            ake_threshold: AKE threshold for viability
            return_geojson: Return GeoJSON format
            
        Returns:
            List of viable building sites
        """
        if ake_threshold is None:
            ake_threshold = self.ake_threshold
        
        if self.dem_data is None:
            raise ValueError("LiDAR data not loaded")
        
        # Find buildings in DEM
        buildings = self._extract_buildings()
        
        # Assess each building
        viable_sites = []
        for building in buildings:
            assessment = self.assess_building(building)
            
            if assessment['ake_score'] >= ake_threshold:
                site = BuildingSite(
                    id=f"BLDG_{len(self.sites):05d}",
                    x=building['x'],
                    y=building['y'],
                    height=building['height'],
                    roof_type=building['roof_type'],
                    area_m2=building['area'],
                    ked_w_m2=assessment['ked'],
                    ake_score=assessment['ake_score'],
                    classification=assessment['classification'],
                    annual_yield_kwh=assessment['annual_yield_kwh'],
                    confidence=assessment['confidence'],
                    metadata=assessment
                )
                viable_sites.append(site)
                self.sites.append(site)
        
        return viable_sites
    
    def _extract_buildings(self) -> List[Dict]:
        """Extract building information from DEM."""
        buildings = []
        
        # Simple threshold-based building extraction
        # In production, use proper building footprint detection
        height_threshold = 5.0  # Minimum height to consider as building
        
        # Find local maxima (potential building locations)
        from scipy import ndimage
        
        # Apply Gaussian filter
        smoothed = ndimage.gaussian_filter(self.dem_data, sigma=2)
        
        # Find local maxima
        neighborhood = np.ones((3, 3))
        local_max = ndimage.maximum_filter(smoothed, footprint=neighborhood) == smoothed
        local_max &= smoothed > height_threshold
        
        # Get coordinates of local maxima
        y_idxs, x_idxs = np.where(local_max)
        
        for y_idx, x_idx in zip(y_idxs, x_idxs):
            # Convert pixel coordinates to world coordinates
            x, y = rasterio.transform.xy(self.dem_transform, y_idx, x_idx)
            
            # Get building height
            height = self.dem_data[y_idx, x_idx]
            
            # Estimate building area (simplified)
            area = np.pi * (10 * (height / 50)) ** 2  # Rough estimate
            
            # Determine roof type based on height and neighbors
            roof_type = self._classify_roof_type(x_idx, y_idx, height)
            
            buildings.append({
                'x': x,
                'y': y,
                'x_idx': x_idx,
                'y_idx': y_idx,
                'height': height,
                'area': area,
                'roof_type': roof_type
            })
        
        return buildings
    
    def _classify_roof_type(self, x_idx: int, y_idx: int, height: float) -> str:
        """Classify roof type based on DEM patterns."""
        # Extract local window
        window_size = 5
        half = window_size // 2
        
        x_start = max(0, x_idx - half)
        x_end = min(self.dem_data.shape[1], x_idx + half + 1)
        y_start = max(0, y_idx - half)
        y_end = min(self.dem_data.shape[0], y_idx + half + 1)
        
        window = self.dem_data[y_start:y_end, x_start:x_end]
        
        # Calculate statistics
        std = np.std(window)
        mean = np.mean(window)
        
        if std < 0.5:
            return 'flat'
        elif std < 1.5:
            if height > mean:
                return 'pitched'
            else:
                return 'shed'
        else:
            return 'complex'
    
    def assess_building(self, building: Dict) -> Dict[str, Any]:
        """Assess wind energy potential for a building.
        
        Args:
            building: Building information dictionary
            
        Returns:
            Assessment results
        """
        # Get wind speed at building height
        if self.pinn is not None:
            # Use PINN for accurate wind field
            wind_profile = self.pinn.infer_profile(
                z=np.array([10, 50, 100, building['height']]),
                x=building['x'],
                y=building['y']
            )
            wind_speed = wind_profile['wind_speed'][-1]  # at building height
        else:
            # Simplified estimation
            wind_speed = 6.0 * (building['height'] / 60) ** 0.2
        
        # Compute KED
        ked = self.ked.compute(wind_speed)
        
        # Apply urban bias correction
        ked_corrected = ked * self.ked.urban_bias(building['height'], 'urban')
        
        # Normalize KED
        ked_norm = self.ked.normalize(ked_corrected)
        
        # Get roughness
        roughness = self.lrc.roughness_by_terrain('urban')
        lrc_norm = self.lrc.normalize(roughness)
        
        # Estimate AKE score (simplified)
        ake_score = 0.7 * ked_norm + 0.3 * lrc_norm
        
        # Classify
        if ake_score >= 0.85:
            classification = 'PREMIUM'
        elif ake_score >= 0.70:
            classification = 'VIABLE'
        elif ake_score >= 0.55:
            classification = 'MARGINAL'
        elif ake_score >= 0.40:
            classification = 'CONSTRAINED'
        else:
            classification = 'BENIGN'
        
        # Calculate annual energy yield
        # Assume 25% efficiency, 8760 hours/year
        annual_yield_kwh = ked_corrected * 0.25 * 8760 * building['area'] / 1000
        
        # Confidence based on data availability
        confidence = 0.7 + 0.2 * (self.pinn is not None)
        
        return {
            'ked': float(ked_corrected),
            'ked_norm': float(ked_norm),
            'wind_speed': float(wind_speed),
            'roughness': float(roughness),
            'ake_score': float(ake_score),
            'classification': classification,
            'annual_yield_kwh': float(annual_yield_kwh),
            'confidence': float(confidence)
        }
    
    def to_geojson(self, sites: List[BuildingSite]) -> Dict:
        """Convert sites to GeoJSON format."""
        features = []
        
        for site in sites:
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [site.x, site.y]
                },
                'properties': {
                    'id': site.id,
                    'height_m': site.height,
                    'roof_type': site.roof_type,
                    'area_m2': site.area_m2,
                    'ked_w_m2': site.ked_w_m2,
                    'ake_score': site.ake_score,
                    'classification': site.classification,
                    'annual_yield_kwh': site.annual_yield_kwh,
                    'confidence': site.confidence
                }
            }
            features.append(feature)
        
        return {
            'type': 'FeatureCollection',
            'features': features,
            'metadata': {
                'total_sites': len(sites),
                'total_yield_kwh': sum(s.annual_yield_kwh for s in sites),
                'threshold_ake': self.ake_threshold,
                'generated': pd.Timestamp.now().isoformat()
            }
        }
    
    def generate_report(self, output_dir: Path):
        """Generate assessment report.
        
        Args:
            output_dir: Output directory for reports
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate summary
        summary = {
            'total_buildings_assessed': len(self.sites),
            'viable_sites': len([s for s in self.sites if s.ake_score >= self.ake_threshold]),
            'total_annual_yield_kwh': sum(s.annual_yield_kwh for s in self.sites),
            'average_ake': np.mean([s.ake_score for s in self.sites]) if self.sites else 0,
            'classification_breakdown': {
                'PREMIUM': sum(1 for s in self.sites if s.classification == 'PREMIUM'),
                'VIABLE': sum(1 for s in self.sites if s.classification == 'VIABLE'),
                'MARGINAL': sum(1 for s in self.sites if s.classification == 'MARGINAL'),
                'CONSTRAINED': sum(1 for s in self.sites if s.classification == 'CONSTRAINED'),
                'BENIGN': sum(1 for s in self.sites if s.classification == 'BENIGN')
            }
        }
        
        # Save summary
        with open(output_dir / 'assessment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save GeoJSON
        geojson = self.to_geojson(self.sites)
        with open(output_dir / 'viable_sites.geojson', 'w') as f:
            json.dump(geojson, f, indent=2)
        
        # Generate CSV
        rows = []
        for site in self.sites:
            rows.append({
                'id': site.id,
                'x': site.x,
                'y': site.y,
                'height_m': site.height,
                'roof_type': site.roof_type,
                'area_m2': site.area_m2,
                'ked_w_m2': site.ked_w_m2,
                'ake_score': site.ake_score,
                'classification': site.classification,
                'annual_yield_kwh': site.annual_yield_kwh,
                'confidence': site.confidence
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / 'assessment_results.csv', index=False)
        
        print(f"📊 Assessment report generated in {output_dir}")
        print(f"   Total viable sites: {summary['viable_sites']}")
        print(f"   Total annual yield: {summary['total_annual_yield_kwh']:.0f} kWh")
