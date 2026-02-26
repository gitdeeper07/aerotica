"""Climate zone definitions and classification."""

from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class ClimateZone(str, Enum):
    """Climate zone enumeration."""
    TROPICAL = "tropical"
    ARID = "arid"
    TEMPERATE = "temperate"
    CONTINENTAL = "continental"
    POLAR = "polar"
    HIGH_ALTITUDE = "high_altitude"


# Zone boundaries [latitude_min, latitude_max, notes]
ZONE_BOUNDARIES: Dict[ClimateZone, List[Tuple[float, float, str]]] = {
    ClimateZone.TROPICAL: [
        (-23.5, 23.5, "Equatorial region")
    ],
    ClimateZone.ARID: [
        (15, 35, "Northern hemisphere deserts"),
        (-35, -15, "Southern hemisphere deserts")
    ],
    ClimateZone.TEMPERATE: [
        (35, 50, "Northern hemisphere temperate"),
        (-50, -35, "Southern hemisphere temperate"),
        (40, 60, "Maritime temperate")
    ],
    ClimateZone.CONTINENTAL: [
        (50, 70, "Northern hemisphere continental")
    ],
    ClimateZone.POLAR: [
        (66.5, 90, "Arctic"),
        (-90, -66.5, "Antarctic")
    ],
    ClimateZone.HIGH_ALTITUDE: []  # Based on elevation, not latitude
}


# Zone characteristics for AKE adjustments
ZONE_CHARACTERISTICS: Dict[ClimateZone, Dict[str, float]] = {
    ClimateZone.TROPICAL: {
        'ked_factor': 1.05,
        'thd_factor': 1.10,
        'hci_factor': 1.15,
        'asi_factor': 0.95,
        'default_temp': 27.0,
        'default_humidity': 80.0
    },
    ClimateZone.ARID: {
        'ked_factor': 1.02,
        'aod_factor': 1.20,
        'hci_factor': 0.80,
        'asi_factor': 1.05,
        'default_temp': 25.0,
        'default_humidity': 30.0
    },
    ClimateZone.TEMPERATE: {
        'ked_factor': 1.00,
        'vsr_factor': 1.05,
        'asi_factor': 1.10,
        'default_temp': 15.0,
        'default_humidity': 70.0
    },
    ClimateZone.CONTINENTAL: {
        'ked_factor': 0.98,
        'asi_factor': 1.15,
        'vsr_factor': 1.10,
        'default_temp': 5.0,
        'default_humidity': 65.0
    },
    ClimateZone.POLAR: {
        'ked_factor': 0.90,
        'aod_factor': 0.70,
        'thd_factor': 0.80,
        'default_temp': -15.0,
        'default_humidity': 85.0
    },
    ClimateZone.HIGH_ALTITUDE: {
        'ked_factor': 1.10,
        'lrc_factor': 1.20,
        'asi_factor': 1.15,
        'default_temp': 5.0,
        'default_humidity': 60.0
    }
}


class ClimateZoneClassifier:
    """Classify locations into climate zones."""
    
    def __init__(self):
        """Initialize classifier."""
        self.zones = list(ClimateZone)
    
    def classify(self, 
                 latitude: float, 
                 longitude: float,
                 elevation: Optional[float] = None) -> ClimateZone:
        """Classify location into climate zone.
        
        Args:
            latitude: Latitude in degrees (-90 to 90)
            longitude: Longitude in degrees (-180 to 180)
            elevation: Elevation in meters (optional)
            
        Returns:
            Climate zone
        """
        # Check high altitude first
        if elevation is not None and elevation > 2500:
            return ClimateZone.HIGH_ALTITUDE
        
        # Check each zone
        for zone, boundaries in ZONE_BOUNDARIES.items():
            for lat_min, lat_max, _ in boundaries:
                if lat_min <= latitude <= lat_max:
                    # Special cases for arid zones (longitude matters)
                    if zone == ClimateZone.ARID:
                        if self._is_arid_longitude(longitude):
                            return zone
                    else:
                        return zone
        
        # Default to temperate
        return ClimateZone.TEMPERATE
    
    def _is_arid_longitude(self, longitude: float) -> bool:
        """Check if longitude is in arid region."""
        # Major desert longitudes
        desert_regions = [
            (-120, -110),  # North American deserts
            (-10, 40),     # Sahara, Arabian
            (60, 80),      # Iranian deserts
            (100, 120),    # Gobi
            (130, 140),    # Australian deserts
            (-70, -60),    # Atacama
            (15, 30)       # Kalahari
        ]
        
        for lon_min, lon_max in desert_regions:
            if lon_min <= longitude <= lon_max:
                return True
        
        return False
    
    def get_characteristics(self, zone: ClimateZone) -> Dict[str, float]:
        """Get characteristic factors for a zone."""
        return ZONE_CHARACTERISTICS.get(zone, ZONE_CHARACTERISTICS[ClimateZone.TEMPERATE])
    
    def get_all_zones(self) -> List[str]:
        """Get list of all zone names."""
        return [z.value for z in self.zones]
    
    def get_zone_info(self, zone: ClimateZone) -> Dict:
        """Get detailed information about a zone."""
        boundaries = ZONE_BOUNDARIES.get(zone, [])
        characteristics = self.get_characteristics(zone)
        
        return {
            'name': zone.value,
            'boundaries': [(lat_min, lat_max, desc) for lat_min, lat_max, desc in boundaries],
            'characteristics': characteristics
        }
    
    def adjust_weights(self, 
                       weights: Dict[str, float],
                       zone: ClimateZone) -> Dict[str, float]:
        """Adjust parameter weights for climate zone."""
        factors = self.get_characteristics(zone)
        adjusted = weights.copy()
        
        # Map zone factors to parameter adjustments
        param_mapping = {
            'ked_factor': 'KED',
            'tii_factor': 'TII',
            'vsr_factor': 'VSR',
            'aod_factor': 'AOD',
            'thd_factor': 'THD',
            'pgf_factor': 'PGF',
            'hci_factor': 'HCI',
            'asi_factor': 'ASI',
            'lrc_factor': 'LRC'
        }
        
        for factor_name, param_name in param_mapping.items():
            if factor_name in factors and param_name in adjusted:
                adjusted[param_name] *= factors[factor_name]
        
        # Renormalize
        total = sum(adjusted.values())
        for param in adjusted:
            adjusted[param] /= total
        
        return adjusted
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ClimateZoneClassifier(zones={len(self.zones)})"
