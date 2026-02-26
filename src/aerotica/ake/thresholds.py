"""AKE Classification Thresholds Module."""

from typing import Dict, Optional, Tuple
import numpy as np
from enum import Enum


class AKEClass(str, Enum):
    """AKE classification levels."""
    PREMIUM = "PREMIUM"
    VIABLE = "VIABLE"
    MARGINAL = "MARGINAL"
    CONSTRAINED = "CONSTRAINED"
    BENIGN = "BENIGN"


class AKETHRESHOLDS:
    """AKE score classification thresholds."""
    
    # Global thresholds (from validation dataset)
    GLOBAL_THRESHOLDS = {
        'PREMIUM': 0.85,
        'VIABLE': 0.70,
        'MARGINAL': 0.55,
        'CONSTRAINED': 0.40,
        'BENIGN': 0.00
    }
    
    # Climate zone specific thresholds
    CLIMATE_THRESHOLDS = {
        'tropical': {
            'PREMIUM': 0.82,
            'VIABLE': 0.68,
            'MARGINAL': 0.53,
            'CONSTRAINED': 0.38,
            'BENIGN': 0.00
        },
        'arid': {
            'PREMIUM': 0.88,
            'VIABLE': 0.72,
            'MARGINAL': 0.57,
            'CONSTRAINED': 0.42,
            'BENIGN': 0.00
        },
        'temperate': {
            'PREMIUM': 0.85,
            'VIABLE': 0.70,
            'MARGINAL': 0.55,
            'CONSTRAINED': 0.40,
            'BENIGN': 0.00
        },
        'continental': {
            'PREMIUM': 0.84,
            'VIABLE': 0.69,
            'MARGINAL': 0.54,
            'CONSTRAINED': 0.39,
            'BENIGN': 0.00
        },
        'polar': {
            'PREMIUM': 0.80,
            'VIABLE': 0.65,
            'MARGINAL': 0.50,
            'CONSTRAINED': 0.35,
            'BENIGN': 0.00
        },
        'high_altitude': {
            'PREMIUM': 0.83,
            'VIABLE': 0.68,
            'MARGINAL': 0.53,
            'CONSTRAINED': 0.38,
            'BENIGN': 0.00
        }
    }
    
    # Application-specific thresholds
    APPLICATION_THRESHOLDS = {
        'offshore_wind': {
            'PREMIUM': 0.82,
            'VIABLE': 0.68,
            'MARGINAL': 0.53,
        },
        'urban_wind': {
            'PREMIUM': 0.80,
            'VIABLE': 0.65,
            'MARGINAL': 0.50,
        },
        'gust_alert': {
            'HIGH_RISK': 0.75,
            'MODERATE_RISK': 0.60,
            'LOW_RISK': 0.40,
        }
    }
    
    def __init__(self, 
                 climate_zone: str = 'temperate',
                 application: Optional[str] = None):
        """Initialize thresholds.
        
        Args:
            climate_zone: Climate zone for thresholds
            application: Specific application (optional)
        """
        self.climate_zone = climate_zone
        self.application = application
        
        # Select thresholds
        if application and application in self.APPLICATION_THRESHOLDS:
            self.thresholds = self.APPLICATION_THRESHOLDS[application]
        elif climate_zone in self.CLIMATE_THRESHOLDS:
            self.thresholds = self.CLIMATE_THRESHOLDS[climate_zone]
        else:
            self.thresholds = self.GLOBAL_THRESHOLDS
    
    def classify(self, score: float) -> AKEClass:
        """Classify AKE score.
        
        Args:
            score: AKE score in [0, 1]
            
        Returns:
            Classification string
        """
        if score >= self.thresholds['PREMIUM']:
            return AKEClass.PREMIUM
        elif score >= self.thresholds['VIABLE']:
            return AKEClass.VIABLE
        elif score >= self.thresholds['MARGINAL']:
            return AKEClass.MARGINAL
        elif score >= self.thresholds['CONSTRAINED']:
            return AKEClass.CONSTRAINED
        else:
            return AKEClass.BENIGN
    
    def get_threshold(self, class_name: str) -> float:
        """Get threshold value for a class."""
        return self.thresholds.get(class_name, 0.0)
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """Get all thresholds."""
        return self.thresholds.copy()
    
    def get_color(self, class_name: AKEClass) -> str:
        """Get color for a classification."""
        colors = {
            AKEClass.PREMIUM: '#0066CC',    # Blue
            AKEClass.VIABLE: '#00CC66',     # Green
            AKEClass.MARGINAL: '#FFCC00',    # Yellow
            AKEClass.CONSTRAINED: '#FF9900',  # Orange
            AKEClass.BENIGN: '#666666'       # Gray
        }
        return colors.get(class_name, '#000000')
    
    def get_description(self, class_name: AKEClass) -> str:
        """Get description for a classification."""
        descriptions = {
            AKEClass.PREMIUM: "Maximum harvestable resource, peak gust hazard",
            AKEClass.VIABLE: "Strong resource, manageable turbulence loads",
            AKEClass.MARGINAL: "Below bankable threshold, site-specific potential",
            AKEClass.CONSTRAINED: "Limited resource, elevated turbulence risk",
            AKEClass.BENIGN: "Negligible kinetic energy resource"
        }
        return descriptions.get(class_name, "")
    
    def get_recommendation(self, class_name: AKEClass) -> str:
        """Get recommendation for a classification."""
        recommendations = {
            AKEClass.PREMIUM: "Priority wind development + full alerting",
            AKEClass.VIABLE: "Standard wind development + monitoring",
            AKEClass.MARGINAL: "Detailed site survey before development",
            AKEClass.CONSTRAINED: "Building-integrated micro-harvest only",
            AKEClass.BENIGN: "No wind development · structural monitoring"
        }
        return recommendations.get(class_name, "")
    
    def confidence_interval(self, score: float, uncertainty: float) -> Tuple[float, float]:
        """Get confidence interval for classification.
        
        Args:
            score: AKE score
            uncertainty: Score uncertainty
            
        Returns:
            (lower_bound, upper_bound)
        """
        lower = max(0, score - 1.96 * uncertainty)
        upper = min(1, score + 1.96 * uncertainty)
        return (lower, upper)
    
    def classification_confidence(self, 
                                 score: float, 
                                 uncertainty: float) -> float:
        """Get confidence in classification.
        
        Higher confidence when score is far from thresholds.
        
        Args:
            score: AKE score
            uncertainty: Score uncertainty
            
        Returns:
            Confidence score [0, 1]
        """
        # Find nearest threshold
        thresholds = list(self.thresholds.values())
        thresholds.sort()
        
        nearest_threshold = min(thresholds, key=lambda t: abs(t - score))
        distance = abs(score - nearest_threshold)
        
        # Confidence based on distance relative to uncertainty
        if distance == 0:
            return 0.5
        else:
            confidence = min(1.0, distance / (2 * uncertainty))
            return float(confidence)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'climate_zone': self.climate_zone,
            'application': self.application,
            'thresholds': self.thresholds
        }
    
    def __repr__(self):
        """String representation."""
        lines = [f"AKETHRESHOLDS (zone: {self.climate_zone}):"]
        for class_name, threshold in self.thresholds.items():
            lines.append(f"  {class_name}: {threshold:.2f}")
        return "\n".join(lines)
