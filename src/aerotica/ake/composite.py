"""AKE Composite Index.

.. math::
    AKE = \\sum_{i=1}^{9} w_i \\cdot \\phi_i

where:
    - :math:`w_i` are Bayesian-optimized weights
    - :math:`\\phi_i` are normalized parameter scores
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import yaml


class AKEComposite:
    """Atmospheric Kinetic Efficiency composite index."""
    
    # Default weights from Bayesian optimization
    DEFAULT_WEIGHTS = {
        'KED': 0.22,
        'TII': 0.16,
        'VSR': 0.14,
        'AOD': 0.12,
        'THD': 0.10,
        'PGF': 0.08,
        'HCI': 0.07,
        'ASI': 0.06,
        'LRC': 0.05
    }
    
    # Classification thresholds
    THRESHOLDS = {
        'PREMIUM': 0.85,
        'VIABLE': 0.70,
        'MARGINAL': 0.55,
        'CONSTRAINED': 0.40,
        'BENIGN': 0.00
    }
    
    def __init__(self, 
                 site_id: str,
                 climate_zone: str,
                 site_type: str,
                 weights_file: Optional[Path] = None):
        """Initialize AKE composite.
        
        Args:
            site_id: Unique site identifier
            climate_zone: Climate zone classification
            site_type: Type of site (urban, offshore, rural, etc.)
            weights_file: Optional custom weights file
        """
        self.site_id = site_id
        self.climate_zone = climate_zone
        self.site_type = site_type
        
        # Load weights
        self.weights = self._load_weights(weights_file)
        
        # Store parameters
        self.parameters = {}
        self.scores = {}
    
    def _load_weights(self, weights_file: Optional[Path]) -> Dict[str, float]:
        """Load Bayesian-optimized weights."""
        if weights_file and weights_file.exists():
            with open(weights_file, 'r') as f:
                return yaml.safe_load(f)
        
        return self.DEFAULT_WEIGHTS.copy()
    
    def load_parameters(self, parameters: Dict[str, float]) -> None:
        """Load normalized parameter scores.
        
        Args:
            parameters: Dictionary with parameter names and normalized scores
        """
        self.parameters = parameters
        
        # Normalize scores if needed
        for param, value in parameters.items():
            if param in self.weights:
                if 0 <= value <= 1:
                    self.scores[param] = value
                else:
                    raise ValueError(f"Parameter {param} must be in [0, 1]")
    
    def compute(self) -> Dict[str, Any]:
        """Compute AKE composite index.
        
        Returns:
            Dictionary with AKE score, classification, and details
        """
        if not self.scores:
            raise ValueError("No parameters loaded. Call load_parameters() first.")
        
        # Check for missing parameters
        missing_params = set(self.weights.keys()) - set(self.scores.keys())
        
        if missing_params:
            # Renormalize weights for available parameters
            available_weights = {
                p: self.weights[p] for p in self.scores.keys()
            }
            total_weight = sum(available_weights.values())
            normalized_weights = {
                p: w / total_weight for p, w in available_weights.items()
            }
        else:
            normalized_weights = self.weights
        
        # Compute weighted sum
        ake_score = 0.0
        contributions = {}
        
        for param, score in self.scores.items():
            weight = normalized_weights[param]
            contribution = weight * score
            ake_score += contribution
            contributions[param] = {
                'score': score,
                'weight': weight,
                'contribution': contribution
            }
        
        # Determine classification
        classification = self._classify(ake_score)
        
        # Assess gust risk
        gust_risk = self._assess_gust_risk()
        
        # Calculate confidence
        confidence = self._calculate_confidence()
        
        return {
            'site_id': self.site_id,
            'climate_zone': self.climate_zone,
            'site_type': self.site_type,
            'score': float(ake_score),
            'classification': classification,
            'gust_risk': gust_risk,
            'confidence': confidence,
            'contributions': contributions,
            'missing_parameters': list(missing_params),
            'weights_used': 'default' if not missing_params else 'renormalized'
        }
    
    def _classify(self, score: float) -> str:
        """Classify AKE score into operational categories."""
        if score >= self.THRESHOLDS['PREMIUM']:
            return 'PREMIUM'
        elif score >= self.THRESHOLDS['VIABLE']:
            return 'VIABLE'
        elif score >= self.THRESHOLDS['MARGINAL']:
            return 'MARGINAL'
        elif score >= self.THRESHOLDS['CONSTRAINED']:
            return 'CONSTRAINED'
        else:
            return 'BENIGN'
    
    def _assess_gust_risk(self) -> str:
        """Assess gust risk based on THD and TII."""
        thd = self.scores.get('THD', 0)
        tii = self.scores.get('TII', 0)
        
        # THD is the primary gust predictor
        if thd > 0.8:
            return 'SEVERE'
        elif thd > 0.6:
            return 'HIGH'
        elif thd > 0.4:
            return 'MODERATE'
        elif tii > 0.15:
            return 'ELEVATED'
        else:
            return 'LOW'
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence in AKE estimate.
        
        Based on number of parameters available and their weights.
        """
        available_weight = sum(
            self.weights[p] for p in self.scores.keys()
        )
        
        # Confidence = available_weight + 0.2 * (9 - missing_count)/9
        missing_count = 9 - len(self.scores)
        completeness_bonus = 0.2 * (missing_count / 9)
        
        confidence = min(available_weight + completeness_bonus, 1.0)
        
        return float(confidence)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export results to dictionary."""
        return self.compute()
    
    def to_json(self) -> str:
        """Export results to JSON string."""
        import json
        return json.dumps(self.compute(), indent=2, default=str)
