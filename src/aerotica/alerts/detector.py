"""Gust Detector Module."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class GustDetector:
    """Detector for convective gust events using THD analysis."""
    
    def __init__(self,
                 thd_threshold: float = 0.7,
                 min_lead_time: int = 240,  # 4 minutes
                 max_lead_time: int = 480):  # 8 minutes
        """Initialize gust detector.
        
        Args:
            thd_threshold: THD threshold for detection
            min_lead_time: Minimum lead time in seconds
            max_lead_time: Maximum lead time in seconds
        """
        self.thd_threshold = thd_threshold
        self.min_lead_time = min_lead_time
        self.max_lead_time = max_lead_time
        
        # Detection history
        self.detections = []
    
    def detect(self,
              thd_series: np.ndarray,
              timestamps: np.ndarray,
              wind_speed: Optional[np.ndarray] = None) -> List[Dict]:
        """Detect gust events from THD time series.
        
        Args:
            thd_series: THD values over time
            timestamps: Corresponding timestamps
            wind_speed: Wind speed values (optional)
            
        Returns:
            List of detected events
        """
        events = []
        
        # Find peaks above threshold
        peaks = self._find_peaks(thd_series, threshold=self.thd_threshold)
        
        for peak_idx in peaks:
            event = self._characterize_event(
                thd_series, timestamps, peak_idx, wind_speed
            )
            events.append(event)
            self.detections.append(event)
        
        return events
    
    def _find_peaks(self, data: np.ndarray, threshold: float) -> List[int]:
        """Find peaks in time series above threshold."""
        peaks = []
        for i in range(1, len(data) - 1):
            if (data[i] > data[i-1] and 
                data[i] > data[i+1] and 
                data[i] >= threshold):
                peaks.append(i)
        return peaks
    
    def _characterize_event(self,
                           thd_series: np.ndarray,
                           timestamps: np.ndarray,
                           peak_idx: int,
                           wind_speed: Optional[np.ndarray]) -> Dict:
        """Characterize a detected gust event."""
        peak_time = timestamps[peak_idx]
        peak_thd = thd_series[peak_idx]
        
        # Estimate lead time based on THD
        lead_time = self._estimate_lead_time(peak_thd)
        
        # Estimate gust intensity
        if wind_speed is not None:
            current_wind = wind_speed[peak_idx]
            gust_factor = 1.2 + 0.4 * peak_thd
            gust_speed = current_wind * gust_factor
        else:
            gust_speed = None
            gust_factor = None
        
        # Calculate confidence
        confidence = self._calculate_confidence(peak_thd, lead_time)
        
        return {
            'detection_time': peak_time,
            'predicted_gust_time': peak_time + timedelta(seconds=lead_time),
            'thd_value': float(peak_thd),
            'lead_time_seconds': int(lead_time),
            'gust_speed': float(gust_speed) if gust_speed else None,
            'gust_factor': float(gust_factor) if gust_factor else None,
            'confidence': float(confidence),
            'verified': False
        }
    
    def _estimate_lead_time(self, thd: float) -> int:
        """Estimate lead time based on THD value."""
        if thd > 0.9:
            lead = self.min_lead_time
        elif thd > 0.8:
            lead = self.min_lead_time + 30
        elif thd > 0.7:
            lead = self.min_lead_time + 60
        else:
            lead = self.max_lead_time
        
        return int(np.clip(lead, self.min_lead_time, self.max_lead_time))
    
    def _calculate_confidence(self, thd: float, lead_time: int) -> float:
        """Calculate detection confidence."""
        # Higher THD = higher confidence
        thd_confidence = thd
        
        # Shorter lead time = higher confidence
        lead_confidence = 1.0 - (lead_time - self.min_lead_time) / (
            self.max_lead_time - self.min_lead_time
        )
        
        # Combined confidence
        confidence = 0.7 * thd_confidence + 0.3 * lead_confidence
        
        return float(np.clip(confidence, 0.5, 0.95))
    
    def verify_detection(self,
                        detection_time: datetime,
                        actual_gust_time: Optional[datetime] = None,
                        actual_gust_speed: Optional[float] = None) -> bool:
        """Verify if a detection was correct."""
        for detection in self.detections:
            if detection['detection_time'] == detection_time:
                detection['verified'] = True
                if actual_gust_time:
                    error = abs((actual_gust_time - detection['predicted_gust_time']).total_seconds())
                    detection['time_error_seconds'] = int(error)
                if actual_gust_speed:
                    detection['actual_gust_speed'] = actual_gust_speed
                return True
        return False
    
    def get_performance_metrics(self) -> Dict:
        """Get detection performance metrics."""
        if not self.detections:
            return {}
        
        verified = [d for d in self.detections if d.get('verified')]
        
        if verified:
            time_errors = [d.get('time_error_seconds', 0) for d in verified]
            speed_errors = []
            for d in verified:
                if d.get('gust_speed') and d.get('actual_gust_speed'):
                    speed_errors.append(abs(d['gust_speed'] - d['actual_gust_speed']))
        else:
            time_errors = []
            speed_errors = []
        
        return {
            'total_detections': len(self.detections),
            'verified_detections': len(verified),
            'verification_rate': len(verified) / len(self.detections) if self.detections else 0,
            'avg_time_error_seconds': np.mean(time_errors) if time_errors else 0,
            'std_time_error_seconds': np.std(time_errors) if time_errors else 0,
            'avg_speed_error_ms': np.mean(speed_errors) if speed_errors else 0,
            'std_speed_error_ms': np.std(speed_errors) if speed_errors else 0
        }
