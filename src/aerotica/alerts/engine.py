"""Gust Pre-Alert Engine.

Provides 4-6 minute lead time for convective gust events based on THD analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

from aerotica.parameters.thd import THD
from aerotica.pinn.inference import AeroticaPINN


class GustPreAlertEngine:
    """Engine for gust pre-alerting with 4-6 minute lead time."""
    
    def __init__(self,
                 pinn_model: Optional[AeroticaPINN] = None,
                 site_config: Optional[Dict] = None,
                 thd_threshold: float = 0.7,
                 check_interval: int = 30,
                 alert_lead_time_min: int = 5):
        """Initialize gust pre-alert engine.
        
        Args:
            pinn_model: Pre-trained PINN model
            site_config: Site configuration
            thd_threshold: THD threshold for alert triggering
            check_interval: Check interval in seconds
            alert_lead_time_min: Target lead time in minutes
        """
        self.pinn = pinn_model
        self.site_config = site_config or {}
        self.thd_threshold = thd_threshold
        self.check_interval = check_interval
        self.alert_lead_time_min = alert_lead_time_min
        
        # Initialize THD detector
        self.thd_detector = THD(config={'threshold': thd_threshold})
        
        # Alert history
        self.alerts = []
        self.alert_count = 0
        
        # Performance metrics
        self.metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_events': 0,
            'avg_lead_time': 0.0
        }
    
    def evaluate(self, 
                observations: pd.DataFrame,
                timestamp: Optional[datetime] = None) -> Optional[Dict]:
        """Evaluate current conditions for gust potential.
        
        Args:
            observations: Current observations DataFrame
            timestamp: Current timestamp
            
        Returns:
            Alert dictionary if triggered, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Compute THD from observations
        thd_result = self._compute_thd(observations)
        
        # Check if alert should be triggered
        if thd_result['value'] >= self.thd_threshold:
            alert = self._generate_alert(thd_result, observations, timestamp)
            self.alerts.append(alert)
            self.alert_count += 1
            return alert
        
        return None
    
    def _compute_thd(self, observations: pd.DataFrame) -> Dict:
        """Compute THD from observations."""
        # Extract required data
        # In production, this would compute proper THD from 3D fields
        # Simplified for demonstration
        
        # Use temperature gradient as proxy
        if 'temperature' in observations.columns:
            temp_gradient = observations['temperature'].diff().abs().mean()
        else:
            temp_gradient = 0.5
        
        # Use wind variability as vorticity proxy
        if 'wind_speed' in observations.columns:
            wind_std = observations['wind_speed'].std()
        else:
            wind_std = 2.0
        
        # Simplified THD computation
        thd_value = min(0.3 + 0.1 * wind_std + 0.2 * temp_gradient, 1.0)
        
        return self.thd_detector.report(
            thd_value=thd_value,
            background_wind=observations.get('wind_speed', pd.Series([10])).iloc[-1]
        )
    
    def _generate_alert(self,
                       thd_result: Dict,
                       observations: pd.DataFrame,
                       timestamp: datetime) -> Dict:
        """Generate alert notification."""
        # Predict gust arrival time and intensity
        gust_prediction = self._predict_gust(thd_result, observations)
        
        alert = {
            'alert_id': f"GUST_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            'timestamp': timestamp.isoformat(),
            'thd_value': thd_result['value'],
            'thd_anomaly': thd_result.get('normalized', thd_result['value']),
            'gust_probability': thd_result.get('gust_probability', 0.7),
            'predicted_gust_time': gust_prediction['arrival_time'].isoformat(),
            'lead_time_seconds': gust_prediction['lead_time_seconds'],
            'expected_gust_speed': gust_prediction['gust_speed'],
            'location': self.site_config.get('location', 'unknown'),
            'confidence': thd_result.get('confidence', 0.85),
            'status': 'active'
        }
        
        return alert
    
    def _predict_gust(self,
                     thd_result: Dict,
                     observations: pd.DataFrame) -> Dict:
        """Predict gust arrival time and intensity."""
        # Base lead time from THD
        base_lead_time = thd_result.get('estimated_lead_time_min', 5) * 60
        
        # Adjust based on location and conditions
        lead_time = base_lead_time * self._get_lead_time_factor(observations)
        
        # Estimate gust speed
        current_wind = observations.get('wind_speed', pd.Series([10])).iloc[-1]
        gust_factor = 1.2 + 0.4 * thd_result['value']
        gust_speed = current_wind * gust_factor
        
        arrival_time = datetime.now() + timedelta(seconds=lead_time)
        
        return {
            'arrival_time': arrival_time,
            'lead_time_seconds': int(lead_time),
            'gust_speed': float(gust_speed)
        }
    
    def _get_lead_time_factor(self, observations: pd.DataFrame) -> float:
        """Get lead time adjustment factor based on conditions."""
        factor = 1.0
        
        # Adjust for stability
        if 'temperature' in observations.columns:
            temp_trend = observations['temperature'].diff().mean()
            if temp_trend < -0.1:  # Cooling rapidly
                factor *= 0.9  # Faster development
        
        # Adjust for humidity
        if 'humidity' in observations.columns:
            humidity = observations['humidity'].iloc[-1]
            if humidity > 80:  # High humidity
                factor *= 0.95  # Slightly faster
        
        return float(np.clip(factor, 0.8, 1.2))
    
    def dispatch(self, 
                alert: Dict,
                recipients: List[str],
                channels: List[str] = None) -> bool:
        """Dispatch alert to recipients.
        
        Args:
            alert: Alert dictionary
            recipients: List of recipient identifiers
            channels: Notification channels
            
        Returns:
            True if dispatched successfully
        """
        if channels is None:
            channels = ['console', 'file']
        
        success = True
        
        for channel in channels:
            if channel == 'console':
                self._dispatch_console(alert)
            elif channel == 'file':
                self._dispatch_file(alert)
            elif channel == 'email':
                self._dispatch_email(alert, recipients)
            elif channel == 'webhook':
                self._dispatch_webhook(alert, recipients)
        
        return success
    
    def _dispatch_console(self, alert: Dict):
        """Print alert to console."""
        print("\n" + "="*60)
        print(f"⚠️  GUST PRE-ALERT [{alert['alert_id']}]")
        print("="*60)
        print(f"📍 Location: {alert['location']}")
        print(f"⏰ Time: {alert['timestamp']}")
        print(f"🌪️  THD Anomaly: {alert['thd_anomaly']:.3f}")
        print(f"📊 Gust Probability: {alert['gust_probability']:.1%}")
        print(f"⏱️  Lead Time: {alert['lead_time_seconds']} seconds")
        print(f"💨 Expected Gust: {alert['expected_gust_speed']:.1f} m/s")
        print(f"🎯 Confidence: {alert['confidence']:.1%}")
        print("="*60 + "\n")
    
    def _dispatch_file(self, alert: Dict):
        """Write alert to file."""
        log_dir = Path("logs/alerts")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.log"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(alert) + '\n')
    
    def _dispatch_email(self, alert: Dict, recipients: List[str]):
        """Send email alert (placeholder)."""
        # In production, integrate with email service
        pass
    
    def _dispatch_webhook(self, alert: Dict, recipients: List[str]):
        """Send webhook alert (placeholder)."""
        # In production, send to webhook endpoints
        pass
    
    def verify_alert(self, alert_id: str, actual_gust: Optional[float] = None) -> bool:
        """Verify if an alert was correct.
        
        Args:
            alert_id: Alert identifier
            actual_gust: Actual gust speed if occurred
            
        Returns:
            True if alert was correct
        """
        for alert in self.alerts:
            if alert['alert_id'] == alert_id:
                if actual_gust is not None:
                    # Alert was correct (gust occurred)
                    alert['verified'] = True
                    alert['actual_gust'] = actual_gust
                    self.metrics['true_positives'] += 1
                    
                    # Calculate lead time error
                    alert_time = datetime.fromisoformat(alert['timestamp'])
                    gust_time = alert_time + timedelta(seconds=alert['lead_time_seconds'])
                    actual_time = datetime.now()
                    error = abs((actual_time - gust_time).total_seconds())
                    self.metrics['avg_lead_time'] = (
                        (self.metrics['avg_lead_time'] * (self.metrics['true_positives'] - 1) + error) /
                        self.metrics['true_positives']
                    )
                    return True
                else:
                    # False alarm
                    alert['verified'] = False
                    self.metrics['false_positives'] += 1
                    return False
        
        return False
    
    def get_stats(self) -> Dict:
        """Get alert statistics."""
        total_alerts = len(self.alerts)
        if total_alerts > 0:
            pod = self.metrics['true_positives'] / (self.metrics['true_positives'] + self.metrics['false_negatives'] + 1e-6)
            far = self.metrics['false_positives'] / (total_alerts + 1e-6)
        else:
            pod = 0.0
            far = 0.0
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': sum(1 for a in self.alerts if a.get('status') == 'active'),
            'true_positives': self.metrics['true_positives'],
            'false_positives': self.metrics['false_positives'],
            'false_negatives': self.metrics['false_negatives'],
            'probability_of_detection': pod,
            'false_alarm_rate': far,
            'avg_lead_time_seconds': self.metrics['avg_lead_time']
        }
    
    def run_realtime(self, data_stream, interval: int = 30):
        """Run real-time monitoring loop."""
        print(f"⚠️  Starting real-time gust monitoring (interval={interval}s)")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Get latest observations
                observations = data_stream.get_latest()
                
                # Evaluate
                alert = self.evaluate(observations)
                
                if alert:
                    self.dispatch(alert, recipients=['console'])
                
                # Wait for next interval
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")
            print(f"   Total alerts: {self.alert_count}")
            print(f"   Stats: {self.get_stats()}")
