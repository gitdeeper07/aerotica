"""Unit tests for alert system."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from aerotica.alerts import GustPreAlertEngine, GustDetector, AlertNotifier


class TestGustDetector:
    """Test gust detection functionality."""
    
    def setup_method(self):
        self.detector = GustDetector(thd_threshold=0.7)
    
    def test_peak_detection(self):
        """Test peak detection in THD series."""
        thd = np.array([0.1, 0.3, 0.8, 0.6, 0.2, 0.1, 0.9, 0.5, 0.2])
        timestamps = pd.date_range('2025-01-01', periods=9, freq='1min')
        
        events = self.detector.detect(thd, timestamps)
        
        assert len(events) == 2  # Two peaks above threshold
        assert events[0]['thd_value'] == 0.8
        assert events[1]['thd_value'] == 0.9
    
    def test_lead_time_estimation(self):
        """Test lead time estimation."""
        assert self.detector._estimate_lead_time(0.9) == 240  # 4 minutes
        assert self.detector._estimate_lead_time(0.8) == 270  # 4.5 minutes
        assert self.detector._estimate_lead_time(0.7) == 300  # 5 minutes
        assert self.detector._estimate_lead_time(0.6) == 480  # 8 minutes
    
    def test_confidence_calculation(self):
        """Test confidence calculation."""
        conf1 = self.detector._calculate_confidence(0.9, 240)
        conf2 = self.detector._calculate_confidence(0.7, 300)
        
        assert 0.5 <= conf1 <= 0.95
        assert 0.5 <= conf2 <= 0.95
        assert conf1 > conf2  # Higher THD = higher confidence


class TestGustPreAlertEngine:
    """Test alert engine functionality."""
    
    def setup_method(self):
        self.engine = GustPreAlertEngine(
            thd_threshold=0.7,
            check_interval=30
        )
        
        # Create mock observations
        self.observations = pd.DataFrame({
            'wind_speed': [10, 12, 11, 13, 15],
            'temperature': [20, 21, 20, 22, 23],
            'humidity': [70, 72, 71, 73, 74]
        })
    
    def test_evaluate_no_alert(self):
        """Test evaluation when no alert should trigger."""
        # Mock THD to be below threshold
        self.engine._compute_thd = lambda x: {'value': 0.5}
        
        alert = self.engine.evaluate(self.observations)
        assert alert is None
        assert len(self.engine.alerts) == 0
    
    def test_evaluate_with_alert(self):
        """Test evaluation when alert triggers."""
        # Mock THD to be above threshold
        self.engine._compute_thd = lambda x: {
            'value': 0.75,
            'normalized': 0.75,
            'gust_probability': 0.7,
            'estimated_lead_time_min': 5,
            'confidence': 0.85
        }
        
        alert = self.engine.evaluate(self.observations)
        
        assert alert is not None
        assert alert['thd_value'] == 0.75
        assert 'alert_id' in alert
        assert 'lead_time_seconds' in alert
        assert len(self.engine.alerts) == 1
    
    def test_dispatch(self):
        """Test alert dispatch."""
        alert = {
            'alert_id': 'TEST_001',
            'location': 'test_site',
            'timestamp': datetime.now().isoformat(),
            'thd_value': 0.8,
            'lead_time_seconds': 300,
            'expected_gust_speed': 25.5,
            'confidence': 0.85
        }
        
        # Test console dispatch
        result = self.engine.dispatch(alert, recipients=['console'], channels=['console'])
        assert result is True
    
    def test_stats(self):
        """Test statistics collection."""
        # Add some test alerts
        self.engine.alerts = [
            {'status': 'active'},
            {'status': 'active'},
            {'status': 'resolved'}
        ]
        self.engine.alert_count = 3
        
        stats = self.engine.get_stats()
        
        assert stats['total_alerts'] == 3
        assert stats['active_alerts'] == 2


class TestAlertNotifier:
    """Test alert notification system."""
    
    def setup_method(self):
        self.notifier = AlertNotifier(config={
            'channels': ['console', 'file'],
            'log_dir': 'test_logs'
        })
    
    def test_console_notification(self):
        """Test console notification."""
        alert = {'alert_id': 'TEST', 'location': 'test'}
        result = self.notifier._send_console(alert)
        assert result is True
    
    def test_file_notification(self):
        """Test file notification."""
        alert = {'alert_id': 'TEST', 'location': 'test'}
        result = self.notifier._send_file(alert)
        assert result is True
    
    def test_send_alert(self):
        """Test sending alert through all channels."""
        alert = {'alert_id': 'TEST', 'location': 'test'}
        results = self.notifier.send_alert(alert, recipients=['test@test.com'])
        
        assert 'console' in results
        assert 'file' in results
        assert results['console'] is True
        assert results['file'] is True
    
    def test_channel_management(self):
        """Test adding/removing channels."""
        self.notifier.add_channel('slack', {'slack_webhook': 'test'})
        assert 'slack' in self.notifier.channels
        
        self.notifier.remove_channel('slack')
        assert 'slack' not in self.notifier.channels
    
    def test_test_channels(self):
        """Test channel testing."""
        results = self.notifier.test_channels()
        assert isinstance(results, dict)
        assert 'console' in results
        assert 'file' in results


if __name__ == '__main__':
    pytest.main([__file__])
