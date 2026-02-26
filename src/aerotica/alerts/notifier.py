"""Alert Notifier Module."""

import smtplib
import requests
import json
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import logging


class AlertNotifier:
    """Multi-channel alert notifier for gust warnings."""
    
    def __init__(self,
                 config: Optional[Dict] = None):
        """Initialize notifier.
        
        Args:
            config: Configuration dictionary with notification settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Set up channels
        self.channels = self.config.get('channels', ['console', 'file'])
        
        # Email settings
        self.smtp_server = self.config.get('smtp_server', 'localhost')
        self.smtp_port = self.config.get('smtp_port', 25)
        self.email_from = self.config.get('email_from', 'alerts@aerotica.org')
        
        # Webhook settings
        self.webhook_urls = self.config.get('webhook_urls', [])
        
        # File settings
        self.log_dir = Path(self.config.get('log_dir', 'logs/alerts'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Slack settings
        self.slack_webhook = self.config.get('slack_webhook')
        
        # SMS settings (Twilio example)
        self.twilio_account_sid = self.config.get('twilio_account_sid')
        self.twilio_auth_token = self.config.get('twilio_auth_token')
        self.twilio_from_number = self.config.get('twilio_from_number')
    
    def send_alert(self,
                   alert: Dict,
                   recipients: List[str]) -> Dict[str, bool]:
        """Send alert through configured channels.
        
        Args:
            alert: Alert dictionary
            recipients: List of recipient identifiers
            
        Returns:
            Dictionary with channel success status
        """
        results = {}
        
        for channel in self.channels:
            try:
                if channel == 'console':
                    results['console'] = self._send_console(alert)
                elif channel == 'file':
                    results['file'] = self._send_file(alert)
                elif channel == 'email':
                    results['email'] = self._send_email(alert, recipients)
                elif channel == 'webhook':
                    results['webhook'] = self._send_webhook(alert)
                elif channel == 'slack':
                    results['slack'] = self._send_slack(alert)
                elif channel == 'sms':
                    results['sms'] = self._send_sms(alert, recipients)
                else:
                    self.logger.warning(f"Unknown channel: {channel}")
                    results[channel] = False
                    
            except Exception as e:
                self.logger.error(f"Error sending to {channel}: {e}")
                results[channel] = False
        
        return results
    
    def _send_console(self, alert: Dict) -> bool:
        """Send alert to console."""
        print("\n" + "="*60)
        print(f"⚠️  GUST PRE-ALERT [{alert.get('alert_id', 'UNKNOWN')}]")
        print("="*60)
        print(f"📍 Location: {alert.get('location', 'Unknown')}")
        print(f"⏰ Time: {alert.get('timestamp', datetime.now().isoformat())}")
        print(f"🌪️  THD: {alert.get('thd_value', 0):.3f}")
        print(f"⏱️  Lead Time: {alert.get('lead_time_seconds', 0)} seconds")
        print(f"💨 Expected Gust: {alert.get('expected_gust_speed', 0):.1f} m/s")
        print(f"🎯 Confidence: {alert.get('confidence', 0):.1%}")
        print("="*60 + "\n")
        return True
    
    def _send_file(self, alert: Dict) -> bool:
        """Write alert to file."""
        date_str = datetime.now().strftime('%Y%m%d')
        log_file = self.log_dir / f"alerts_{date_str}.log"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'alert': alert
            }) + '\n')
        
        return True
    
    def _send_email(self, alert: Dict, recipients: List[str]) -> bool:
        """Send email alert."""
        if not recipients:
            return False
        
        try:
            # Create email message
            subject = f"⚠️ GUST PRE-ALERT: {alert.get('location', 'Unknown')}"
            
            body = f"""
GUST PRE-ALERT DETECTED
=======================

Location: {alert.get('location', 'Unknown')}
Time: {alert.get('timestamp', 'Unknown')}
THD Value: {alert.get('thd_value', 0):.3f}
Lead Time: {alert.get('lead_time_seconds', 0)} seconds
Expected Gust: {alert.get('expected_gust_speed', 0):.1f} m/s
Confidence: {alert.get('confidence', 0):.1%}

Alert ID: {alert.get('alert_id', 'UNKNOWN')}

--
AEROTICA Gust Pre-Alert System
            """
            
            # In production, use actual SMTP
            # with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            #     server.sendmail(self.email_from, recipients, message)
            
            self.logger.info(f"Email alert would be sent to {recipients}")
            return True
            
        except Exception as e:
            self.logger.error(f"Email sending failed: {e}")
            return False
    
    def _send_webhook(self, alert: Dict) -> bool:
        """Send webhook alert."""
        if not self.webhook_urls:
            return False
        
        success = True
        for url in self.webhook_urls:
            try:
                response = requests.post(
                    url,
                    json=alert,
                    headers={'Content-Type': 'application/json'},
                    timeout=5
                )
                if response.status_code >= 400:
                    success = False
                    self.logger.error(f"Webhook {url} returned {response.status_code}")
            except Exception as e:
                self.logger.error(f"Webhook {url} failed: {e}")
                success = False
        
        return success
    
    def _send_slack(self, alert: Dict) -> bool:
        """Send Slack notification."""
        if not self.slack_webhook:
            return False
        
        try:
            # Format for Slack
            color = "danger" if alert.get('confidence', 0) > 0.8 else "warning"
            
            slack_message = {
                "attachments": [{
                    "color": color,
                    "title": f"⚠️ Gust Pre-Alert: {alert.get('location', 'Unknown')}",
                    "fields": [
                        {
                            "title": "Lead Time",
                            "value": f"{alert.get('lead_time_seconds', 0)} seconds",
                            "short": True
                        },
                        {
                            "title": "Expected Gust",
                            "value": f"{alert.get('expected_gust_speed', 0):.1f} m/s",
                            "short": True
                        },
                        {
                            "title": "THD Value",
                            "value": f"{alert.get('thd_value', 0):.3f}",
                            "short": True
                        },
                        {
                            "title": "Confidence",
                            "value": f"{alert.get('confidence', 0):.1%}",
                            "short": True
                        }
                    ],
                    "footer": "AEROTICA Gust Detection System",
                    "ts": datetime.now().timestamp()
                }]
            }
            
            response = requests.post(
                self.slack_webhook,
                json=slack_message,
                timeout=5
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Slack notification failed: {e}")
            return False
    
    def _send_sms(self, alert: Dict, recipients: List[str]) -> bool:
        """Send SMS alert (using Twilio)."""
        if not self.twilio_account_sid or not recipients:
            return False
        
        try:
            # In production, use Twilio client
            # from twilio.rest import Client
            # client = Client(self.twilio_account_sid, self.twilio_auth_token)
            # 
            # for recipient in recipients:
            #     message = client.messages.create(
            #         body=f"⚠️ GUST ALERT: {alert.get('expected_gust_speed', 0):.1f} m/s in {alert.get('lead_time_seconds', 0)}s",
            #         from_=self.twilio_from_number,
            #         to=recipient
            #     )
            
            self.logger.info(f"SMS alert would be sent to {recipients}")
            return True
            
        except Exception as e:
            self.logger.error(f"SMS sending failed: {e}")
            return False
    
    def add_channel(self, channel: str, config: Optional[Dict] = None):
        """Add a notification channel."""
        if channel not in self.channels:
            self.channels.append(channel)
            if config:
                self.config.update(config)
    
    def remove_channel(self, channel: str):
        """Remove a notification channel."""
        if channel in self.channels:
            self.channels.remove(channel)
    
    def test_channels(self) -> Dict[str, bool]:
        """Test all configured channels."""
        test_alert = {
            'alert_id': 'TEST_ALERT',
            'location': 'Test Location',
            'timestamp': datetime.now().isoformat(),
            'thd_value': 0.75,
            'lead_time_seconds': 300,
            'expected_gust_speed': 25.5,
            'confidence': 0.85,
            'status': 'test'
        }
        
        return self.send_alert(test_alert, recipients=['test@aerotica.org'])
