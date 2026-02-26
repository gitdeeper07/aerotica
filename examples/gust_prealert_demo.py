#!/usr/bin/env python3
"""Gust pre-alerting demonstration."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from aerotica.alerts import GustPreAlertEngine


def generate_sample_data():
    """Generate sample observation data."""
    np.random.seed(42)
    
    n_points = 100
    timestamps = [datetime.now() + timedelta(seconds=i*10) 
                  for i in range(n_points)]
    
    # Base wind with gust signature
    t = np.linspace(0, 4*np.pi, n_points)
    wind_speed = 10 + 3 * np.sin(t) + np.random.randn(n_points) * 0.5
    
    # Add a gust event
    gust_start = 30
    wind_speed[gust_start:gust_start+10] += 8 * np.exp(-np.arange(10)/3)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'wind_speed': wind_speed,
        'wind_direction': 180 + 20 * np.sin(t/2),
        'temperature': 20 + 2 * np.sin(t/4),
        'pressure': 1013 + 2 * np.random.randn(n_points),
        'humidity': 70 + 5 * np.random.randn(n_points)
    })
    
    return data


def main():
    """Run gust pre-alert demo."""
    print("="*60)
    print("AEROTICA Gust Pre-Alert Demonstration")
    print("="*60)
    
    # Generate sample data
    print("\n📊 Generating sample observation data...")
    data = generate_sample_data()
    print(f"   {len(data)} observations generated")
    
    # Initialize alert engine
    print("\n🚀 Initializing gust pre-alert engine...")
    engine = GustPreAlertEngine(
        site_config={'location': 'demo_site'},
        thd_threshold=0.7
    )
    
    # Process data in real-time
    print("\n🔴 Processing observations (simulating real-time)...")
    
    alerts = []
    for i in range(0, len(data), 5):
        window = data.iloc[:i+1] if i > 0 else data.iloc[:1]
        
        alert = engine.evaluate(window)
        
        if alert:
            alerts.append(alert)
            print(f"\n⚠️  GUST PRE-ALERT at step {i}")
            print(f"   Lead time: {alert['lead_time_seconds']} seconds")
            print(f"   Expected gust: {alert['expected_gust_speed']:.1f} m/s")
            print(f"   Confidence: {alert['confidence']:.1%}")
    
    # Summary
    print("\n" + "="*60)
    print(f"✅ Demo completed!")
    print(f"   Total alerts: {len(alerts)}")
    
    if alerts:
        print(f"   Average lead time: {np.mean([a['lead_time_seconds'] for a in alerts]):.0f} seconds")
        print(f"   Average confidence: {np.mean([a['confidence'] for a in alerts]):.1%}")


if __name__ == "__main__":
    main()
