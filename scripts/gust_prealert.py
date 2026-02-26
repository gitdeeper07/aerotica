#!/usr/bin/env python3
"""Real-time gust pre-alerting script."""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aerotica.alerts import GustPreAlertEngine
from aerotica.pinn import AeroticaPINN


class MockDataStream:
    """Mock data stream for testing."""
    
    def __init__(self, site: str, data_file: Optional[Path] = None):
        self.site = site
        self.data_file = data_file
        self.index = 0
        
        if data_file and data_file.exists():
            self.data = pd.read_csv(data_file)
        else:
            # Generate mock data
            self.data = self._generate_mock_data()
    
    def _generate_mock_data(self) -> pd.DataFrame:
        """Generate mock observation data."""
        np.random.seed(42)
        
        n_points = 1000
        timestamps = pd.date_range(
            start=datetime.now() - pd.Timedelta(hours=1),
            periods=n_points,
            freq='1s'
        )
        
        # Base wind with gusts
        base_wind = 10 + 2 * np.sin(np.linspace(0, 4*np.pi, n_points))
        gusts = np.random.gamma(2, 2, n_points) * 2
        wind_speed = base_wind + gusts
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'wind_speed': wind_speed,
            'wind_direction': 180 + 30 * np.sin(np.linspace(0, 2*np.pi, n_points)),
            'temperature': 20 + 5 * np.sin(np.linspace(0, 2*np.pi, n_points)),
            'pressure': 1013 + 10 * np.random.randn(n_points),
            'humidity': 70 + 10 * np.random.randn(n_points)
        })
        
        return data
    
    def get_latest(self) -> pd.DataFrame:
        """Get latest observations."""
        if self.index >= len(self.data):
            self.index = 0
        
        # Return window of last 10 observations
        start = max(0, self.index - 10)
        window = self.data.iloc[start:self.index+1]
        
        self.index += 1
        return window


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run gust pre-alerting')
    parser.add_argument('--site', required=True, help='Site identifier')
    parser.add_argument('--config', type=Path, help='Configuration file')
    parser.add_argument('--interval', type=int, default=30,
                       help='Update interval in seconds')
    parser.add_argument('--data-file', type=Path, help='Data file for replay')
    parser.add_argument('--pinn-model', type=Path, help='PINN model directory')
    parser.add_argument('--output', type=Path, help='Output file for alerts')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"⚠️  AEROTICA Gust Pre-Alert System")
    print(f"📍 Site: {args.site}")
    print(f"⏱️  Update interval: {args.interval} seconds")
    print("="*50)
    
    # Load configuration
    config = {}
    if args.config and args.config.exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded config from {args.config}")
    
    # Load PINN model if provided
    pinn_model = None
    if args.pinn_model and args.pinn_model.exists():
        try:
            pinn_model = AeroticaPINN.from_pretrained(
                args.pinn_model,
                climate_zone=config.get('climate_zone', 'temperate')
            )
            print(f"✅ Loaded PINN model from {args.pinn_model}")
        except Exception as e:
            print(f"⚠️  Failed to load PINN model: {e}")
    
    # Initialize alert engine
    engine = GustPreAlertEngine(
        pinn_model=pinn_model,
        site_config={'location': args.site, **config},
        thd_threshold=config.get('thd_threshold', 0.7),
        check_interval=args.interval
    )
    
    # Initialize data stream
    data_stream = MockDataStream(args.site, args.data_file)
    
    # Output file
    output_file = None
    if args.output:
        output_file = open(args.output, 'w')
        output_file.write('timestamp,alert_id,thd,lead_time_seconds,gust_speed\n')
    
    try:
        print("\n🔴 Monitoring started. Press Ctrl+C to stop.\n")
        
        while True:
            # Get observations
            obs = data_stream.get_latest()
            
            # Evaluate
            alert = engine.evaluate(obs)
            
            if alert:
                # Dispatch to console
                engine.dispatch(alert, recipients=['console'])
                
                # Write to output file
                if output_file:
                    output_file.write(
                        f"{alert['timestamp']},"
                        f"{alert['alert_id']},"
                        f"{alert['thd_value']:.3f},"
                        f"{alert['lead_time_seconds']},"
                        f"{alert['expected_gust_speed']:.1f}\n"
                    )
                    output_file.flush()
            
            # Wait for next interval
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
        
        # Print statistics
        stats = engine.get_stats()
        print(f"\n📊 Alert Statistics:")
        print(f"   Total alerts: {stats['total_alerts']}")
        print(f"   Active alerts: {stats['active_alerts']}")
        print(f"   Probability of detection: {stats['probability_of_detection']:.3f}")
        print(f"   False alarm rate: {stats['false_alarm_rate']:.3f}")
        
        # Close output file
        if output_file:
            output_file.close()
            print(f"\n💾 Alerts saved to {args.output}")


if __name__ == '__main__':
    main()
