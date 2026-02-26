"""Configuration management for AEROTICA."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


class AEROTICAConfig:
    """AEROTICA configuration manager."""
    
    DEFAULT_CONFIG = {
        'server': {
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 4,
            'timeout': 60
        },
        'database': {
            'url': 'postgresql://aerotica:${DB_PASSWORD}@localhost:5432/aerotica',
            'pool_size': 10,
            'max_overflow': 20
        },
        'pinn': {
            'model_path': 'models/pinn_v1/',
            'device': 'cpu',
            'batch_size': 32,
            'inference_workers': 2
        },
        'parameters': {
            'ked': {'weight': 0.22, 'enabled': True, 'air_density': 1.225},
            'tii': {'weight': 0.16, 'enabled': True, 'sampling_freq': 1.0},
            'vsr': {'weight': 0.14, 'enabled': True, 'stability_correction': True},
            'aod': {'weight': 0.12, 'enabled': True, 'wavelength': 550},
            'thd': {'weight': 0.10, 'enabled': True, 'threshold': 0.7},
            'pgf': {'weight': 0.08, 'enabled': True},
            'hci': {'weight': 0.07, 'enabled': True},
            'asi': {'weight': 0.06, 'enabled': True},
            'lrc': {'weight': 0.05, 'enabled': True, 'resolution': 2.0}
        },
        'alerts': {
            'enabled': True,
            'check_interval': 30,
            'thd_threshold': 0.7,
            'notification_channels': [
                {'type': 'console'},
                {'type': 'file', 'path': 'logs/alerts.log'}
            ]
        },
        'monitoring': {
            'metrics_port': 9090,
            'health_check_interval': 30,
            'log_level': 'INFO'
        },
        'data': {
            'raw': 'data/raw/',
            'processed': 'data/processed/',
            'reference': 'data/reference/',
            'realtime': 'data/realtime/'
        },
        'climate_zones': [
            'tropical', 'arid', 'temperate',
            'continental', 'polar', 'high_altitude'
        ]
    }
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize configuration.
        
        Args:
            config_file: Path to configuration file (YAML or JSON)
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from file if provided
        if config_file and config_file.exists():
            self.load(config_file)
        
        # Override with environment variables
        self._load_from_env()
    
    def load(self, config_file: Path):
        """Load configuration from file."""
        suffix = config_file.suffix.lower()
        
        with open(config_file, 'r') as f:
            if suffix == '.yaml' or suffix == '.yml':
                loaded = yaml.safe_load(f)
            elif suffix == '.json':
                loaded = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
        
        # Deep update
        self._deep_update(self.config, loaded)
    
    def _deep_update(self, base: Dict, update: Dict):
        """Recursively update dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Database
        if 'DB_PASSWORD' in os.environ:
            self.config['database']['url'] = self.config['database']['url'].replace(
                '${DB_PASSWORD}', os.environ['DB_PASSWORD']
            )
        
        # Server
        if 'SERVER_HOST' in os.environ:
            self.config['server']['host'] = os.environ['SERVER_HOST']
        if 'SERVER_PORT' in os.environ:
            self.config['server']['port'] = int(os.environ['SERVER_PORT'])
        
        # PINN device
        if 'PINN_DEVICE' in os.environ:
            self.config['pinn']['device'] = os.environ['PINN_DEVICE']
        
        # Log level
        if 'LOG_LEVEL' in os.environ:
            self.config['monitoring']['log_level'] = os.environ['LOG_LEVEL']
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key.
        
        Example:
            config.get('server.host')
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by dot-separated key."""
        keys = key.split('.')
        target = self.config
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        target[keys[-1]] = value
    
    def save(self, config_file: Path):
        """Save configuration to file."""
        suffix = config_file.suffix.lower()
        
        with open(config_file, 'w') as f:
            if suffix == '.yaml' or suffix == '.yml':
                yaml.dump(self.config, f, default_flow_style=False)
            elif suffix == '.json':
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return self.config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Dictionary-style assignment."""
        self.set(key, value)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"AEROTICAConfig({json.dumps(self.config, indent=2)})"


# Global configuration instance
config = AEROTICAConfig()


def get_config() -> AEROTICAConfig:
    """Get global configuration instance."""
    return config


def load_config(config_file: Path) -> AEROTICAConfig:
    """Load configuration from file and update global instance."""
    config.load(config_file)
    return config
