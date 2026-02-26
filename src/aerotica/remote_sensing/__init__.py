"""Remote sensing data processing module."""

from aerotica.remote_sensing.modis import MODISProcessor
from aerotica.remote_sensing.satellite import SatelliteProcessor

__all__ = [
    'MODISProcessor',
    'SatelliteProcessor'
]
