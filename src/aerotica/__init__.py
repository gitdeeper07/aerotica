"""AEROTICA - Atmospheric Kinetic Energy Mapping Framework."""

__version__ = "1.0.0"
__author__ = "Samir Baladi"
__email__ = "gitdeeper@gmail.com"
__license__ = "MIT"

# Core components
from aerotica.ake import AKEComposite
from aerotica.parameters import (
    KED, TII, VSR, AOD, THD, PGF, HCI, ASI, LRC
)

# Try to import optional components
try:
    from aerotica.utils import (
        latlon_to_utm, haversine_distance,
        wind_rose_stats
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

__all__ = [
    '__version__', '__author__', '__email__', '__license__',
    'AKEComposite',
    'KED', 'TII', 'VSR', 'AOD', 'THD', 'PGF', 'HCI', 'ASI', 'LRC',
]
