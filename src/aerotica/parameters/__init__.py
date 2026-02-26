"""AKE Parameters Module.

This module contains the nine parameters of the AKE index:
- KED: Kinetic Energy Density (22%)
- TII: Turbulence Intensity Index (16%)
- VSR: Vertical Shear Ratio (14%)
- AOD: Aerosol Optical Depth (12%)
- THD: Thermal Helicity Dynamics (10%)
- PGF: Pressure Gradient Force (8%)
- HCI: Humidity-Convection Interaction (7%)
- ASI: Atmospheric Stability Integration (6%)
- LRC: Local Roughness Coefficient (5%)
"""

from aerotica.parameters.ked import KED
from aerotica.parameters.tii import TII
from aerotica.parameters.vsr import VSR
from aerotica.parameters.aod import AOD
from aerotica.parameters.thd import THD
from aerotica.parameters.pgf import PGF
from aerotica.parameters.hci import HCI
from aerotica.parameters.asi import ASI
from aerotica.parameters.lrc import LRC

__all__ = [
    'KED', 'TII', 'VSR', 'AOD', 'THD',
    'PGF', 'HCI', 'ASI', 'LRC'
]
