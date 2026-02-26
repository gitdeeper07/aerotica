"""Physics-Informed Neural Network module."""

from aerotica.pinn.velocity_net import VelocityNetwork
from aerotica.pinn.pressure_net import PressureNetwork
from aerotica.pinn.temperature_net import TemperatureNetwork
from aerotica.pinn.loss import NavierStokesLoss
from aerotica.pinn.inference import AeroticaPINN
from aerotica.pinn.trainer import PINNTrainer

__all__ = [
    'VelocityNetwork',
    'PressureNetwork',
    'TemperatureNetwork',
    'NavierStokesLoss',
    'AeroticaPINN',
    'PINNTrainer'
]
