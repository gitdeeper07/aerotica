"""PINN Inference Module."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import xarray as xr

from aerotica.pinn.velocity_net import VelocityNetwork
from aerotica.pinn.pressure_net import PressureNetwork
from aerotica.pinn.temperature_net import TemperatureNetwork


class AeroticaPINN:
    """AEROTICA Physics-Informed Neural Network for wind field prediction."""
    
    def __init__(self,
                 velocity_net: nn.Module,
                 pressure_net: nn.Module,
                 temperature_net: nn.Module,
                 device: str = 'cuda'):
        """Initialize PINN.
        
        Args:
            velocity_net: Velocity network
            pressure_net: Pressure network
            temperature_net: Temperature network
            device: Device for inference
        """
        self.velocity_net = velocity_net.to(device).eval()
        self.pressure_net = pressure_net.to(device).eval()
        self.temperature_net = temperature_net.to(device).eval()
        self.device = device
    
    @classmethod
    def from_pretrained(cls,
                       model_dir: Path,
                       climate_zone: str = 'temperate',
                       device: str = 'cuda'):
        """Load pre-trained model.
        
        Args:
            model_dir: Directory with model weights
            climate_zone: Climate zone for model selection
            device: Device for inference
            
        Returns:
            Loaded PINN model
        """
        model_dir = Path(model_dir)
        
        # Initialize networks
        velocity_net = VelocityNetwork()
        pressure_net = PressureNetwork()
        temperature_net = TemperatureNetwork()
        
        # Load weights
        velocity_net.load_state_dict(
            torch.load(model_dir / f'velocity_net_{climate_zone}.pt',
                      map_location=device)
        )
        pressure_net.load_state_dict(
            torch.load(model_dir / f'pressure_net_{climate_zone}.pt',
                      map_location=device)
        )
        temperature_net.load_state_dict(
            torch.load(model_dir / f'temperature_net_{climate_zone}.pt',
                      map_location=device)
        )
        
        return cls(velocity_net, pressure_net, temperature_net, device)
    
    def infer(self,
             surface_obs: Optional[xr.Dataset] = None,
             radiosonde: Optional[xr.Dataset] = None,
             radar: Optional[xr.Dataset] = None,
             aod_field: Optional[xr.DataArray] = None,
             domain_size: Tuple[float, float, float] = (50e3, 50e3, 2000),
             resolution: Tuple[int, int, int] = (500, 500, 40),
             time: float = 0.0) -> Dict[str, Any]:
        """Infer 3D wind field.
        
        Args:
            surface_obs: Surface observations
            radiosonde: Radiosonde profile data
            radar: Radar data
            aod_field: Aerosol Optical Depth field
            domain_size: Domain size (x, y, z) in meters
            resolution: Grid resolution (nx, ny, nz)
            time: Current time
            
        Returns:
            Dictionary with inferred fields
        """
        # Create grid
        x = torch.linspace(-domain_size[0]/2, domain_size[0]/2, resolution[0])
        y = torch.linspace(-domain_size[1]/2, domain_size[1]/2, resolution[1])
        z = torch.linspace(0, domain_size[2], resolution[2])
        
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        # Flatten coordinates
        coords = torch.stack([
            X.flatten(), Y.flatten(), Z.flatten(),
            torch.full_like(X.flatten(), time)
        ], dim=-1).to(self.device)
        
        # Infer fields
        with torch.no_grad():
            # Velocity
            velocity = self.velocity_net(coords)
            u = velocity[:, 0].reshape(X.shape)
            v = velocity[:, 1].reshape(X.shape)
            w = velocity[:, 2].reshape(X.shape)
            
            # Pressure
            pressure = self.pressure_net(coords).reshape(X.shape)
            
            # Temperature
            temperature = self.temperature_net(coords).reshape(X.shape)
        
        # Compute derived quantities
        wind_speed = torch.sqrt(u**2 + v**2 + w**2)
        
        # AKE map (simplified)
        ked = 0.5 * 1.225 * wind_speed**3
        ked_norm = (ked - ked.min()) / (ked.max() - ked.min() + 1e-8)
        
        # Return as numpy arrays
        result = {
            'u': u.cpu().numpy(),
            'v': v.cpu().numpy(),
            'w': w.cpu().numpy(),
            'pressure': pressure.cpu().numpy(),
            'temperature': temperature.cpu().numpy(),
            'wind_speed': wind_speed.cpu().numpy(),
            'ake_map': ked_norm.cpu().numpy(),
            'x': x.cpu().numpy(),
            'y': y.cpu().numpy(),
            'z': z.cpu().numpy(),
            'time': time
        }
        
        return result
    
    def infer_point(self,
                   x: float, y: float, z: float, t: float) -> Dict[str, float]:
        """Infer fields at a single point."""
        coords = torch.tensor([[x, y, z, t]]).to(self.device)
        
        with torch.no_grad():
            velocity = self.velocity_net(coords)[0]
            pressure = self.pressure_net(coords)[0, 0]
            temperature = self.temperature_net(coords)[0, 0]
        
        return {
            'u': velocity[0].item(),
            'v': velocity[1].item(),
            'w': velocity[2].item(),
            'pressure': pressure.item(),
            'temperature': temperature.item(),
            'wind_speed': torch.norm(velocity).item()
        }
    
    def infer_profile(self,
                     z: np.ndarray,
                     x: float = 0.0,
                     y: float = 0.0,
                     t: float = 0.0) -> Dict[str, np.ndarray]:
        """Infer vertical profile at a point."""
        z_tensor = torch.tensor(z, device=self.device)
        x_tensor = torch.full_like(z_tensor, x)
        y_tensor = torch.full_like(z_tensor, y)
        t_tensor = torch.full_like(z_tensor, t)
        
        coords = torch.stack([x_tensor, y_tensor, z_tensor, t_tensor], dim=-1)
        
        with torch.no_grad():
            velocity = self.velocity_net(coords)
            pressure = self.pressure_net(coords)
            temperature = self.temperature_net(coords)
        
        return {
            'z': z,
            'u': velocity[:, 0].cpu().numpy(),
            'v': velocity[:, 1].cpu().numpy(),
            'w': velocity[:, 2].cpu().numpy(),
            'pressure': pressure[:, 0].cpu().numpy(),
            'temperature': temperature[:, 0].cpu().numpy(),
            'wind_speed': torch.norm(velocity, dim=1).cpu().numpy()
        }
    
    def to_xarray(self, result: Dict[str, Any]) -> xr.Dataset:
        """Convert inference result to xarray Dataset."""
        ds = xr.Dataset(
            data_vars={
                'u': (['x', 'y', 'z'], result['u']),
                'v': (['x', 'y', 'z'], result['v']),
                'w': (['x', 'y', 'z'], result['w']),
                'pressure': (['x', 'y', 'z'], result['pressure']),
                'temperature': (['x', 'y', 'z'], result['temperature']),
                'wind_speed': (['x', 'y', 'z'], result['wind_speed']),
                'ake': (['x', 'y'], result['ake_map'][:, :, 0])  # Surface AKE
            },
            coords={
                'x': result['x'],
                'y': result['y'],
                'z': result['z'],
                'time': result['time']
            },
            attrs={
                'description': 'AEROTICA PINN inference result',
                'domain_size': '50km x 50km x 2km',
                'resolution': f"{len(result['x'])}x{len(result['y'])}x{len(result['z'])}"
            }
        )
        return ds
