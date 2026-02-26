"""Velocity Network - Maps (x,y,z,t) to (u,v,w) velocity components."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class FourierFeatureEmbedding(nn.Module):
    """Fourier feature embedding for multi-scale spatial representation."""
    
    def __init__(self, input_dim: int, mapping_size: int, scale: float = 1.0):
        super().__init__()
        self.mapping_size = mapping_size
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)
    
    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class VelocityNetwork(nn.Module):
    """Neural network for velocity field approximation.
    
    Maps spatial-temporal coordinates (x, y, z, t) to velocity components (u, v, w).
    Uses Fourier features for multi-scale resolution and 8-layer architecture.
    """
    
    def __init__(self,
                 input_dim: int = 4,  # (x, y, z, t)
                 output_dim: int = 3,  # (u, v, w)
                 hidden_dim: int = 512,
                 num_layers: int = 8,
                 fourier_scale: float = 1.0,
                 activation: str = 'tanh'):
        """Initialize Velocity Network.
        
        Args:
            input_dim: Input dimension (default: 4 for x,y,z,t)
            output_dim: Output dimension (default: 3 for u,v,w)
            hidden_dim: Hidden layer dimension (default: 512)
            num_layers: Number of hidden layers (default: 8)
            fourier_scale: Scale for Fourier features (default: 1.0)
            activation: Activation function ('tanh', 'relu', 'gelu')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Fourier feature embedding for multi-scale resolution
        self.fourier = FourierFeatureEmbedding(input_dim, hidden_dim // 4, fourier_scale)
        fourier_dim = hidden_dim // 2  # sin + cos
        
        # Input projection
        self.input_proj = nn.Linear(fourier_dim, hidden_dim)
        
        # Hidden layers
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(0.1) if i % 2 == 0 else nn.Identity()
            ])
        self.hidden_layers = nn.ModuleList(layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, name: str):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            coords: Input coordinates (x, y, z, t) of shape (batch_size, 4)
            
        Returns:
            Velocity components (u, v, w) of shape (batch_size, 3)
        """
        # Fourier features
        x_fourier = self.fourier(coords)
        
        # Input projection
        x = self.input_proj(x_fourier)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Output layer
        velocity = self.output_layer(x)
        
        return velocity
    
    def get_velocity(self, 
                    x: torch.Tensor,
                    y: torch.Tensor,
                    z: torch.Tensor,
                    t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get velocity components at specific points.
        
        Args:
            x, y, z: Spatial coordinates
            t: Time coordinate
            
        Returns:
            Tuple of (u, v, w) velocity components
        """
        coords = torch.stack([x, y, z, t], dim=-1)
        velocity = self.forward(coords)
        
        return velocity[..., 0], velocity[..., 1], velocity[..., 2]
    
    def get_velocity_field(self,
                          x_grid: torch.Tensor,
                          y_grid: torch.Tensor,
                          z_grid: torch.Tensor,
                          t: float) -> torch.Tensor:
        """Get velocity field over a grid.
        
        Args:
            x_grid, y_grid, z_grid: Meshgrid coordinates
            t: Time value
            
        Returns:
            Velocity field tensor of shape (nx, ny, nz, 3)
        """
        original_shape = x_grid.shape
        flat_coords = torch.stack([
            x_grid.flatten(),
            y_grid.flatten(),
            z_grid.flatten(),
            torch.full_like(x_grid.flatten(), t)
        ], dim=-1)
        
        velocity_flat = self.forward(flat_coords)
        velocity_field = velocity_flat.reshape(*original_shape, 3)
        
        return velocity_field
