"""Pressure Network - Maps (x,y,z,t) to pressure field p."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class PressureNetwork(nn.Module):
    """Neural network for pressure field approximation.
    
    Maps spatial-temporal coordinates (x, y, z, t) to pressure p.
    """
    
    def __init__(self,
                 input_dim: int = 4,  # (x, y, z, t)
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 activation: str = 'tanh'):
        """Initialize Pressure Network.
        
        Args:
            input_dim: Input dimension (default: 4 for x,y,z,t)
            hidden_dim: Hidden layer dimension (default: 512)
            num_layers: Number of hidden layers (default: 6)
            activation: Activation function
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                self._get_activation(activation),
                nn.BatchNorm1d(hidden_dim) if i % 3 == 0 else nn.Identity()
            ])
        self.hidden_layers = nn.ModuleList(layers)
        
        # Output layer (single pressure value)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, name: str):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'silu':
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            coords: Input coordinates (x, y, z, t) of shape (batch_size, 4)
            
        Returns:
            Pressure values of shape (batch_size, 1)
        """
        # Normalize inputs to [-1, 1] range
        coords_norm = 2 * (coords - coords.min(dim=0, keepdim=True)[0]) / \
                     (coords.max(dim=0, keepdim=True)[0] - coords.min(dim=0, keepdim=True)[0]) - 1
        
        # Input layer
        x = self.input_layer(coords_norm)
        x = torch.tanh(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Output layer (ensure positive pressure)
        pressure = torch.exp(self.output_layer(x))  # Exponential for positivity
        
        return pressure
    
    def get_pressure_gradient(self,
                             coords: torch.Tensor,
                             create_graph: bool = True) -> torch.Tensor:
        """Compute pressure gradient using automatic differentiation.
        
        Args:
            coords: Input coordinates
            create_graph: Whether to create computation graph
            
        Returns:
            Pressure gradient ∇p of shape (batch_size, 4)
        """
        coords.requires_grad_(True)
        pressure = self.forward(coords)
        
        gradient = torch.autograd.grad(
            pressure.sum(), coords,
            create_graph=create_graph,
            retain_graph=True
        )[0]
        
        return gradient
    
    def get_pressure_laplacian(self,
                              coords: torch.Tensor) -> torch.Tensor:
        """Compute pressure Laplacian ∇²p."""
        gradient = self.get_pressure_gradient(coords, create_graph=True)
        
        laplacian = 0
        for i in range(4):
            grad_component = gradient[:, i:i+1]
            laplacian += torch.autograd.grad(
                grad_component.sum(), coords,
                create_graph=True,
                retain_graph=True
            )[0][:, i:i+1]
        
        return laplacian
    
    def get_pressure_field(self,
                          x_grid: torch.Tensor,
                          y_grid: torch.Tensor,
                          z_grid: torch.Tensor,
                          t: float) -> torch.Tensor:
        """Get pressure field over a grid."""
        original_shape = x_grid.shape
        flat_coords = torch.stack([
            x_grid.flatten(),
            y_grid.flatten(),
            z_grid.flatten(),
            torch.full_like(x_grid.flatten(), t)
        ], dim=-1)
        
        pressure_flat = self.forward(flat_coords)
        pressure_field = pressure_flat.reshape(*original_shape)
        
        return pressure_field
