"""Temperature Network - Maps (x,y,z,t) to potential temperature θ."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TemperatureNetwork(nn.Module):
    """Neural network for temperature field approximation.
    
    Maps spatial-temporal coordinates (x, y, z, t) to potential temperature θ.
    """
    
    def __init__(self,
                 input_dim: int = 4,  # (x, y, z, t)
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 dropout: float = 0.1):
        """Initialize Temperature Network.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        
        for i in range(num_layers):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            
            # Residual projection if dimensions differ
            self.residual_proj.append(
                nn.Identity() if i == 0 else nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            coords: Input coordinates (x, y, z, t) of shape (batch_size, 4)
            
        Returns:
            Potential temperature θ of shape (batch_size, 1)
        """
        # Input projection
        x = self.input_layer(coords)
        x = torch.relu(x)
        
        # Hidden layers with residual connections
        for i, layer in enumerate(self.hidden_layers):
            residual = self.residual_proj[i](x)
            x = layer(x) + residual
        
        # Output layer
        theta = self.output_layer(x)
        
        # Ensure physically reasonable temperature (250-320 K)
        theta = 250 + 70 * torch.sigmoid(theta)
        
        return theta
    
    def get_temperature_gradient(self,
                                coords: torch.Tensor,
                                create_graph: bool = True) -> torch.Tensor:
        """Compute temperature gradient ∇θ."""
        coords.requires_grad_(True)
        theta = self.forward(coords)
        
        gradient = torch.autograd.grad(
            theta.sum(), coords,
            create_graph=create_graph,
            retain_graph=True
        )[0]
        
        return gradient
    
    def get_buoyancy(self,
                    coords: torch.Tensor,
                    theta_ref: float = 300.0,
                    g: float = 9.81) -> torch.Tensor:
        """Compute buoyancy force.
        
        F_buoyancy = -g (θ - θ₀)/θ₀
        """
        theta = self.forward(coords)
        buoyancy = -g * (theta - theta_ref) / theta_ref
        return buoyancy
    
    def get_temperature_field(self,
                            x_grid: torch.Tensor,
                            y_grid: torch.Tensor,
                            z_grid: torch.Tensor,
                            t: float) -> torch.Tensor:
        """Get temperature field over a grid."""
        original_shape = x_grid.shape
        flat_coords = torch.stack([
            x_grid.flatten(),
            y_grid.flatten(),
            z_grid.flatten(),
            torch.full_like(x_grid.flatten(), t)
        ], dim=-1)
        
        theta_flat = self.forward(flat_coords)
        theta_field = theta_flat.reshape(*original_shape)
        
        return theta_field
    
    def get_potential_temperature_profile(self,
                                         z: torch.Tensor,
                                         t: float,
                                         x: float = 0.0,
                                         y: float = 0.0) -> torch.Tensor:
        """Get potential temperature profile at a point."""
        coords = torch.stack([
            torch.full_like(z, x),
            torch.full_like(z, y),
            z,
            torch.full_like(z, t)
        ], dim=-1)
        
        return self.forward(coords)
