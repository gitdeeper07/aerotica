"""Physics-Informed Loss Functions for Navier-Stokes constraints."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


class NavierStokesLoss(nn.Module):
    """Physics-informed loss enforcing Navier-Stokes equations.
    
    Computes residuals of:
    - Momentum equation
    - Continuity equation
    - Boundary conditions
    - Initial conditions
    """
    
    def __init__(self,
                 rho: float = 1.225,  # Air density [kg/m³]
                 mu: float = 1.8e-5,  # Dynamic viscosity [Pa·s]
                 g: float = 9.81,     # Gravity [m/s²]
                 omega: float = 7.2921e-5,  # Earth rotation rate [rad/s]
                 latitude: float = 45.0):   # Latitude [degrees]
        super().__init__()
        
        self.rho = rho
        self.mu = mu
        self.g = g
        self.omega = omega
        self.latitude = latitude
        
        # Coriolis parameter
        self.f = 2 * omega * np.sin(np.radians(latitude))
        
        # Loss weights (adaptive)
        self.register_buffer('weight_data', torch.tensor(1.0))
        self.register_buffer('weight_ns', torch.tensor(1.0))
        self.register_buffer('weight_continuity', torch.tensor(1.0))
        self.register_buffer('weight_bc', torch.tensor(1.0))
        self.register_buffer('weight_ic', torch.tensor(1.0))
    
    def forward(self,
               velocity_net: nn.Module,
               pressure_net: nn.Module,
               temperature_net: nn.Module,
               coords: torch.Tensor,
               observed_data: Optional[Dict] = None,
               boundary_coords: Optional[torch.Tensor] = None,
               boundary_values: Optional[torch.Tensor] = None,
               initial_coords: Optional[torch.Tensor] = None,
               initial_values: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute total physics-informed loss.
        
        Args:
            velocity_net: Velocity network
            pressure_net: Pressure network
            temperature_net: Temperature network
            coords: Collocation points for PDE residuals
            observed_data: Dictionary with observed values
            boundary_coords: Boundary points
            boundary_values: Boundary values
            initial_coords: Initial condition points
            initial_values: Initial values
            
        Returns:
            Dictionary with loss components
        """
        coords.requires_grad_(True)
        
        # Get network outputs
        velocity = velocity_net(coords)  # (batch, 3)
        pressure = pressure_net(coords)  # (batch, 1)
        temperature = temperature_net(coords)  # (batch, 1)
        
        # Split velocity components
        u = velocity[:, 0:1]
        v = velocity[:, 1:2]
        w = velocity[:, 2:3]
        
        # Compute gradients for Navier-Stokes
        gradients = self._compute_gradients(u, v, w, pressure, temperature, coords)
        
        # Compute loss components
        losses = {}
        
        # Data loss (if observations available)
        if observed_data is not None:
            losses['data'] = self._data_loss(velocity, pressure, temperature, observed_data)
        else:
            losses['data'] = torch.tensor(0.0, device=coords.device)
        
        # Navier-Stokes momentum residuals
        ns_loss = self._navier_stokes_residual(u, v, w, pressure, temperature, gradients, coords)
        losses['navier_stokes'] = ns_loss
        
        # Continuity equation (incompressibility)
        continuity_loss = self._continuity_residual(gradients)
        losses['continuity'] = continuity_loss
        
        # Boundary conditions
        if boundary_coords is not None and boundary_values is not None:
            losses['boundary'] = self._boundary_loss(
                velocity_net, pressure_net, temperature_net,
                boundary_coords, boundary_values
            )
        else:
            losses['boundary'] = torch.tensor(0.0, device=coords.device)
        
        # Initial conditions
        if initial_coords is not None and initial_values is not None:
            losses['initial'] = self._initial_loss(
                velocity_net, pressure_net, temperature_net,
                initial_coords, initial_values
            )
        else:
            losses['initial'] = torch.tensor(0.0, device=coords.device)
        
        # Weighted total loss
        total_loss = (
            self.weight_data * losses['data'] +
            self.weight_ns * losses['navier_stokes'] +
            self.weight_continuity * losses['continuity'] +
            self.weight_bc * losses['boundary'] +
            self.weight_ic * losses['initial']
        )
        
        losses['total'] = total_loss
        
        return losses
    
    def _compute_gradients(self,
                          u: torch.Tensor,
                          v: torch.Tensor,
                          w: torch.Tensor,
                          p: torch.Tensor,
                          T: torch.Tensor,
                          coords: torch.Tensor) -> Dict:
        """Compute spatial and temporal gradients."""
        batch_size = u.shape[0]
        gradients = {}
        
        # Time derivative (assuming first column is t)
        dt = coords[:, 3:4]
        
        # Compute gradients using autograd
        def compute_grad(y, x):
            return torch.autograd.grad(y, x,
                                      grad_outputs=torch.ones_like(y),
                                      create_graph=True,
                                      retain_graph=True)[0]
        
        # Velocity gradients
        grad_u = compute_grad(u, coords)
        grad_v = compute_grad(v, coords)
        grad_w = compute_grad(w, coords)
        
        gradients['u_t'] = grad_u[:, 3:4]
        gradients['u_x'] = grad_u[:, 0:1]
        gradients['u_y'] = grad_u[:, 1:2]
        gradients['u_z'] = grad_u[:, 2:3]
        
        gradients['v_t'] = grad_v[:, 3:4]
        gradients['v_x'] = grad_v[:, 0:1]
        gradients['v_y'] = grad_v[:, 1:2]
        gradients['v_z'] = grad_v[:, 2:3]
        
        gradients['w_t'] = grad_w[:, 3:4]
        gradients['w_x'] = grad_w[:, 0:1]
        gradients['w_y'] = grad_w[:, 1:2]
        gradients['w_z'] = grad_w[:, 2:3]
        
        # Pressure gradients
        grad_p = compute_grad(p, coords)
        gradients['p_x'] = grad_p[:, 0:1]
        gradients['p_y'] = grad_p[:, 1:2]
        gradients['p_z'] = grad_p[:, 2:3]
        
        # Temperature gradients
        grad_T = compute_grad(T, coords)
        gradients['T_x'] = grad_T[:, 0:1]
        gradients['T_y'] = grad_T[:, 1:2]
        gradients['T_z'] = grad_T[:, 2:3]
        gradients['T_t'] = grad_T[:, 3:4]
        
        # Second derivatives for viscosity
        grad_u_x = compute_grad(grad_u[:, 0:1], coords)
        grad_u_y = compute_grad(grad_u[:, 1:2], coords)
        grad_u_z = compute_grad(grad_u[:, 2:3], coords)
        
        gradients['u_xx'] = grad_u_x[:, 0:1]
        gradients['u_yy'] = grad_u_y[:, 1:2]
        gradients['u_zz'] = grad_u_z[:, 2:3]
        
        return gradients
    
    def _navier_stokes_residual(self,
                                u: torch.Tensor,
                                v: torch.Tensor,
                                w: torch.Tensor,
                                p: torch.Tensor,
                                T: torch.Tensor,
                                grad: Dict,
                                coords: torch.Tensor) -> torch.Tensor:
        """Compute Navier-Stokes momentum residuals.
        
        ρ(∂u/∂t + u·∇u) = −∇p + μ∇²u + ρg + F_Coriolis + F_buoyancy
        """
        # Material derivative: Du/Dt = ∂u/∂t + u·∇u
        u_t = grad['u_t']
        u_x = grad['u_x']
        u_y = grad['u_y']
        u_z = grad['u_z']
        
        v_t = grad['v_t']
        v_x = grad['v_x']
        v_y = grad['v_y']
        v_z = grad['v_z']
        
        w_t = grad['w_t']
        w_x = grad['w_x']
        w_y = grad['w_y']
        w_z = grad['w_z']
        
        # Material derivatives
        Du_Dt = u_t + u * u_x + v * u_y + w * u_z
        Dv_Dt = v_t + u * v_x + v * v_y + w * v_z
        Dw_Dt = w_t + u * w_x + v * w_y + w * w_z
        
        # Pressure gradients
        p_x = grad['p_x']
        p_y = grad['p_y']
        p_z = grad['p_z']
        
        # Viscous terms (Laplacian)
        laplacian_u = grad['u_xx'] + grad['u_yy'] + grad['u_zz']
        laplacian_v = grad.get('v_xx', torch.zeros_like(u)) + \
                      grad.get('v_yy', torch.zeros_like(u)) + \
                      grad.get('v_zz', torch.zeros_like(u))
        laplacian_w = grad.get('w_xx', torch.zeros_like(u)) + \
                      grad.get('w_yy', torch.zeros_like(u)) + \
                      grad.get('w_zz', torch.zeros_like(u))
        
        # Coriolis force
        F_coriolis_x = -self.f * v
        F_coriolis_y = self.f * u
        F_coriolis_z = 0.0
        
        # Buoyancy force (using potential temperature)
        theta_ref = 300.0
        F_buoyancy = -self.g * (T - theta_ref) / theta_ref
        
        # Momentum residuals
        residual_x = self.rho * Du_Dt + p_x - self.mu * laplacian_u - F_coriolis_x
        residual_y = self.rho * Dv_Dt + p_y - self.mu * laplacian_v - F_coriolis_y
        residual_z = self.rho * Dw_Dt + p_z - self.mu * laplacian_w - self.rho * self.g - F_buoyancy
        
        # Mean squared residual
        residual = torch.mean(residual_x**2 + residual_y**2 + residual_z**2)
        
        return residual
    
    def _continuity_residual(self, grad: Dict) -> torch.Tensor:
        """Compute continuity equation residual.
        
        ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
        """
        div_u = grad['u_x'] + grad['v_y'] + grad['w_z']
        residual = torch.mean(div_u**2)
        return residual
    
    def _data_loss(self,
                  velocity: torch.Tensor,
                  pressure: torch.Tensor,
                  temperature: torch.Tensor,
                  observed: Dict) -> torch.Tensor:
        """Compute loss against observed data."""
        loss = 0.0
        
        if 'velocity' in observed:
            loss += torch.mean((velocity - observed['velocity'])**2)
        
        if 'pressure' in observed:
            loss += torch.mean((pressure - observed['pressure'])**2)
        
        if 'temperature' in observed:
            loss += torch.mean((temperature - observed['temperature'])**2)
        
        return loss
    
    def _boundary_loss(self,
                      velocity_net: nn.Module,
                      pressure_net: nn.Module,
                      temperature_net: nn.Module,
                      boundary_coords: torch.Tensor,
                      boundary_values: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss."""
        velocity = velocity_net(boundary_coords)
        pressure = pressure_net(boundary_coords)
        temperature = temperature_net(boundary_coords)
        
        loss = 0.0
        if 'velocity' in boundary_values:
            loss += torch.mean((velocity - boundary_values['velocity'])**2)
        if 'pressure' in boundary_values:
            loss += torch.mean((pressure - boundary_values['pressure'])**2)
        if 'temperature' in boundary_values:
            loss += torch.mean((temperature - boundary_values['temperature'])**2)
        
        return loss
    
    def _initial_loss(self,
                     velocity_net: nn.Module,
                     pressure_net: nn.Module,
                     temperature_net: nn.Module,
                     initial_coords: torch.Tensor,
                     initial_values: torch.Tensor) -> torch.Tensor:
        """Compute initial condition loss."""
        return self._boundary_loss(
            velocity_net, pressure_net, temperature_net,
            initial_coords, initial_values
        )
    
    def update_weights(self, losses: Dict[str, torch.Tensor]):
        """Adaptively update loss weights using NTK-based algorithm."""
        # Simplified adaptive weighting
        with torch.no_grad():
            total = losses['total']
            for key in ['data', 'navier_stokes', 'continuity', 'boundary', 'initial']:
                if key in losses and losses[key] > 0:
                    current_weight = getattr(self, f'weight_{key}')
                    ratio = losses[key] / (total + 1e-8)
                    new_weight = current_weight * (1.0 + 0.1 * (ratio - 0.2))
                    setattr(self, f'weight_{key}', torch.clamp(new_weight, 0.1, 10.0))
