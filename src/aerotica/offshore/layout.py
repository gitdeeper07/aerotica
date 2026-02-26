"""Turbine Layout Optimization Module."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import itertools

from aerotica.offshore.wake_model import WakeModel, Turbine


@dataclass
class LayoutConfig:
    """Turbine layout configuration."""
    n_turbines: int
    min_spacing: float  # Minimum spacing in rotor diameters
    max_spacing: float  # Maximum spacing in rotor diameters
    boundary_x: Tuple[float, float]  # x bounds [m]
    boundary_y: Tuple[float, float]  # y bounds [m]
    rotor_diameter: float = 236.0  # Next-gen 15MW turbine
    hub_height: float = 150.0
    rated_power: float = 15000.0  # kW


class TurbineLayout:
    """Optimize turbine layout for maximum energy production."""
    
    def __init__(self,
                 config: LayoutConfig,
                 wake_model: Optional[WakeModel] = None):
        """Initialize layout optimizer.
        
        Args:
            config: Layout configuration
            wake_model: Wake model instance
        """
        self.config = config
        self.wake_model = wake_model or WakeModel()
        
        # Current layout
        self.turbines: List[Turbine] = []
        
        # Performance metrics
        self.aep = 0.0  # Annual Energy Production [MWh]
        self.wake_losses = 0.0  # Wake loss fraction
    
    def generate_initial_layout(self, pattern: str = 'grid') -> List[Turbine]:
        """Generate initial turbine layout.
        
        Args:
            pattern: Layout pattern ('grid', 'staggered', 'random')
            
        Returns:
            List of turbines
        """
        turbines = []
        
        if pattern == 'grid':
            turbines = self._grid_layout()
        elif pattern == 'staggered':
            turbines = self._staggered_layout()
        elif pattern == 'random':
            turbines = self._random_layout()
        
        self.turbines = turbines
        return turbines
    
    def _grid_layout(self) -> List[Turbine]:
        """Generate regular grid layout."""
        spacing = self.config.min_spacing * self.config.rotor_diameter
        
        nx = int(np.sqrt(self.config.n_turbines))
        ny = (self.config.n_turbines + nx - 1) // nx
        
        x_start = (self.config.boundary_x[0] + self.config.boundary_x[1]) / 2 - (nx - 1) * spacing / 2
        y_start = (self.config.boundary_y[0] + self.config.boundary_y[1]) / 2 - (ny - 1) * spacing / 2
        
        turbines = []
        for i in range(nx):
            for j in range(ny):
                if len(turbines) >= self.config.n_turbines:
                    break
                
                x = x_start + i * spacing
                y = y_start + j * spacing
                
                # Check bounds
                if (self.config.boundary_x[0] <= x <= self.config.boundary_x[1] and
                    self.config.boundary_y[0] <= y <= self.config.boundary_y[1]):
                    
                    turbines.append(Turbine(
                        x=x,
                        y=y,
                        hub_height=self.config.hub_height,
                        rotor_diameter=self.config.rotor_diameter,
                        rated_power=self.config.rated_power,
                        thrust_coefficient=0.8
                    ))
        
        return turbines
    
    def _staggered_layout(self) -> List[Turbine]:
        """Generate staggered (offset) grid layout."""
        spacing = self.config.min_spacing * self.config.rotor_diameter
        offset = spacing / 2
        
        nx = int(np.sqrt(self.config.n_turbines))
        ny = (self.config.n_turbines + nx - 1) // nx
        
        x_start = (self.config.boundary_x[0] + self.config.boundary_x[1]) / 2 - (nx - 1) * spacing / 2
        y_start = (self.config.boundary_y[0] + self.config.boundary_y[1]) / 2 - (ny - 1) * spacing / 2
        
        turbines = []
        for j in range(ny):
            for i in range(nx):
                if len(turbines) >= self.config.n_turbines:
                    break
                
                x = x_start + i * spacing + (offset if j % 2 == 1 else 0)
                y = y_start + j * spacing
                
                # Check bounds
                if (self.config.boundary_x[0] <= x <= self.config.boundary_x[1] and
                    self.config.boundary_y[0] <= y <= self.config.boundary_y[1]):
                    
                    turbines.append(Turbine(
                        x=x,
                        y=y,
                        hub_height=self.config.hub_height,
                        rotor_diameter=self.config.rotor_diameter,
                        rated_power=self.config.rated_power,
                        thrust_coefficient=0.8
                    ))
        
        return turbines
    
    def _random_layout(self) -> List[Turbine]:
        """Generate random layout."""
        np.random.seed(42)
        
        turbines = []
        min_dist = self.config.min_spacing * self.config.rotor_diameter
        
        for _ in range(self.config.n_turbines * 2):  # Try more points
            if len(turbines) >= self.config.n_turbines:
                break
            
            x = np.random.uniform(self.config.boundary_x[0], self.config.boundary_x[1])
            y = np.random.uniform(self.config.boundary_y[0], self.config.boundary_y[1])
            
            # Check minimum distance
            valid = True
            for t in turbines:
                dist = np.sqrt((x - t.x)**2 + (y - t.y)**2)
                if dist < min_dist:
                    valid = False
                    break
            
            if valid:
                turbines.append(Turbine(
                    x=x,
                    y=y,
                    hub_height=self.config.hub_height,
                    rotor_diameter=self.config.rotor_diameter,
                    rated_power=self.config.rated_power,
                    thrust_coefficient=0.8
                ))
        
        return turbines
    
    def evaluate_layout(self,
                       wind_rose: Dict[float, float],
                       stability_distribution: Optional[Dict[str, float]] = None) -> Dict:
        """Evaluate layout performance.
        
        Args:
            wind_rose: Dictionary {direction: frequency}
            stability_distribution: Dictionary {stability: frequency}
            
        Returns:
            Performance metrics
        """
        if stability_distribution is None:
            stability_distribution = {'neutral': 1.0}
        
        total_aep = 0.0
        total_wake_loss = 0.0
        
        for direction, freq in wind_rose.items():
            for stability, stab_freq in stability_distribution.items():
                # Compute effective wind speeds with wakes
                speeds = self.wake_model.compute_multiple_wakes(
                    self.turbines,
                    wind_direction=direction,
                    ambient_speed=10.0,  # Reference speed
                    stability=stability
                )
                
                # Calculate power
                for i, speed in enumerate(speeds):
                    power = self.wake_model.calculate_power(speed, self.turbines[i])
                    
                    # No-wake power for loss calculation
                    no_wake_power = self.wake_model.calculate_power(10.0, self.turbines[i])
                    
                    # Weight by frequencies
                    total_aep += power * freq * stab_freq * 8760 / 1000  # MWh
                    
                    if no_wake_power > 0:
                        wake_loss = (no_wake_power - power) / no_wake_power
                        total_wake_loss += wake_loss * freq * stab_freq
        
        self.aep = total_aep
        self.wake_losses = total_wake_loss / len(self.turbines) if self.turbines else 0
        
        return {
            'aep_mwh': total_aep,
            'wake_loss_fraction': self.wake_losses,
            'capacity_factor': total_aep / (self.config.n_turbines * self.config.rated_power * 8760 / 1000),
            'n_turbines': len(self.turbines)
        }
    
    def optimize(self,
                wind_rose: Dict[float, float],
                n_iterations: int = 100,
                step_size: float = 50.0) -> List[Turbine]:
        """Optimize turbine layout using gradient-free method.
        
        Args:
            wind_rose: Wind rose data
            n_iterations: Number of optimization iterations
            step_size: Initial step size [m]
            
        Returns:
            Optimized turbine layout
        """
        best_layout = self.turbines.copy()
        best_aep = self.evaluate_layout(wind_rose)['aep_mwh']
        
        for iteration in range(n_iterations):
            # Try moving each turbine
            new_layout = best_layout.copy()
            
            for i, turbine in enumerate(new_layout):
                # Try random move
                dx = np.random.uniform(-step_size, step_size)
                dy = np.random.uniform(-step_size, step_size)
                
                new_x = turbine.x + dx
                new_y = turbine.y + dy
                
                # Check bounds
                if (self.config.boundary_x[0] <= new_x <= self.config.boundary_x[1] and
                    self.config.boundary_y[0] <= new_y <= self.config.boundary_y[1]):
                    
                    # Check spacing with other turbines
                    valid = True
                    min_dist = self.config.min_spacing * self.config.rotor_diameter
                    
                    for j, other in enumerate(new_layout):
                        if j != i:
                            dist = np.sqrt((new_x - other.x)**2 + (new_y - other.y)**2)
                            if dist < min_dist:
                                valid = False
                                break
                    
                    if valid:
                        new_layout[i] = Turbine(
                            x=new_x,
                            y=new_y,
                            hub_height=turbine.hub_height,
                            rotor_diameter=turbine.rotor_diameter,
                            rated_power=turbine.rated_power,
                            thrust_coefficient=turbine.thrust_coefficient
                        )
            
            # Evaluate new layout
            self.turbines = new_layout
            metrics = self.evaluate_layout(wind_rose)
            
            if metrics['aep_mwh'] > best_aep:
                best_aep = metrics['aep_mwh']
                best_layout = new_layout.copy()
                step_size *= 1.05  # Increase step size
            else:
                step_size *= 0.95  # Decrease step size
            
            step_size = max(10.0, min(500.0, step_size))
        
        self.turbines = best_layout
        return best_layout
    
    def get_positions(self) -> List[Tuple[float, float]]:
        """Get turbine positions."""
        return [(t.x, t.y) for t in self.turbines]
    
    def to_dict(self) -> Dict:
        """Export layout to dictionary."""
        return {
            'n_turbines': len(self.turbines),
            'rotor_diameter': self.config.rotor_diameter,
            'hub_height': self.config.hub_height,
            'turbines': [
                {
                    'x': t.x,
                    'y': t.y,
                    'hub_height': t.hub_height,
                    'rotor_diameter': t.rotor_diameter,
                    'rated_power': t.rated_power
                }
                for t in self.turbines
            ],
            'aep_mwh': self.aep,
            'wake_losses': self.wake_losses
        }
    
    def plot_layout(self, ax=None, show_wakes: bool = False):
        """Plot turbine layout."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        
        x = [t.x for t in self.turbines]
        y = [t.y for t in self.turbines]
        
        ax.scatter(x, y, s=100, c='blue', marker='o', alpha=0.6)
        
        # Add rotor diameter circles
        for t in self.turbines:
            circle = plt.Circle(
                (t.x, t.y),
                t.rotor_diameter / 2,
                fill=False,
                color='blue',
                alpha=0.3
            )
            ax.add_patch(circle)
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'Turbine Layout ({len(self.turbines)} turbines)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return ax
