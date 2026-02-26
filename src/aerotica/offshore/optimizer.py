"""Offshore Wind Farm Optimizer."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

from aerotica.offshore.wake_model import WakeModel, Turbine
from aerotica.offshore.layout import TurbineLayout, LayoutConfig
from aerotica.offshore.resource import OffshoreResource


class OffshoreOptimizer:
    """Optimize offshore wind farm layout and operation."""
    
    def __init__(self,
                 site_latitude: float,
                 site_longitude: float,
                 water_depth: float,
                 n_turbines: int,
                 area_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                 turbine_model: str = '15MW'):
        """Initialize offshore optimizer.
        
        Args:
            site_latitude: Site latitude [degrees]
            site_longitude: Site longitude [degrees]
            water_depth: Water depth [m]
            n_turbines: Number of turbines
            area_bounds: Farm boundaries ((xmin, xmax), (ymin, ymax))
            turbine_model: Turbine model
        """
        self.site_latitude = site_latitude
        self.site_longitude = site_longitude
        self.water_depth = water_depth
        self.n_turbines = n_turbines
        self.area_bounds = area_bounds
        
        # Turbine specifications (15MW next-gen)
        self.turbine_specs = {
            '15MW': {
                'rotor_diameter': 236.0,
                'hub_height': 150.0,
                'rated_power': 15000.0,
                'thrust_coefficient': 0.8
            }
        }.get(turbine_model, {
            'rotor_diameter': 236.0,
            'hub_height': 150.0,
            'rated_power': 15000.0,
            'thrust_coefficient': 0.8
        })
        
        # Initialize components
        self.resource = OffshoreResource(
            latitude=site_latitude,
            longitude=site_longitude,
            water_depth=water_depth
        )
        
        self.wake_model = WakeModel()
        
        self.layout = None
        
        # Optimization results
        self.results = {}
    
    def setup(self, years: List[int] = [2020, 2021, 2022, 2023, 2024]):
        """Setup optimizer with resource data.
        
        Args:
            years: Years for resource assessment
        """
        print("📊 Loading wind resource data...")
        self.resource.load_era5_data(years)
        
        print("📈 Computing wind rose...")
        self.resource.compute_wind_rose()
        
        print("📐 Fitting Weibull distribution...")
        self.resource.fit_weibull()
        
        print("✅ Setup complete")
    
    def create_initial_layout(self, pattern: str = 'staggered') -> TurbineLayout:
        """Create initial turbine layout.
        
        Args:
            pattern: Layout pattern
            
        Returns:
            TurbineLayout object
        """
        config = LayoutConfig(
            n_turbines=self.n_turbines,
            min_spacing=7,  # 7 rotor diameters
            max_spacing=15,  # 15 rotor diameters
            boundary_x=(self.area_bounds[0][0], self.area_bounds[0][1]),
            boundary_y=(self.area_bounds[1][0], self.area_bounds[1][1]),
            rotor_diameter=self.turbine_specs['rotor_diameter'],
            hub_height=self.turbine_specs['hub_height'],
            rated_power=self.turbine_specs['rated_power']
        )
        
        self.layout = TurbineLayout(config, self.wake_model)
        self.layout.generate_initial_layout(pattern)
        
        print(f"✅ Created {pattern} layout with {len(self.layout.turbines)} turbines")
        
        return self.layout
    
    def evaluate_current_layout(self) -> Dict:
        """Evaluate current layout performance.
        
        Returns:
            Performance metrics
        """
        if self.layout is None:
            raise ValueError("No layout created. Call create_initial_layout() first.")
        
        # Use wind rose from resource assessment
        wind_rose = self.resource.wind_rose
        
        # Simple stability distribution
        stability_dist = {'neutral': 0.5, 'stable': 0.3, 'unstable': 0.2}
        
        metrics = self.layout.evaluate_layout(wind_rose, stability_dist)
        
        # Add resource assessment
        site_assessment = self.resource.assess_site()
        metrics['site'] = site_assessment
        
        return metrics
    
    def optimize_layout(self,
                       n_iterations: int = 200,
                       verbose: bool = True) -> Dict:
        """Optimize turbine layout.
        
        Args:
            n_iterations: Number of optimization iterations
            verbose: Print progress
            
        Returns:
            Optimization results
        """
        if self.layout is None:
            raise ValueError("No layout created. Call create_initial_layout() first.")
        
        print("\n🚀 Starting layout optimization...")
        print(f"   Iterations: {n_iterations}")
        print(f"   Turbines: {self.n_turbines}")
        
        # Initial evaluation
        initial_metrics = self.evaluate_current_layout()
        initial_aep = initial_metrics['aep_mwh']
        
        print(f"\n📊 Initial AEP: {initial_aep:.0f} MWh")
        print(f"   Wake losses: {initial_metrics['wake_loss_fraction']:.2%}")
        
        # Run optimization
        wind_rose = self.resource.wind_rose
        optimized_layout = self.layout.optimize(wind_rose, n_iterations)
        
        # Final evaluation
        final_metrics = self.evaluate_current_layout()
        final_aep = final_metrics['aep_mwh']
        
        improvement = (final_aep - initial_aep) / initial_aep * 100
        
        print(f"\n✅ Optimization complete!")
        print(f"   Final AEP: {final_aep:.0f} MWh")
        print(f"   Improvement: {improvement:.1f}%")
        print(f"   Final wake losses: {final_metrics['wake_loss_fraction']:.2%}")
        
        self.results = {
            'initial': initial_metrics,
            'final': final_metrics,
            'improvement_percent': improvement,
            'n_iterations': n_iterations
        }
        
        return self.results
    
    def compute_wake_map(self, resolution: float = 100.0) -> Dict:
        """Compute wake deficit map.
        
        Args:
            resolution: Grid resolution [m]
            
        Returns:
            Wake map data
        """
        if self.layout is None:
            raise ValueError("No layout created")
        
        domain_size = (
            self.area_bounds[0][1] - self.area_bounds[0][0],
            self.area_bounds[1][1] - self.area_bounds[1][0]
        )
        
        # Use predominant wind direction (270° for westerlies)
        wind_direction = 270.0
        
        wake_map = self.wake_model.compute_wake_map(
            self.layout.turbines,
            domain_size,
            resolution,
            wind_direction,
            ambient_speed=10.0
        )
        
        return wake_map
    
    def calculate_financials(self,
                            capex_per_mw: float = 3.2e6,  # €3.2M per MW
                            opex_percent: float = 2.0,    # 2% of CAPEX
                            electricity_price: float = 80, # €80 per MWh
                            discount_rate: float = 0.06,   # 6%
                            lifetime_years: int = 25) -> Dict:
        """Calculate financial metrics.
        
        Args:
            capex_per_mw: Capital expenditure per MW [€]
            opex_percent: Operating expenditure [% of CAPEX]
            electricity_price: Electricity price [€/MWh]
            discount_rate: Discount rate
            lifetime_years: Project lifetime [years]
            
        Returns:
            Financial metrics
        """
        if not self.results:
            metrics = self.evaluate_current_layout()
            aep = metrics['aep_mwh']
        else:
            aep = self.results['final']['aep_mwh']
        
        total_capacity = self.n_turbines * self.turbine_specs['rated_power'] / 1000  # MW
        capex = total_capacity * capex_per_mw * 1e6  # €
        opex_annual = capex * opex_percent / 100  # € per year
        
        # Annual revenue
        revenue_annual = aep * electricity_price  # €
        
        # Simple NPV calculation
        npv = -capex
        for year in range(1, lifetime_years + 1):
            cash_flow = revenue_annual - opex_annual
            npv += cash_flow / (1 + discount_rate) ** year
        
        # IRR (simplified)
        if capex > 0:
            irr = (revenue_annual - opex_annual) / capex
        else:
            irr = 0
        
        # LCOE
        lcoe = (capex + sum(opex_annual / (1 + discount_rate) ** y 
                           for y in range(1, lifetime_years + 1))) / \
               sum(aep / (1 + discount_rate) ** y for y in range(1, lifetime_years + 1))
        
        return {
            'total_capacity_mw': total_capacity,
            'capex_euro': capex,
            'opex_annual_euro': opex_annual,
            'revenue_annual_euro': revenue_annual,
            'npv_euro': npv,
            'irr_percent': irr * 100,
            'lcoe_euro_per_mwh': lcoe,
            'payback_years': capex / (revenue_annual - opex_annual) if revenue_annual > opex_annual else 999
        }
    
    def generate_report(self, output_dir: Path):
        """Generate comprehensive optimization report.
        
        Args:
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get final metrics
        metrics = self.evaluate_current_layout()
        
        # Financial analysis
        financials = self.calculate_financials()
        
        # Compile report
        report = {
            'site': {
                'latitude': self.site_latitude,
                'longitude': self.site_longitude,
                'water_depth_m': self.water_depth,
                'area_km2': (self.area_bounds[0][1] - self.area_bounds[0][0]) * \
                            (self.area_bounds[1][1] - self.area_bounds[1][0]) / 1e6
            },
            'turbines': {
                'model': '15MW Next-Gen',
                'n_turbines': self.n_turbines,
                'rotor_diameter_m': self.turbine_specs['rotor_diameter'],
                'hub_height_m': self.turbine_specs['hub_height'],
                'total_capacity_mw': self.n_turbines * self.turbine_specs['rated_power'] / 1000
            },
            'resource': metrics['site'],
            'performance': {
                'aep_mwh': metrics['aep_mwh'],
                'aep_gwh': metrics['aep_mwh'] / 1000,
                'capacity_factor': metrics['capacity_factor'],
                'wake_losses_percent': metrics['wake_loss_fraction'] * 100
            },
            'financial': financials,
            'optimization': self.results if self.results else {}
        }
        
        # Save report
        with open(output_dir / 'optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate plots
        try:
            import matplotlib.pyplot as plt
            
            # Layout plot
            fig, ax = plt.subplots(figsize=(12, 8))
            self.layout.plot_layout(ax)
            plt.savefig(output_dir / 'layout.png', dpi=150, bbox_inches='tight')
            
            # Wake map
            wake_map = self.compute_wake_map()
            if wake_map:
                fig, ax = plt.subplots(figsize=(12, 8))
                X, Y = np.meshgrid(wake_map['x'], wake_map['y'])
                contour = ax.contourf(X, Y, wake_map['wake_map'], levels=20, cmap='RdYlBu_r')
                plt.colorbar(contour, ax=ax, label='Wake deficit factor')
                
                # Plot turbines
                for t in self.layout.turbines:
                    ax.plot(t.x, t.y, 'k^', markersize=8)
                
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                ax.set_title('Wake Deficit Map')
                ax.set_aspect('equal')
                plt.savefig(output_dir / 'wake_map.png', dpi=150, bbox_inches='tight')
            
            plt.close('all')
            
        except Exception as e:
            print(f"⚠️  Could not generate plots: {e}")
        
        print(f"📊 Report generated in {output_dir}")
        print(f"   AEP: {report['performance']['aep_gwh']:.1f} GWh/year")
        print(f"   Capacity factor: {report['performance']['capacity_factor']:.2%}")
        print(f"   NPV: €{report['financial']['npv_euro']:.0f}")
        print(f"   IRR: {report['financial']['irr_percent']:.1f}%")
        
        return report
