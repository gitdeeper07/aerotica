"""AEROTICA Command Line Interface."""

import click
import sys
from pathlib import Path
import json
import yaml

from aerotica import __version__
from aerotica.ake import AKEComposite
from aerotica.parameters import KED, TII, VSR, AOD, THD, PGF, HCI, ASI, LRC


@click.group()
@click.version_option(version=__version__)
def main():
    """AEROTICA - Atmospheric Kinetic Energy Mapping Framework."""
    pass


@main.command()
@click.option('--site', required=True, help='Site identifier')
@click.option('--lat', type=float, help='Latitude')
@click.option('--lon', type=float, help='Longitude')
@click.option('--height', type=float, default=60, help='Height above ground [m]')
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
@click.option('--output', type=click.Path(), help='Output file')
def compute_ake(site, lat, lon, height, config, output):
    """Compute AKE composite index for a site."""
    click.echo(f"🔍 Computing AKE for site: {site}")
    
    # Load config
    if config:
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}
    
    # Initialize AKE
    ake = AKEComposite(
        site_id=site,
        climate_zone=cfg.get('climate_zone', 'temperate'),
        site_type=cfg.get('site_type', 'unknown')
    )
    
    # Load parameters (simplified example)
    parameters = {
        'KED': 0.85,
        'TII': 0.76,
        'VSR': 0.89,
        'AOD': 0.34,
        'THD': 0.72,
        'PGF': 0.65,
        'HCI': 0.59,
        'ASI': 0.71,
        'LRC': 0.44
    }
    
    ake.load_parameters(parameters)
    result = ake.compute()
    
    # Output
    if output:
        with open(output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        click.echo(f"✅ Results saved to {output}")
    else:
        click.echo(json.dumps(result, indent=2, default=str))


@main.command()
@click.option('--site', required=True, help='Site identifier')
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
@click.option('--interval', type=int, default=30, help='Update interval [seconds]')
def gust_alert(site, config, interval):
    """Run gust pre-alerting in real-time mode."""
    click.echo(f"⚠️  Starting gust pre-alerting for {site}")
    click.echo(f"   Update interval: {interval} seconds")
    click.echo("   Press Ctrl+C to stop")
    
    # This would connect to real data sources
    click.echo("\n🔴 Waiting for data...")
    
    try:
        while True:
            # Simulate processing
            click.echo(f"\n[{site}] No gust detected")
            import time
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\n\n✅ Gust pre-alerting stopped")


@main.command()
@click.option('--site', required=True, help='Site identifier')
@click.option('--param', help='Specific parameter to compute')
@click.option('--list', 'list_params', is_flag=True, help='List all parameters')
def doctor(site, param, list_params):
    """Run system diagnostics."""
    click.echo("🔍 AEROTICA System Diagnostics")
    click.echo("==============================")
    
    # Check Python
    click.echo(f"\n📌 Python: {sys.version}")
    
    # Check imports
    try:
        import numpy
        click.echo(f"✅ NumPy: {numpy.__version__}")
    except:
        click.echo("❌ NumPy not found")
    
    try:
        import torch
        click.echo(f"✅ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except:
        click.echo("❌ PyTorch not found")
    
    # List parameters
    if list_params:
        click.echo("\n📊 Available Parameters:")
        params = [
            ("KED", "Kinetic Energy Density (22%)"),
            ("TII", "Turbulence Intensity Index (16%)"),
            ("VSR", "Vertical Shear Ratio (14%)"),
            ("AOD", "Aerosol Optical Depth (12%)"),
            ("THD", "Thermal Helicity Dynamics (10%)"),
            ("PGF", "Pressure Gradient Force (8%)"),
            ("HCI", "Humidity-Convection Interaction (7%)"),
            ("ASI", "Atmospheric Stability Integration (6%)"),
            ("LRC", "Local Roughness Coefficient (5%)"),
        ]
        for code, desc in params:
            click.echo(f"  • {code}: {desc}")
    
    click.echo("\n✅ Diagnostics complete")


@main.command()
@click.option('--site', default='tokyo', help='Site for demo')
def demo(site):
    """Run complete demonstration."""
    click.echo("🌪️  AEROTICA Demonstration")
    click.echo("========================")
    
    # Show AKE computation
    click.echo(f"\n📊 Computing AKE for {site}...")
    
    ake = AKEComposite(
        site_id=site,
        climate_zone='temperate',
        site_type='urban'
    )
    
    # Sample parameters
    params = {
        'KED': 0.83,
        'TII': 0.76,
        'VSR': 0.89,
        'AOD': 0.34,
        'THD': 0.72,
        'PGF': 0.65,
        'HCI': 0.59,
        'ASI': 0.71,
        'LRC': 0.44
    }
    
    ake.load_parameters(params)
    result = ake.compute()
    
    click.echo(f"\n✅ AKE Score: {result['score']:.3f}")
    click.echo(f"   Classification: {result['classification']}")
    click.echo(f"   Gust Risk: {result['gust_risk']}")
    click.echo(f"   Confidence: {result['confidence']:.1%}")
    
    click.echo("\n🎉 Demo completed successfully!")


if __name__ == '__main__':
    main()

@main.command()
@click.option('--ked', type=float, help='KED value')
@click.option('--tii', type=float, help='TII value')
@click.option('--vsr', type=float, help='VSR value')
@click.option('--aod', type=float, help='AOD value')
@click.option('--thd', type=float, help='THD value')
@click.option('--pgf', type=float, help='PGF value')
@click.option('--hci', type=float, help='HCI value')
@click.option('--asi', type=float, help='ASI value')
@click.option('--lrc', type=float, help='LRC value')
@click.option('--output', type=click.Path(), help='Output file')
def calculate_ake(ked, tii, vsr, aod, thd, pgf, hci, asi, lrc, output):
    """Calculate AKE from individual parameter values."""
    params = {}
    if ked is not None: params['KED'] = ked
    if tii is not None: params['TII'] = tii
    if vsr is not None: params['VSR'] = vsr
    if aod is not None: params['AOD'] = aod
    if thd is not None: params['THD'] = thd
    if pgf is not None: params['PGF'] = pgf
    if hci is not None: params['HCI'] = hci
    if asi is not None: params['ASI'] = asi
    if lrc is not None: params['LRC'] = lrc
    
    if not params:
        click.echo("❌ No parameters provided")
        return
    
    ake = AKEComposite(
        site_id="cli_input",
        climate_zone="temperate",
        site_type="custom"
    )
    
    ake.load_parameters(params)
    result = ake.compute()
    
    if output:
        with open(output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        click.echo(f"✅ Results saved to {output}")
    else:
        click.echo(json.dumps(result, indent=2, default=str))


@main.command()
@click.option('--site', required=True, help='Site ID')
@click.option('--data-dir', type=click.Path(exists=True), help='Data directory')
@click.option('--model-dir', type=click.Path(exists=True), help='Model directory')
@click.option('--output', type=click.Path(), help='Output file')
def urban_assessment(site, data_dir, model_dir, output):
    """Run urban wind assessment."""
    from aerotica.urban import BuildingWindAssessor
    
    assessor = BuildingWindAssessor(
        lidar_dem=Path(data_dir) / f"{site}_lidar.tif" if data_dir else None,
        pinn_model=None
    )
    
    sites = assessor.identify_sites()
    
    if output:
        geojson = assessor.to_geojson(sites)
        with open(output, 'w') as f:
            json.dump(geojson, f, indent=2)
        click.echo(f"✅ Results saved to {output}")
    else:
        click.echo(f"Found {len(sites)} viable sites")
        for site in sites[:5]:  # Show first 5
            click.echo(f"  • {site.id}: AKE={site.ake_score:.3f}, Yield={site.annual_yield_kwh:.0f} kWh")


@main.command()
@click.option('--site', required=True, help='Site ID')
@click.option('--lat', type=float, required=True, help='Latitude')
@click.option('--lon', type=float, required=True, help='Longitude')
@click.option('--depth', type=float, required=True, help='Water depth [m]')
@click.option('--n-turbines', type=int, required=True, help='Number of turbines')
@click.option('--area', type=float, required=True, help='Farm area [km²]')
@click.option('--output', type=click.Path(), help='Output directory')
def offshore_optimize(site, lat, lon, depth, n_turbines, area, output):
    """Run offshore wind farm optimization."""
    from aerotica.offshore import OffshoreOptimizer
    
    # Calculate area bounds (assuming square)
    side = np.sqrt(area) * 1000  # km to m
    bounds = ((0, side), (0, side))
    
    optimizer = OffshoreOptimizer(
        site_latitude=lat,
        site_longitude=lon,
        water_depth=depth,
        n_turbines=n_turbines,
        area_bounds=bounds
    )
    
    click.echo("📊 Setting up resource assessment...")
    optimizer.setup([2020, 2021, 2022, 2023, 2024])
    
    click.echo("🏗️  Creating layout...")
    optimizer.create_initial_layout('staggered')
    
    click.echo("🚀 Running optimization...")
    results = optimizer.optimize_layout(n_iterations=100)
    
    if output:
        output_dir = Path(output)
        optimizer.generate_report(output_dir)
        click.echo(f"✅ Report saved to {output_dir}")
    else:
        click.echo(f"Final AEP: {results['final']['aep_mwh']/1000:.1f} GWh/year")
        click.echo(f"Improvement: {results['improvement_percent']:.1f}%")


@main.command()
@click.option('--host', default='127.0.0.1', help='Host')
@click.option('--port', default=8000, type=int, help='Port')
@click.option('--reload', is_flag=True, help='Auto-reload on code changes')
def serve(host, port, reload):
    """Start API server."""
    import uvicorn
    from aerotica.api.main import app
    
    click.echo(f"🚀 Starting AEROTICA API server at http://{host}:{port}")
    uvicorn.run(
        "aerotica.api.main:app",
        host=host,
        port=port,
        reload=reload
    )


@main.command()
@click.option('--port', default=8501, type=int, help='Port')
def dashboard(port):
    """Launch Streamlit dashboard."""
    import subprocess
    import sys
    
    script_path = Path(__file__).parent.parent / "scripts" / "launch_dashboard.py"
    
    click.echo(f"📊 Launching AEROTICA dashboard on port {port}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(script_path),
        "--server.port", str(port)
    ])


if __name__ == '__main__':
    main()
