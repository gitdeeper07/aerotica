# AEROTICA Shell Completion

AEROTICA provides shell completion for bash, zsh, and fish shells to enhance command-line productivity.

## Installation

### Bash

Add to your `~/.bashrc`:
```bash
eval "$(_AEROTICA_COMPLETE=bash_source aerotica)"
```

Zsh

Add to your ~/.zshrc:

```zsh
eval "$(_AEROTICA_COMPLETE=zsh_source aerotica)"
```

Fish

Add to your ~/.config/fish/config.fish:

```fish
eval (env _AEROTICA_COMPLETE=fish_source aerotica)
```

Available Commands

```bash
aerotica --help
```

Core Commands

Command Description
aerotica compute-ake Compute AKE composite index for a site
aerotica gust-alert Run gust pre-alerting engine
aerotica wind-field Generate 3D wind field using PINN
aerotica urban-assessment Assess building-integrated wind potential
aerotica offshore-optimize Optimize offshore wind farm layout
aerotica dashboard Launch monitoring dashboard
aerotica validate Validate against benchmark data
aerotica export Export results to various formats

Parameter-Specific Commands

```bash
aerotica compute ked --site SITE_ID     # Compute Kinetic Energy Density
aerotica compute tii --site SITE_ID     # Compute Turbulence Intensity Index
aerotica compute vsr --site SITE_ID     # Compute Vertical Shear Ratio
aerotica compute aod --site SITE_ID     # Get Aerosol Optical Depth
aerotica compute thd --site SITE_ID     # Compute Thermal Helicity Dynamics
aerotica compute pgf --site SITE_ID     # Compute Pressure Gradient Force
aerotica compute hci --site SITE_ID     # Compute Humidity-Convection Interaction
aerotica compute asi --site SITE_ID     # Compute Atmospheric Stability Integration
aerotica compute lrc --site SITE_ID     # Compute Local Roughness Coefficient
```

Options

```bash
--site SITE           # Site name or ID
--lat LAT             # Latitude (decimal degrees)
--lon LON             # Longitude (decimal degrees)
--height H            # Height above ground (m)
--start-date DATE     # Start date (YYYY-MM-DD)
--end-date DATE       # End date (YYYY-MM-DD)
--format FORMAT       # Output format (json, csv, table, netcdf)
--output FILE         # Output file path
--config FILE         # Configuration file
--verbose             # Verbose output
--debug               # Debug mode
```

Examples

```bash
# Compute AKE for a site
aerotica compute-ake --site casablanca_port --height 60

# Run gust pre-alerting with real-time data
aerotica gust-alert --config config/casablanca.yaml --interval 30

# Generate 3D wind field
aerotica wind-field --lat 33.5 --lon -7.5 --domain 50 --output wind_field.nc

# Assess building-integrated wind potential
aerotica urban-assessment --lidar data/lidar/casablanca.tif --output viable_sites.geojson

# Export results as JSON
aerotica export --site edinburgh --format json --output edinburgh_data.json

# Check active gust alerts
aerotica gust-alert --status active

# Run validation
aerotica validate --benchmark les --dataset validation_set
```

Tab Completion Features

The completion system provides:

· Command name completion
· Site name completion (from config/sites/)
· Date completion (YYYY-MM-DD format)
· File path completion for --output
· Format completion (json, csv, table, netcdf)

Troubleshooting

If completion isn't working:

1. Ensure aerotica is installed: which aerotica
2. Reload shell: exec $SHELL
3. Check installation: aerotica --version
4. Reinstall completions: re-run the eval command

For more help: aerotica help completion
