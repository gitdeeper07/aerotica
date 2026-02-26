# AEROTICA Installation Guide

This guide covers installation of the AEROTICA atmospheric kinetic energy mapping framework.

## Table of Contents
- [System Requirements](#system-requirements)
- [Quick Installation](#quick-installation)
- [Detailed Installation](#detailed-installation)
  - [1. Python Environment](#1-python-environment)
  - [2. Install AEROTICA](#2-install-aerotica)
  - [3. PINN Model Weights](#3-pinn-model-weights)
  - [4. Database Setup](#4-database-setup)
  - [5. Configuration](#5-configuration)
  - [6. Verify Installation](#6-verify-installation)
- [Platform-Specific Instructions](#platform-specific-instructions)
  - [Linux / Ubuntu](#linux--ubuntu)
  - [macOS](#macos)
  - [Windows](#windows)
  - [Termux (Android)](#termux-android)
- [Docker Installation](#docker-installation)
- [Development Installation](#development-installation)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **Python**: 3.11 or higher
- **RAM**: 8 GB (16 GB recommended for PINN inference)
- **Storage**: 10 GB free space (including model weights)
- **OS**: Linux, macOS, Windows, or Termux (Android)

### Recommended Requirements
- **GPU**: CUDA-capable with 8+ GB VRAM (for PINN training)
- **Storage**: 80 GB for full validation dataset
- **Database**: PostgreSQL 13+ with PostGIS (for production)
- **Internet**: For downloading reanalysis data and model weights

### Optional Dependencies
- **CUDA 11.8+**: For GPU acceleration
- **GDAL 3.6+**: For LiDAR and geospatial processing
- **NetCDF4**: For atmospheric data formats

## Quick Installation

```bash
# Install from PyPI
pip install aerotica-ake

# Verify installation
aerotica --version
aerotica doctor  # Check system compatibility
```

Detailed Installation

1. Python Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

2. Install AEROTICA

```bash
# Basic installation (CPU only)
pip install aerotica-ake

# With GPU support (CUDA)
pip install "aerotica-ake[cuda]"

# With all optional dependencies
pip install "aerotica-ake[all]"

# Or specific extras
pip install "aerotica-ake[ml]"      # For PINN training/inference
pip install "aerotica-ake[viz]"      # For visualization tools
pip install "aerotica-ake[web]"      # For web dashboard
pip install "aerotica-ake[gpu]"      # GPU acceleration
pip install "aerotica-ake[docs]"     # For documentation
pip install "aerotica-ake[dev]"      # For development
```

3. PINN Model Weights

Download pre-trained Physics-Informed Neural Network weights:

```bash
# Using DVC (if installed)
dvc pull models/pinn_v1/

# Or direct download
wget https://zenodo.org/record/aerotica2025/files/pinn_v1.tar.gz
tar -xzf pinn_v1.tar.gz -C models/

# Verify weights
ls -la models/pinn_v1/
# Should see: velocity_net.pt, pressure_net.pt, temperature_net.pt
```

4. Database Setup (Optional)

For production deployments:

```bash
# Install PostgreSQL with PostGIS (Ubuntu)
sudo apt install postgresql postgresql-contrib postgis

# Create database
sudo -u postgres createdb aerotica
sudo -u postgres createuser --interactive
# Create aerotica user with password

# Enable PostGIS
sudo -u postgres psql -d aerotica -c "CREATE EXTENSION postgis;"

# Initialize schema
psql -U aerotica -d aerotica -f schema.sql
```

5. Configuration

```bash
# Create configuration directory
mkdir -p ~/.aerotica
mkdir -p ~/.aerotica/data
mkdir -p ~/.aerotica/logs

# Copy default configuration
cp config/aerotica.default.yaml ~/.aerotica/config.yaml

# Edit configuration
nano ~/.aerotica/config.yaml
# Set database credentials, API keys, model paths, etc.

# Set environment variable
export AEROTICA_CONFIG=~/.aerotica/config.yaml
# Add to .bashrc or .zshrc for persistence
```

6. Verify Installation

```bash
# Run diagnostics
aerotica doctor

# Expected output:
# ✓ Python 3.11+ detected
# ✓ PyTorch 2.3+ found
# ✓ Dependencies installed
# ✓ PINN model weights found
# ✓ Configuration file found

# Run tests
pytest --pyargs aerotica -v

# Test with sample data
aerotica demo --site tokyo

# Compute AKE for a sample site
aerotica compute-ake --lat 35.7 --lon 139.7 --height 60
```

Platform-Specific Instructions

Linux / Ubuntu

```bash
# Install system dependencies
sudo apt update
sudo apt install -y \
    python3.11 python3.11-dev python3.11-venv \
    build-essential libssl-dev libffi-dev \
    libgdal-dev gdal-bin \
    libnetcdf-dev libhdf5-dev

# Install AEROTICA
pip install aerotica-ake
```

macOS

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python@3.11 gdal netcdf hdf5

# Install AEROTICA
pip install aerotica-ake
```

Windows

Using WSL2 (Recommended)

```bash
# In PowerShell as Administrator
wsl --install -d Ubuntu

# Then follow Linux instructions inside WSL
```

Native Windows

```bash
# Download Python 3.11 from python.org
# Open PowerShell as Administrator

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install AEROTICA
pip install aerotica-ake
```

Termux (Android)

```bash
# Update packages
pkg update && pkg upgrade

# Install Python and dependencies
pkg install python python-pip python-numpy python-scipy
pkg install libgdal netcdf-bin hdf5

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install AEROTICA (CPU-only version for mobile)
pip install aerotica-ake

# Note: PINN inference may be slower on mobile devices
# For real-time monitoring, use API mode
```

Docker Installation

Using pre-built image

```bash
# Pull image
docker pull gitlab.com/gitdeeper07/aerotica:latest

# Run container
docker run -it \
  --name aerotica \
  -v ~/.aerotica:/root/.aerotica \
  -e AEROTICA_CONFIG=/root/.aerotica/config.yaml \
  -p 8000:8000 \
  gitlab.com/gitdeeper07/aerotica:latest
```

Docker Compose (full stack)

```bash
# Clone repository
git clone https://gitlab.com/gitdeeper07/aerotica.git
cd aerotica

# Start services
docker-compose up -d

# Services:
# - API:8000
# - Dashboard:8501
# - PostgreSQL:5432 (with PostGIS)
# - Redis:6379

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Development Installation

For contributors and developers:

```bash
# Clone repository
git clone https://gitlab.com/gitdeeper07/aerotica.git
cd aerotica

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all extras
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Download test data
dvc pull data/test/

# Run tests
pytest tests/ -v --cov=aerotica
```

Troubleshooting

Common Issues

Package not found

```bash
# Ensure pip is up to date
pip install --upgrade pip

# Try installing with --no-cache-dir
pip install --no-cache-dir aerotica-ake
```

Import errors

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Ensure virtual environment is activated
which python
```

PINN model not loading

```bash
# Check model path
ls -la models/pinn_v1/

# Download weights if missing
aerotica download-weights --model pinn_v1
```

Memory issues

```bash
# Reduce batch size for inference
export AEROTICA_BATCH_SIZE=16

# Use CPU-only mode if GPU memory insufficient
export CUDA_VISIBLE_DEVICES=""
```

GDAL/NetCDF errors

```bash
# On Ubuntu
sudo apt install libgdal-dev libnetcdf-dev

# On Termux
pkg install libgdal netcdf-bin
```

Getting Help

· Documentation: https://aerotica-science.netlify.app
· Issues: https://gitlab.com/gitdeeper07/aerotica/-/issues
· Discussions: https://gitlab.com/gitdeeper07/aerotica/-/discussions
· Email: gitdeeper@gmail.com

Verification Script

```python
# verify.py
import aerotica
print(f"AEROTICA version: {aerotica.__version__}")

from aerotica.parameters import KED, TII, VSR
print("✓ Parameter modules imported")

from aerotica.pinn import AeroticaPINN
print("✓ PINN module imported")

from aerotica.ake import AKEComposite
print("✓ AKE composite imported")

print("Installation successful!")
```

Run it:

```bash
python verify.py
```

