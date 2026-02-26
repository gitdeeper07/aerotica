# Contributing to AEROTICA

First off, thank you for considering contributing to AEROTICA! We welcome contributions from atmospheric scientists, fluid dynamicists, machine learning engineers, renewable energy experts, and anyone passionate about advancing atmospheric kinetic energy research.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Adding New Parameters](#adding-new-parameters)
- [PINN Development](#pinn-development)
- [Reporting Issues](#reporting-issues)
- [Contact](#contact)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to gitdeeper@gmail.com.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- PyTorch 2.3+
- CUDA 11.8+ (for PINN training)
- GDAL 3.6+ (for LiDAR processing)
- Basic knowledge of fluid dynamics and atmospheric physics

### Setup Development Environment

```bash
# Fork the repository on GitLab, then clone your fork
git clone https://gitlab.com/YOUR_USERNAME/aerotica.git
cd aerotica

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
pre-commit install
```

Verify Setup

```bash
pytest tests/unit/ -v
ruff check aerotica/
mypy aerotica/
```

Development Workflow

1. Create an issue describing your proposed changes
2. Discuss with maintainers to ensure alignment
3. Fork and branch:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```
4. Make changes following our coding standards
5. Write/update tests for your changes
6. Run tests locally and ensure they pass
7. Commit with clear messages
8. Push to your fork
9. Open a Merge Request

Branch Naming

· feature/ - New features
· fix/ - Bug fixes
· docs/ - Documentation updates
· refactor/ - Code refactoring
· test/ - Test improvements
· perf/ - Performance optimizations
· parameter/ - New AKE parameter additions

Pull Request Process

1. Update documentation for any changed functionality
2. Add tests for new features (coverage should not decrease)
3. Update CHANGELOG.md with your changes under "Unreleased"
4. Ensure CI passes (tests, linting, type checking)
5. Request review from maintainers
6. Address review feedback
7. Merge after approval and CI passes

PR Template

```markdown
## Description
Brief description of changes

## Related Issue
Fixes #(issue number)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] New AKE parameter
- [ ] PINN architecture improvement
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Test update

## How Has This Been Tested?
Describe tests you added/ran

## Validation Results
- Impact on AKE accuracy: ±X%
- Computational performance change: Y%
- LES benchmark agreement: Z%

## Checklist
- [ ] Tests pass locally
- [ ] Docs updated
- [ ] CHANGELOG updated
- [ ] Code follows style guidelines
- [ ] Type hints added/updated
- [ ] PINN physics constraints verified
```

Coding Standards

Python

· Format: Black (line length 88)
· Imports: isort with black profile
· Linting: ruff (see pyproject.toml for rules)
· Type Hints: Required for all public functions
· Docstrings: Google style with physics equations in LaTeX

Example

```python
"""Kinetic Energy Density parameter module.

This module implements the KED (Kinetic Energy Density) parameter,
which captures the available power flux in lower atmospheric layers.

.. math::
    KED = \\frac{1}{2}\\rho v^3 \\quad [W/m^2]

where:
    - :math:`\\rho` is air density [kg/m³]
    - :math:`v` is wind speed [m/s]
"""

from typing import Optional, Union

import numpy as np
import torch


def compute_ked(
    wind_speed: Union[np.ndarray, torch.Tensor],
    air_density: Optional[float] = None,
    temperature: Optional[float] = None,
    pressure: Optional[float] = None,
    humidity: Optional[float] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """Compute Kinetic Energy Density.
    
    Args:
        wind_speed: Wind speed time series [m/s]
        air_density: Air density [kg/m³] (if None, computed from T, P, q)
        temperature: Air temperature [K] (required if air_density is None)
        pressure: Atmospheric pressure [Pa] (required if air_density is None)
        humidity: Specific humidity [kg/kg] (optional, for density correction)
    
    Returns:
        KED values [W/m²]
        
    Raises:
        ValueError: If inputs are invalid or missing required parameters
        
    Examples:
        >>> ked = compute_ked(wind_speed=12.5, air_density=1.225)
        >>> print(f"{ked:.1f} W/m²")
        1148.4 W/m²
    """
    if air_density is None:
        if temperature is None or pressure is None:
            raise ValueError(
                "Must provide either air_density or both temperature and pressure"
            )
        # Compute air density using ideal gas law with humidity correction
        R_d = 287.05  # J/(kg·K) - specific gas constant for dry air
        if humidity is not None:
            T_v = temperature * (1 + 0.608 * humidity)  # Virtual temperature
        else:
            T_v = temperature
        air_density = pressure / (R_d * T_v)
    
    return 0.5 * air_density * wind_speed**3
```

Testing Guidelines

Test Structure

```
tests/
├── conftest.py              # Fixtures
├── unit/                    # Unit tests
│   ├── parameters/          # Test each of the 9 parameters
│   │   ├── test_ked.py
│   │   ├── test_tii.py
│   │   └── ...
│   ├── pinn/                 # Test PINN components
│   └── alerts/               # Test gust pre-alerting
├── integration/             # Integration tests
│   ├── test_ake_pipeline.py
│   └── test_end_to_end.py
└── fixtures/                # Test data
    ├── sample_wind_data.nc
    ├── sample_lidar.tif
    └── sample_radar.nc
```

Writing Tests

```python
import pytest
import numpy as np
from aerotica.parameters.ked import compute_ked

def test_ked_basic():
    """Test KED calculation with basic inputs."""
    wind_speed = np.array([10.0, 12.0, 8.0])
    air_density = 1.225
    
    ked = compute_ked(wind_speed, air_density=air_density)
    
    expected = 0.5 * 1.225 * wind_speed**3
    np.testing.assert_array_almost_equal(ked, expected)

@pytest.mark.parametrize("wind_speed,air_density,expected", [
    (10.0, 1.225, 612.5),
    (12.0, 1.225, 1058.4),
    (15.0, 1.0, 1687.5),
])
def test_ked_scalar(wind_speed, air_density, expected):
    """Test KED with scalar inputs."""
    result = compute_ked(wind_speed, air_density=air_density)
    assert abs(result - expected) < 0.1

def test_ked_density_computation():
    """Test KED with automatic density computation."""
    wind_speed = 10.0
    temperature = 288.15  # 15°C
    pressure = 101325.0  # 1 atm
    
    ked = compute_ked(wind_speed, temperature=temperature, pressure=pressure)
    
    # Expected density: p/(R_d*T) = 101325/(287.05*288.15) ≈ 1.225
    expected_ked = 0.5 * 1.225 * 1000
    assert abs(ked - expected_ked) < 10.0
```

Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov=aerotica --cov-report=html

# Specific test
pytest tests/unit/parameters/test_ked.py::test_ked_basic -v
```

Documentation

Building Documentation

```bash
cd docs
make html  # or make latexpdf for PDF
```

Documentation Standards

· README.md: Project overview, quick start
· docs/: Detailed documentation with physics derivations
· docstrings: In-code documentation with LaTeX equations
· notebooks: Example notebooks with real-world cases

Adding New Parameters

If adding a new parameter to the AKE index:

1. Create module in aerotica/parameters/new_param.py
2. Implement the physical derivation following established patterns
3. Add to aerotica/parameters/__init__.py
4. Update AKE composite in aerotica/ake/composite.py
5. Add documentation in docs/parameters/new_param.md
6. Add tests in tests/unit/parameters/test_new_param.py
7. Validate against LES benchmarks
8. Update weights via Bayesian optimization

Parameter Template

```python
"""New Parameter Module.

.. math::
    \\text{NewParam} = f(x, y, z)

where:
    - x: description
    - y: description
"""

from typing import Optional
import numpy as np

class NewParameter:
    """New parameter implementation."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
    
    def compute(self, data: dict) -> float:
        """Compute parameter value.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Normalized parameter value in [0, 1]
        """
        # Implementation
        pass
```

PINN Development

For contributions to the Physics-Informed Neural Network:

1. Understand the Navier-Stokes constraints
2. Modify aerotica/pinn/ modules
3. Test physics residuals
4. Validate against LES benchmarks
5. Document architecture changes

Reporting Issues

Bug Reports

Include:

· Clear title and description
· Steps to reproduce
· Expected vs actual behavior
· Environment details (OS, Python version, package versions)
· Logs or screenshots
· Sample data if possible

Feature Requests

Include:

· Use case description
· Expected behavior
· Potential implementation approach
· References to similar features in literature
· Impact on AKE accuracy

Contact

· Issues: GitLab Issues
· Discussions: GitLab Discussions
· Email: gitdeeper@gmail.com
· Scientific questions: Include "AEROTICA" in subject line

---

Thank you for contributing to AEROTICA! 🌪️
