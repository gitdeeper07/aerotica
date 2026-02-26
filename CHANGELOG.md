
# Changelog

All notable changes to the AEROTICA project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Baseline documentation (BASELINE.md) for Termux stable state
- Simple contract tests for Alerts, Urban, and Offshore modules
- Property-based test templates for parameter validation

### Improved
- Test isolation for environment-dependent modules
- Error handling in parameter computation edge cases

## [1.0.0] - 2026-02-24

### Added
- **Initial Release of AEROTICA Framework**
- Nine-parameter AKE (Atmospheric Kinetic Efficiency) index implementation
  - KED (22%): Kinetic Energy Density
  - TII (16%): Turbulence Intensity Index
  - VSR (14%): Vertical Shear Ratio
  - AOD (12%): Aerosol Optical Depth
  - THD (10%): Thermal Helicity Dynamics
  - PGF (8%): Pressure Gradient Force
  - HCI (7%): Humidity-Convection Interaction
  - ASI (6%): Atmospheric Stability Integration
  - LRC (5%): Local Roughness Coefficient

### Core Infrastructure
- Physics-Informed Neural Network (PINN) architecture
  - Velocity Network: U(x,y,z,t) → (u,v,w)
  - Pressure Network: P(x,y,z,t) → p
  - Temperature Network: T(x,y,z,t) → θ
  - 8-layer backbone with 512 neurons
  - Fourier feature embeddings for multi-scale resolution

### Applications
- Gust pre-alerting engine with 4-6 minute lead time
- Building-integrated wind energy assessment
- Offshore wind farm optimization module
- Real-time wind field mapping at 30-second resolution

### Testing & Validation
- **62 comprehensive parameter tests** (100% passing)
  - KED: 10 tests
  - TII: 10 tests
  - VSR: 12 tests
  - Comprehensive parameters: 30 tests
- **7 simple module tests** for environment-constrained components
  - Alerts: 2 tests
  - Urban: 2 tests
  - Offshore: 3 tests
- Test coverage for edge cases and error handling

### Documentation
- Complete API reference
- Installation guides for multiple platforms
- Parameter-level mathematical documentation
- PINN architecture specifications
- Deployment guides (Docker, cloud, local)
- Case study templates (Tokyo, Brest, Edinburgh)

### Performance Metrics
- AKE classification accuracy: 96.2%
- Gust timing precision: ±28 seconds
- PINN inference: < 90 seconds per site
- Urban bias correction: 18.7%
- Offshore wake model improvement: 34%

### Validation Dataset
- 3,412 station-years from 24 national networks
- 35 countries across 6 climate zones
- 1,247 severe wind events recorded
- 47 stations in validation campaign

### Infrastructure
- Complete Python package structure
- CLI interface with 10+ commands
- FastAPI REST API
- Streamlit dashboard
- Docker support
- CI/CD pipeline (GitLab CI)
- Comprehensive configuration system

### Developer Tools
- Pre-commit hooks for code quality
- Black formatting
- Ruff linting
- MyPy type checking
- Pytest with coverage reporting
- MkDocs for documentation

## [0.9.0] - 2026-01-15

### Added
- Beta release
- Core parameter modules implementation
- Basic PINN training pipeline
- Initial validation framework
- Example notebooks and tutorials

### Changed
- Refactored parameter inheritance structure
- Improved error handling in data preprocessing

### Fixed
- Memory leak in PINN inference
- Numerical stability in THD computation
- Edge cases in wind rose statistics

## [0.8.0] - 2025-12-01

### Added
- Alpha release for internal testing
- Basic AKE composite index
- Simple wake model implementation
- Preliminary documentation

---

## 📊 Test Status Summary (as of 2026-02-24)

### Termux Stable Baseline
```

✅ Parameter Tests: 62/62 passing
✅ Simple Module Tests: 7/7 passing
✅ Total: 69/69 tests passing

```

### Environment Requirements
| Module | Status | Requirements |
|--------|--------|--------------|
| Core Parameters | ✅ Full | numpy, pandas |
| Alerts (full) | ⚠️ Needs full env | PyTorch |
| Urban (full) | ⚠️ Needs full env | rasterio, GDAL |
| Offshore (full) | ⚠️ Needs full env | scipy, matplotlib |
| Integration | ⚠️ Needs full env | All libraries |

---

## 🔖 Version Tags
- `v1.0.0` - Production release
- `baseline-termux-stable` - Stable Termux baseline (69 tests passing)
- `v0.9.0` - Beta release
- `v0.8.0` - Alpha release

---

*For detailed information about each release, see the [GitLab repository](https://gitlab.com/gitdeeper07/aerotica)
