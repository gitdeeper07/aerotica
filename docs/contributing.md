# Contributing to AEROTICA

We welcome contributions! Please see our [contributing guidelines](../CONTRIBUTING.md) for details.

## Quick Start

```bash
# Fork and clone
git clone https://gitlab.com/YOUR_USERNAME/aerotica.git
cd aerotica

# Install development dependencies
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/ -v
```

Development Workflow

1. Create an issue describing your changes
2. Fork the repository
3. Create a feature branch
4. Write tests for your changes
5. Submit a merge request

Code Style

· Black for formatting
· Ruff for linting
· MyPy for type checking
· Google-style docstrings

Testing

· Write unit tests for new features
· Maintain test coverage > 80%
· Run integration tests for changes

Documentation

· Update docstrings for new functions
· Add examples for new features
· Update the changelog
