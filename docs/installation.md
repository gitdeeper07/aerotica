# Installation Guide

## System Requirements

- Python 3.8 or higher
- 8GB RAM (16GB recommended)
- 10GB free disk space
- CUDA-capable GPU (optional, for training)

## Quick Install

```bash
pip install aerotica-ake
```

Install from Source

```bash
git clone https://gitlab.com/gitdeeper07/aerotica.git
cd aerotica
pip install -e ".[dev]"
```

Docker Installation

```bash
docker pull registry.gitlab.com/gitdeeper07/aerotica:latest
docker run -p 8000:8000 aerotica:latest
```

Verify Installation

```python
import aerotica
print(aerotica.__version__)
```

