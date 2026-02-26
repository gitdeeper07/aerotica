# PINN Architecture

AEROTICA uses a Physics-Informed Neural Network with three coupled networks:

## Network Structure

```

Input Layer: (x, y, z, t)
↓
Fourier Feature Embedding (512 dims)
↓
8-layer Backbone (512 neurons each)
↓
┌─────────────────┬─────────────────┬─────────────────┐
│ Velocity Net    │ Pressure Net    │ Temperature Net │
│ (u, v, w)       │ (p)             │ (θ)             │
└─────────────────┴─────────────────┴─────────────────┘

```

## Loss Function

```

L_total = α·L_data + β·L_NS + γ·L_BC + δ·L_IC

```

- **L_data**: Fit to observations
- **L_NS**: Navier-Stokes residuals
- **L_BC**: Boundary conditions
- **L_IC**: Initial conditions

## Training

- 72 GPU-hours on 8× NVIDIA A100
- 18 months of historical data
- LES benchmarks as ground truth

## Inference

- < 2 seconds per forecast
- 50×50×2 km domain
- 10m horizontal resolution
