"""Fourier feature embedding for multi-scale spatial representation."""

import torch
import torch.nn as nn
import numpy as np


class FourierFeatureEmbedding(nn.Module):
    """Fourier feature embedding for multi-scale resolution.
    
    Maps input coordinates to Fourier features to help neural networks
    represent high-frequency functions.
    """
    
    def __init__(self,
                 input_dim: int,
                 mapping_size: int,
                 scale: float = 1.0,
                 trainable: bool = False):
        """Initialize Fourier feature embedding.
        
        Args:
            input_dim: Input dimension (typically 4 for x,y,z,t)
            mapping_size: Size of Fourier feature mapping
            scale: Scale factor for random frequencies
            trainable: Whether to make frequencies trainable
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        
        # Random frequency matrix B ~ N(0, scale^2)
        B = torch.randn(input_dim, mapping_size) * scale
        
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature mapping.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Fourier features of shape (batch_size, 2*mapping_size)
        """
        # Project input through random frequencies
        x_proj = 2 * np.pi * x @ self.B  # (batch_size, mapping_size)
        
        # Concatenate sin and cos
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    def get_frequency_bands(self) -> torch.Tensor:
        """Get frequency magnitudes for analysis."""
        return torch.norm(self.B, dim=0)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        return x + self.pe[:, :x.size(1)]


class GaussianRandomFeatures(nn.Module):
    """Gaussian random features for kernel approximation."""
    
    def __init__(self, input_dim: int, output_dim: int, gamma: float = 1.0):
        super().__init__()
        
        self.W = nn.Parameter(
            torch.randn(input_dim, output_dim // 2) * np.sqrt(2 * gamma),
            requires_grad=False
        )
        self.b = nn.Parameter(
            torch.rand(output_dim // 2) * 2 * np.pi,
            requires_grad=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random Fourier features."""
        projection = x @ self.W + self.b
        return torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1) * np.sqrt(2.0 / projection.size(-1))


def get_fourier_embedding(input_dim: int,
                         mapping_size: int,
                         encoding_type: str = 'basic',
                         **kwargs) -> nn.Module:
    """Factory function to get Fourier embedding.
    
    Args:
        input_dim: Input dimension
        mapping_size: Size of mapping
        encoding_type: Type of encoding ('basic', 'positional', 'gaussian')
        **kwargs: Additional arguments
        
    Returns:
        Fourier embedding module
    """
    if encoding_type == 'basic':
        return FourierFeatureEmbedding(
            input_dim, mapping_size,
            scale=kwargs.get('scale', 1.0),
            trainable=kwargs.get('trainable', False)
        )
    elif encoding_type == 'positional':
        return PositionalEncoding(mapping_size)
    elif encoding_type == 'gaussian':
        return GaussianRandomFeatures(
            input_dim, mapping_size,
            gamma=kwargs.get('gamma', 1.0)
        )
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
