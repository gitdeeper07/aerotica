"""PINN Training Module."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import json
from datetime import datetime


class PINNTrainer:
    """Trainer for Physics-Informed Neural Networks."""
    
    def __init__(self,
                 velocity_net: nn.Module,
                 pressure_net: nn.Module,
                 temperature_net: nn.Module,
                 loss_fn: nn.Module,
                 learning_rate: float = 1e-3,
                 device: str = 'cuda'):
        """Initialize trainer.
        
        Args:
            velocity_net: Velocity network
            pressure_net: Pressure network
            temperature_net: Temperature network
            loss_fn: Loss function
            learning_rate: Learning rate
            device: Device for training
        """
        self.velocity_net = velocity_net.to(device)
        self.pressure_net = pressure_net.to(device)
        self.temperature_net = temperature_net.to(device)
        self.loss_fn = loss_fn
        self.device = device
        
        # Optimizer
        params = list(velocity_net.parameters()) + \
                 list(pressure_net.parameters()) + \
                 list(temperature_net.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
        
        # Training history
        self.history = {
            'epoch': [],
            'loss_total': [],
            'loss_data': [],
            'loss_ns': [],
            'loss_continuity': [],
            'loss_bc': [],
            'loss_ic': [],
            'learning_rate': []
        }
    
    def train_epoch(self,
                   dataloader: DataLoader,
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: DataLoader with training data
            epoch: Current epoch number
            
        Returns:
            Dictionary with average losses
        """
        self.velocity_net.train()
        self.pressure_net.train()
        self.temperature_net.train()
        
        epoch_losses = {
            'total': 0.0,
            'data': 0.0,
            'navier_stokes': 0.0,
            'continuity': 0.0,
            'boundary': 0.0,
            'initial': 0.0
        }
        n_batches = 0
        
        for batch in dataloader:
            # Move data to device
            coords = batch['coords'].to(self.device)
            
            # Prepare observed data
            observed = {}
            if 'observed_velocity' in batch:
                observed['velocity'] = batch['observed_velocity'].to(self.device)
            if 'observed_pressure' in batch:
                observed['pressure'] = batch['observed_pressure'].to(self.device)
            if 'observed_temperature' in batch:
                observed['temperature'] = batch['observed_temperature'].to(self.device)
            
            # Boundary conditions
            boundary_coords = None
            boundary_values = None
            if 'boundary_coords' in batch:
                boundary_coords = batch['boundary_coords'].to(self.device)
                boundary_values = {}
                if 'boundary_velocity' in batch:
                    boundary_values['velocity'] = batch['boundary_velocity'].to(self.device)
                if 'boundary_pressure' in batch:
                    boundary_values['pressure'] = batch['boundary_pressure'].to(self.device)
                if 'boundary_temperature' in batch:
                    boundary_values['temperature'] = batch['boundary_temperature'].to(self.device)
            
            # Initial conditions
            initial_coords = None
            initial_values = None
            if 'initial_coords' in batch:
                initial_coords = batch['initial_coords'].to(self.device)
                initial_values = {}
                if 'initial_velocity' in batch:
                    initial_values['velocity'] = batch['initial_velocity'].to(self.device)
                if 'initial_pressure' in batch:
                    initial_values['pressure'] = batch['initial_pressure'].to(self.device)
                if 'initial_temperature' in batch:
                    initial_values['temperature'] = batch['initial_temperature'].to(self.device)
            
            # Forward pass
            losses = self.loss_fn(
                self.velocity_net,
                self.pressure_net,
                self.temperature_net,
                coords,
                observed if observed else None,
                boundary_coords,
                boundary_values,
                initial_coords,
                initial_values
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.velocity_net.parameters()) +
                list(self.pressure_net.parameters()) +
                list(self.temperature_net.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses.keys():
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            n_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        # Update loss weights adaptively
        self.loss_fn.update_weights(losses)
        
        return epoch_losses
    
    def train(self,
             train_loader: DataLoader,
             val_loader: Optional[DataLoader],
             n_epochs: int,
             save_dir: Optional[Path] = None,
             callback: Optional[Callable] = None):
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            callback: Optional callback function
        """
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader is not None:
                val_losses = self.validate(val_loader)
            else:
                val_losses = train_losses
            
            # Update learning rate
            self.scheduler.step(val_losses['total'])
            
            # Save history
            self.history['epoch'].append(epoch)
            self.history['loss_total'].append(train_losses['total'])
            self.history['loss_data'].append(train_losses['data'])
            self.history['loss_ns'].append(train_losses['navier_stokes'])
            self.history['loss_continuity'].append(train_losses['continuity'])
            self.history['loss_bc'].append(train_losses['boundary'])
            self.history['loss_ic'].append(train_losses['initial'])
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: "
                      f"Loss = {train_losses['total']:.6f}, "
                      f"NS = {train_losses['navier_stokes']:.6f}, "
                      f"Data = {train_losses['data']:.6f}, "
                      f"LR = {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint
            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Save best model
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    self.save_checkpoint(save_dir / 'best_model.pt')
                
                # Regular checkpoint
                if epoch % 100 == 0:
                    self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch}.pt')
            
            # Callback
            if callback is not None:
                callback(epoch, train_losses, val_losses)
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary with validation losses
        """
        self.velocity_net.eval()
        self.pressure_net.eval()
        self.temperature_net.eval()
        
        val_losses = {
            'total': 0.0,
            'data': 0.0,
            'navier_stokes': 0.0,
            'continuity': 0.0,
            'boundary': 0.0,
            'initial': 0.0
        }
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                coords = batch['coords'].to(self.device)
                
                # Prepare data (simplified)
                losses = self.loss_fn(
                    self.velocity_net,
                    self.pressure_net,
                    self.temperature_net,
                    coords,
                    None, None, None, None, None
                )
                
                for key in val_losses.keys():
                    if key in losses:
                        val_losses[key] += losses[key].item()
                
                n_batches += 1
        
        for key in val_losses:
            val_losses[key] /= n_batches
        
        return val_losses
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        checkpoint = {
            'velocity_net': self.velocity_net.state_dict(),
            'pressure_net': self.pressure_net.state_dict(),
            'temperature_net': self.temperature_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'history': self.history,
            'loss_weights': {
                'data': self.loss_fn.weight_data.item(),
                'ns': self.loss_fn.weight_ns.item(),
                'continuity': self.loss_fn.weight_continuity.item(),
                'bc': self.loss_fn.weight_bc.item(),
                'ic': self.loss_fn.weight_ic.item()
            }
        }
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.velocity_net.load_state_dict(checkpoint['velocity_net'])
        self.pressure_net.load_state_dict(checkpoint['pressure_net'])
        self.temperature_net.load_state_dict(checkpoint['temperature_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.history = checkpoint['history']
        
        # Load loss weights
        if 'loss_weights' in checkpoint:
            weights = checkpoint['loss_weights']
            self.loss_fn.weight_data.data = torch.tensor(weights['data'])
            self.loss_fn.weight_ns.data = torch.tensor(weights['ns'])
            self.loss_fn.weight_continuity.data = torch.tensor(weights['continuity'])
            self.loss_fn.weight_bc.data = torch.tensor(weights['bc'])
            self.loss_fn.weight_ic.data = torch.tensor(weights['ic'])
        
        print(f"✅ Checkpoint loaded from {path}")
    
    def save_history(self, path: Path):
        """Save training history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✅ History saved to {path}")
