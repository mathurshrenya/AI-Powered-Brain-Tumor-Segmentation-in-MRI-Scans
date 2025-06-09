"""
3D U-Net model implementation with residual connections and instance normalization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch.amp import autocast
from pathlib import Path

class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm3d(out_channels)
        
        # Residual connection
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):
        identity = self.residual(x)
        
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out += identity
        return out

class UNet3D(nn.Module):
    """3D U-Net architecture with residual connections."""
    def __init__(self, in_channels=4, n_classes=4, base_filters=8):
        super().__init__()
        print(f"Initializing UNet3D with:")
        print(f"  in_channels: {in_channels}")
        print(f"  n_classes: {n_classes}")
        print(f"  base_filters: {base_filters}")
        
        # Encoder
        self.enc1 = ResidualBlock(in_channels, base_filters)
        self.enc2 = ResidualBlock(base_filters, base_filters * 2)
        self.enc3 = ResidualBlock(base_filters * 2, base_filters * 4)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(base_filters * 4, base_filters * 8)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_filters * 8, base_filters * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_filters * 4, base_filters * 2)
        
        self.upconv1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_filters * 2, base_filters)
        
        self.final_conv = nn.Conv3d(base_filters, n_classes, kernel_size=1)
        
        # Deep supervision
        self.deep3 = nn.Conv3d(base_filters * 4, n_classes, kernel_size=1)
        self.deep2 = nn.Conv3d(base_filters * 2, n_classes, kernel_size=1)

    def debug_shape(self, x: torch.Tensor, name: str) -> None:
        """Print shape of tensor for debugging."""
        print(f"Shape of {name}: {x.shape}")

    @autocast(device_type='cuda', enabled=torch.cuda.is_available())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input
        self.debug_shape(x, "input")
        
        # Encoder
        enc1 = self.enc1(x)
        self.debug_shape(enc1, "enc1")
        
        enc2 = self.enc2(F.max_pool3d(enc1, 2))
        self.debug_shape(enc2, "enc2")
        
        enc3 = self.enc3(F.max_pool3d(enc2, 2))
        self.debug_shape(enc3, "enc3")
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool3d(enc3, 2))
        self.debug_shape(bottleneck, "bottleneck")
        
        # Decoder with deep supervision
        dec3 = self.upconv3(bottleneck, output_size=enc3.shape)
        self.debug_shape(dec3, "dec3 before concat")
        self.debug_shape(enc3, "enc3 for concat")
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        deep3 = self.deep3(dec3)
        self.debug_shape(dec3, "dec3 after conv")
        
        dec2 = self.upconv2(dec3, output_size=enc2.shape)
        self.debug_shape(dec2, "dec2 before concat")
        self.debug_shape(enc2, "enc2 for concat")
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        deep2 = self.deep2(dec2)
        self.debug_shape(dec2, "dec2 after conv")
        
        dec1 = self.upconv1(dec2, output_size=enc1.shape)
        self.debug_shape(dec1, "dec1 before concat")
        self.debug_shape(enc1, "enc1 for concat")
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        self.debug_shape(dec1, "dec1 after conv")
        
        # Final 1x1 convolution
        out = self.final_conv(dec1)
        self.debug_shape(out, "output")
        
        if self.training:
            # During training return deep supervision outputs
            return out, F.interpolate(deep3, size=out.shape[2:]), F.interpolate(deep2, size=out.shape[2:])
        else:
            return out

def get_model(device='cuda'):
    """Create and initialize the model."""
    model = UNet3D(in_channels=4, n_classes=4, base_filters=8)
    model = model.to(device).float()  # Ensure float32
    return model

def save_model(model: nn.Module,
             optimizer: torch.optim.Optimizer,
             epoch: int,
             output_dir: str,
             is_best: bool = False) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch number
        output_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    if is_best:
        torch.save(checkpoint, output_dir / 'model_best.pth')
    else:
        torch.save(checkpoint, output_dir / 'model_latest.pth')

def load_model(model: nn.Module,
              path: str,
              optimizer: torch.optim.Optimizer = None) -> Tuple[nn.Module, dict]:
    """Load model checkpoint."""
    device = next(model.parameters()).device
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, checkpoint 