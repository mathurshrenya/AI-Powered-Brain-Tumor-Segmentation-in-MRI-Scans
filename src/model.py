"""
Model architecture definition for brain tumor segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class ConvBlock(nn.Module):
    """Double convolution block."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNet3D(nn.Module):
    """3D U-Net architecture for medical image segmentation."""
    def __init__(self, in_channels: int = 4, n_classes: int = 4, base_filters: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_filters)
        self.enc2 = ConvBlock(base_filters, base_filters*2)
        self.enc3 = ConvBlock(base_filters*2, base_filters*4)
        self.enc4 = ConvBlock(base_filters*4, base_filters*8)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_filters*8, base_filters*16)
        
        # Decoder
        self.dec4 = ConvBlock(base_filters*16, base_filters*8)
        self.dec3 = ConvBlock(base_filters*8, base_filters*4)
        self.dec2 = ConvBlock(base_filters*4, base_filters*2)
        self.dec1 = ConvBlock(base_filters*2, base_filters)
        
        # Final layer
        self.final = nn.Conv3d(base_filters, n_classes, kernel_size=1)
        
        # Max pooling
        self.pool = nn.MaxPool3d(2)
        
        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.dec4(torch.cat([self.up(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up(dec2), enc1], dim=1))
        
        return self.final(dec1)

def get_model(in_channels: int = 4,
             n_classes: int = 4,
             base_filters: int = 16,
             device: str = 'cuda') -> UNet3D:
    """Initialize and return the U-Net model."""
    model = UNet3D(in_channels=in_channels,
                   n_classes=n_classes,
                   base_filters=base_filters)
    return model.to(device)

def save_model(model: nn.Module,
              epoch: int,
              optimizer: torch.optim.Optimizer,
              loss: float,
              path: str) -> None:
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_model(path: str,
              model: nn.Module,
              optimizer: torch.optim.Optimizer = None) -> Tuple[nn.Module, dict]:
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, checkpoint 