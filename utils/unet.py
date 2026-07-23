"""
UNet architectures for semantic segmentation of mineral images.

This module implements three variants of UNet for handling scale-variant microscopy images:
- UNet_1: Standard UNet without scale conditioning
- UNet_2: Dual UNet with routing based on pixel scale threshold
- UNet_3: UNet with continuous scale conditioning at bottleneck
"""

import torch
import torch.nn as nn
from typing import Optional


class UNet_1(nn.Module):
    """
    Standard UNet architecture for semantic segmentation.
    
    This is a baseline model that does not use scale information.
    
    Args:
        in_channels: Number of input channels (typically 1 for grayscale)
        out_channels: Number of output classes
        dropout: Dropout probability (default: 0.3)
    """
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(UNet_1, self).__init__()
        
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x, scale=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            scale: Not used in this model (for API compatibility)
            
        Returns:
            Output logits of shape [B, out_channels, H, W]
        """
        x1 = self.conv1(x)
        x2 = self.pool1(x1)

        x3 = self.conv2(x2)
        x4 = self.pool2(x3)

        x5 = self.conv3(x4)
        x6 = self.pool3(x5)
        x7 = self.conv4(x6)

        x8 = self.upconv3(x7)
        x9 = torch.cat([x5, x8], dim=1)
        x10 = self.conv5(x9)

        x11 = self.upconv2(x10)
        x12 = torch.cat([x3, x11], dim=1)
        x13 = self.conv6(x12)

        x14 = self.upconv1(x13)
        x15 = torch.cat([x1, x14], dim=1)

        output = self.conv7(x15)
        return output


class UNet_2(nn.Module):
    """
    Dual UNet with routing based on pixel scale threshold.
    
    This model uses two separate UNet_1 instances: one for large-scale images
    (pixel size > threshold) and one for small-scale images (pixel size <= threshold).
    Samples are routed to the appropriate UNet based on their pixel scale.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        dropout: Dropout probability (default: 0.3)
        threshold_default: Pixel scale threshold in micrometers (default: 1.8)
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3, threshold_default: float = 1.8):
        super().__init__()
        self.large_unet = UNet_1(in_channels, out_channels, dropout=dropout)
        self.small_unet = UNet_1(in_channels, out_channels, dropout=dropout)
        self.threshold_default = float(threshold_default)
        self.out_channels = int(out_channels)

    def set_threshold(self, value: float):
        """Set the pixel scale threshold for routing."""
        self.threshold_default = float(value)

    @torch.no_grad()
    def route_counts(self, scale, threshold: Optional[float] = None):
        """
        Debug utility: return counts of samples routed to large/small UNets.
        
        Args:
            scale: Pixel scale values
            threshold: Optional threshold override
            
        Returns:
            Tuple of (n_large, n_small) counts
        """
        th = self.threshold_default if threshold is None else float(threshold)
        s = scale if torch.is_tensor(scale) else torch.tensor(scale)
        if s.dim() == 0:
            s = s.view(1)
        if s.dim() == 2 and s.size(1) == 1:
            s = s.view(-1)
        if s.dim() == 3 and s.size(1) == 1 and s.size(2) == 1:
            s = s.view(-1)
        n_large = int((s > th).sum().item())
        n_small = s.numel() - n_large
        return n_large, n_small

    def _prepare_scale(self, scale, B, device, dtype):
        """Prepare scale tensor to shape (B,)."""
        if scale is None:
            raise ValueError("UNet_2 requires `scale` (per-sample pixel size).")
        if not torch.is_tensor(scale):
            s = torch.tensor(scale, dtype=dtype, device=device)
        else:
            s = scale.to(device=device, dtype=dtype)
        # Normalize to (B,)
        if s.dim() == 0:
            s = s.view(1).expand(B)
        elif s.dim() == 2 and s.size(1) == 1:
            s = s.view(-1)
        elif s.dim() == 3 and s.size(1) == 1 and s.size(2) == 1:
            s = s.view(-1)
        if s.dim() != 1 or s.size(0) != B:
            raise ValueError(f"`scale` must have shape (B,) matching batch size. Got {tuple(s.size())}, B={B}")
        return s

    def forward(self, x: torch.Tensor, scale=None, threshold: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass with routing based on pixel scale.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            scale: Pixel scale per sample, shape (B,) or compatible
            threshold: Optional threshold override
            
        Returns:
            Output logits of shape [B, out_channels, H, W]
        """
        B, _, H, W = x.shape
        th = self.threshold_default if threshold is None else float(threshold)

        # Prepare scale to (B,)
        s = self._prepare_scale(scale, B, x.device, x.dtype)

        # Routing mask (using '>' to match dataset split criterion)
        mask_large = (s > th)
        mask_small = ~mask_large

        # Output buffer for entire batch
        out = x.new_zeros((B, self.out_channels, H, W))

        # Large-scale path
        if mask_large.any():
            idx_large = mask_large.nonzero(as_tuple=True)[0]
            x_large = x.index_select(0, idx_large)
            out_large = self.large_unet(x_large)
            out.index_copy_(0, idx_large, out_large)

        # Small-scale path
        if mask_small.any():
            idx_small = mask_small.nonzero(as_tuple=True)[0]
            x_small = x.index_select(0, idx_small)
            out_small = self.small_unet(x_small)
            out.index_copy_(0, idx_small, out_small)

        return out


class UNet_3(nn.Module):
    """
    UNet with continuous scale conditioning at bottleneck.
    
    This model incorporates pixel scale information by applying learnable
    scale-dependent modulation at the bottleneck layer.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        dropout: Dropout probability (default: 0.3)
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3):
        super(UNet_3, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

        # Continuous scale parameters (learnable)
        self.scale_weight = nn.Parameter(torch.ones(1))
        self.scale_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, scale=None) -> torch.Tensor:
        """
        Forward pass with continuous scale conditioning.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            scale: Pixel scale per sample, shape (B,) or compatible.
                   Applied at bottleneck layer (x6).
                   
        Returns:
            Output logits of shape [B, out_channels, H, W]
        """
        # Encoder
        x1 = self.conv1(x)
        x2 = self.pool1(x1)

        x3 = self.conv2(x2)
        x4 = self.pool2(x3)

        x5 = self.conv3(x4)
        x6 = self.pool3(x5)

        # Continuous scaling at bottleneck
        if scale is None:
            raise ValueError("UNet_3 expects `scale` but got None. Pass a float or a (B,) tensor.")
        # Convert to tensor on same device/dtype
        if not torch.is_tensor(scale):
            scale = torch.tensor(scale, dtype=x6.dtype, device=x6.device)
        else:
            scale = scale.to(device=x6.device, dtype=x6.dtype)

        # Make shape (B,1,1,1) for broadcasting
        if scale.dim() == 0:
            scale = scale.view(1).expand(x6.size(0))
        elif scale.dim() == 1 and scale.size(0) != x6.size(0):
            raise ValueError(f"Scale length {scale.size(0)} must match batch size {x6.size(0)}.")
        scale = scale.view(-1, 1, 1, 1)

        adjusted_scale = self.scale_weight * scale + self.scale_bias
        x6 = x6 * adjusted_scale

        # Bottleneck
        x7 = self.conv4(x6)

        # Decoder
        x8 = self.upconv3(x7)
        x9 = torch.cat([x5, x8], dim=1)
        x10 = self.conv5(x9)

        x11 = self.upconv2(x10)
        x12 = torch.cat([x3, x11], dim=1)
        x13 = self.conv6(x12)

        x14 = self.upconv1(x13)
        x15 = torch.cat([x1, x14], dim=1)

        out = self.conv7(x15)
        return out
