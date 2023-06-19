import torch
import torch.nn as nn

dropout = 0.3

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            # Input: (batch_size, in_channels, 256, 256)  
            # Output: (batch_size, 128, 256, 256)
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Input: (batch_size, 128, 256, 256)
            # Output: (batch_size, 128, 256, 256)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        # Input: (batch_size, 128, 256, 256)  
        # Output: (batch_size, 128, 128, 128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            # Input: (batch_size, 128, 128, 128) 
            # Output: (batch_size, 256, 128, 128)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Input: (batch_size, 256, 128, 128)
            # Output: (batch_size, 256, 128, 128)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        # Input: (batch_size, 256, 128, 128)
        # Output: (batch_size, 256, 64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            # Input: (batch_size, 256, 64, 64)
            # Output: (batch_size, 512, 64, 64)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Input: ((batch_size, 512, 64, 64)
            # Output: (batch_size, 512, 64, 64)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        # Input: ((batch_size, 512, 64, 64)
        # Output: (batch_size, 512, 32, 32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Middle
        self.conv4 = nn.Sequential(
            # Input: (batch_size, 512, 32, 32)  
            # Output: (batch_size, 1024, 32, 32)
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # Input: (batch_size, 1024, 32, 32)  
            # Output: (batch_size, 1024, 32, 32)
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        # Decoder
        # Input: (batch_size, 1024, 32, 32)  
        # Output: (batch_size, 512, 64, 64)
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  
        self.conv5 = nn.Sequential(
            # Input: (batch_size, 1024, 64, 64)  
            # Output: (batch_size, 512, 64, 64)
            nn.Conv2d(1024, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Input: (batch_size, 512, 64, 64)  
            # Output: (batch_size, 512, 64, 64)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # Input: (batch_size, 512, 64, 64)  
        # Output: (batch_size, 256, 128, 128)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  
        self.conv6 = nn.Sequential(
            # Input: (batch_size, 512, 128, 128)  
            # Output: (batch_size, 256, 128, 128)
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Input: (batch_size, 256, 128, 128)  
            # Output: (batch_size, 256, 128, 128)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        # Input: (batch_size, 256, 128, 128)  
        # Output: (batch_size, 128, 256, 256)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  
        self.conv7 = nn.Sequential(
            # Input: (batch_size, 256, 256, 256)  
            # Output: (batch_size, 128, 256, 256)
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Input: (batch_size, 128, 256, 256)  
            # Output: (batch_size, 128, 256, 256)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            # Input: (batch_size, 128, 256, 256)  
            # Output: (batch_size, out_channels, 256, 256)
            nn.Conv2d(128, out_channels, kernel_size=1)  
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)

        x3 = self.conv2(x2)
        x4 = self.pool2(x3)

        x5 = self.conv3(x4)
        x6 = self.pool3(x5)

        x7 = self.conv4(x6)

        x8 = self.upconv3(x7)
        x9 = torch.cat([x5, x8], dim=1)  # Skip connection
        x10 = self.conv5(x9)

        x11 = self.upconv2(x10)
        x12 = torch.cat([x3, x11], dim=1)  # Skip connection
        x13 = self.conv6(x12)

        x14 = self.upconv1(x13)
        x15 = torch.cat([x1, x14], dim=1)  # Skip connection

        output = self.conv7(x15)
        # output = torch.softmax(self.conv7(x15), dim=1)

        return output