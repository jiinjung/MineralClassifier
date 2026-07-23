"""
Data loading utilities for semantic segmentation of mineral images.

This module provides dataset classes and utilities for loading SEM images
with corresponding segmentation masks and pixel scale information.
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms


def get_pixel_size(filename):
    """
    Extract pixel size from a text file.
    
    The file should contain a line starting with "pixel_size:" followed by
    a numeric value.
    
    Args:
        filename: Path to the text file containing pixel size information
        
    Returns:
        Pixel size as float (in micrometers), or None if file not found or format invalid
        
    Example:
        >>> pixel_size = get_pixel_size("data/features/image_001.txt")
        >>> print(pixel_size)
        2.38
    """
    try:
        with open(filename, 'r') as f:
            line = f.readline().strip()
            if line.startswith("pixel_size:"):
                return float(line.split(":")[1].strip())
    except FileNotFoundError:
        return None
    return None


def create_segmentation_dataset(input_files, output_files, pixel_size_dict, filename_list):
    """
    Create a PyTorch dataset for semantic segmentation.
    
    Args:
        input_files: List of paths to input images (256×256 grayscale SEM images)
        output_files: List of paths to segmentation mask images (256×256 labeled masks)
        pixel_size_dict: Dictionary mapping image base names to pixel sizes (in micrometers).
                         Keys should match the base names (without extension) of files in input_files.
        filename_list: List of filenames corresponding to input_files
        
    Returns:
        A SegmentationDataset instance
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.int64))),
    ])

    class SegmentationDataset(data.Dataset):
        """
        Dataset for semantic segmentation of mineral images.
        
        Returns tuples of (input_image, output_mask, pixel_size, filename).
        """
        def __init__(self, input_files, output_files, pixel_size_dict, filename_list):
            self.input_files = input_files
            self.output_files = output_files
            self.pixel_size_dict = pixel_size_dict
            self.filename_list = filename_list
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.input_files)

        def __getitem__(self, idx):
            input_image = Image.open(self.input_files[idx])
            output_image = Image.open(self.output_files[idx])
            filename = os.path.basename(self.input_files[idx])
            pixel_key = os.path.splitext(filename)[0]
            pixel_size = torch.tensor(
                self.pixel_size_dict.get(pixel_key, -1.0),
                dtype=torch.float32
            )

            if self.transform:
                input_image = self.transform(input_image)
            if self.target_transform:
                output_image = self.target_transform(output_image)

            return input_image, output_image, pixel_size, filename

    return SegmentationDataset(input_files, output_files, pixel_size_dict, filename_list)

