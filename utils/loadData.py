import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

# load segmentation dataset
class SegmentationDataset(data.Dataset):
    def __init__(self, input_folder, output_folder, transform=None, target_transform=None):
        super().__init__()

        self.input_images = [f for f in sorted(os.listdir(input_folder)) if f.endswith('.jpg')]
        self.output_images = [f for f in sorted(os.listdir(output_folder)) if f.endswith('.jpg')]

        self.input_folder = input_folder
        self.output_folder = output_folder

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, index):
        input_image_path = os.path.join(self.input_folder, self.input_images[index])
        output_image_path = os.path.join(self.output_folder, self.output_images[index])

        input_image = Image.open(input_image_path)
        output_image = Image.open(output_image_path)

        if self.transform:
            input_image = self.transform(input_image)
        if self.target_transform:
            output_image = self.target_transform(output_image)

        return input_image, output_image


def create_segmentation_dataset(input_folder, output_folder):
    # transformation (256 x 256 x 1 (gray scale intensity map))
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    # transformation (256 x 256 (10 classes))
    target_transform = transforms.Compose([
        transforms.Grayscale(),  # Convert image to grayscale
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),  # Scale the tensor values back up to [0, 255]
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.int64))),
    ])

    # Create the dataset
    dataset = SegmentationDataset(
        input_folder, 
        output_folder, 
        transform=transform, 
        target_transform=target_transform
    )
    
    return dataset
