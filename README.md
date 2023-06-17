# MineralClassifier
This system employs deep learning semantic segmentation models to process the input Scanning Electron Microscopy (SEM) images of lunar mare basalts and corresponding pixel scales, and generated a segmented mineral map as output.
<p align="center">
<img width="700" alt="inputoutput" src="https://github.com/jiinjung/MineralClassifier/assets/87342008/f8f45716-4341-40e7-8645-d60469582e9a">
</p>
Typically, SEM images (or other microscopic images) is given a unique scale, the pixel scale ranging approximately between 0.01 µm/px and 10 µm/px. As this scale variability leads to different patterns or mineral occurrences within the same sample, we employed multi U-Net architectures, that assign images to different U-Net architectures based on their scale.

<p align="center">
<img width="300" alt="arc3-01" src="https://github.com/jiinjung/MineralClassifier/assets/87342008/ffc1c1f0-af7b-45e1-afa6-41299670aa3a">
</p>

The model has been trained on a dataset of 3457 images of Apollo mare basalt SEM images (split into 2605 images for training, 652 for validation, and 200 for testing)

## Performances of the current model
<p align="center">
<img width="700" alt="comparison_poster-01" src="https://github.com/jiinjung/MineralClassifier/assets/87342008/a2a91df0-aced-40e2-b2f3-d116c4a34c32">
</p>
Overall, the Multi U-Net outperformed the Single U-Net.

<p align="center">
<img width="700" alt="confusion_matrix" src="https://github.com/jiinjung/MineralClassifier/assets/87342008/6c0befa3-e965-4828-9e58-8c544fb6e454">
</p>

