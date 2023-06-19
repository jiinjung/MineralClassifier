# MineralClassifier
The purpose of this repository is for the code example and visualization for the current research.

## Dataset and Model Description
This current system employs deep learning semantic segmentation models to process the input Scanning Electron Microscopy (SEM) images of lunar mare basalts and corresponding pixel scales, and generated a segmented mineral map as output.
<p align="center">
<img width="700" alt="inputoutput" src="https://github.com/jiinjung/MineralClassifier/assets/87342008/f8f45716-4341-40e7-8645-d60469582e9a">
</p>

Typically, SEM images (or other microscopic images) is given a unique scale, the pixel scale ranging approximately between 0.01 µm/px and 10 µm/px. As this scale variability leads to different patterns or mineral occurrences within the same sample, we employed multi U-Net architectures, that assign images to different U-Net architectures based on their scale.

<p align="center">
<img width="300" alt="arc3-01" src="https://github.com/jiinjung/MineralClassifier/assets/87342008/ffc1c1f0-af7b-45e1-afa6-41299670aa3a">
</p>

The model has been trained on a dataset of 3457 images of Apollo mare basalt SEM images (split into 2605 images for training, 652 for validation, and 200 for testing)

## Performances of the current model

Overall, **the Multi U-Net outperformed the Single U-Net**. Both algorithms effectively captured larger and commonly occurring mineral grains such as _Spinel_, _Pyroxene_, _Plagioclase_, and _Void_, identifiable based on their intensity and shapes. The smaller grains became more apparent for Multi U-Net model (e.g., _FeNi_ grains or _Troilite_). However, mineral groups underrepresented in the dataset due to their minimal occurrences in the actual rocks and the scale of the image (e.g., _Zircon_, _Phosphate_, and _Olivine_) exhibited a Intersection over Union (IoU) scores.

<p align="center">
<img width="700" alt="comparison_poster-01" src="https://github.com/jiinjung/MineralClassifier/assets/87342008/a2a91df0-aced-40e2-b2f3-d116c4a34c32">
</p>

The model gives a pixel accuracy (PA) of 0.90 for the training set, 0.90 for the validation set, and 0.91 for the test set, when applied to the large scale (> 0.25µm/px) image dataset. The confusion matrix for this datset indicated successful identification of _Pyroxene_ and _Void_. However, given that the _Pyroxene_ group is the most abundant mineral in Apollo mare basalt samples, these large scale images often misclassified other minerals as _Pyroxene_. This may be due to blurry boundaries in SEM images, leading to misclassification into the adjacent mineral category. Nonetheless, apart from the _Pyroxene_ column, commonly occurred minerals such as _Spinel_ and _Plagioclase_ were correctly-classified, and their normalized abundances closely matched the actual modal abundances described in the Apollo sample compendium. As for the minerals _FeNi_ and _FeS_, their small size made their allocations in the confusion matrix seem incorrect and leading low IoU values.

As for the small scale image dataset (< 0.25µm/px), the model demonstrated a PA of 0.83 for the training set, 0.86 for the validation set, and 0.88 for the test set. The confusion matrix revealed excellent identification of well-represented groups such as _Void_, _FeNi_, _FeS_, _Spinel_, _Pyroxene_, and _Plagioclase_, with respectable IoU scores. However, _Glass_ in small scale images often blended indistinguishably with _Plagioclase_, making it difficult to be accurately identified.

<p align="center">
<img width="700" alt="confusion_matrix" src="https://github.com/jiinjung/MineralClassifier/assets/87342008/6c0befa3-e965-4828-9e58-8c544fb6e454">
</p>

To improve the detection of these underrepresented groups, the dataset could be expanded to include more of these minerals, and the image scale could be further divided into more than just two categories for the future work.

